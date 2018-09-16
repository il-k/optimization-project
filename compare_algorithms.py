#! /usr/bin/python

from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
import os,sys

#omit warnings in minimize() about unneeded hessians 
import warnings
warnings.filterwarnings("ignore")

#remove old plots
os.system("rm eps=*")
os.system("rm trajectories")

#setup
methods = ['BFGS','CG','trust-exact']
#epsilon = [1e-01,1e-02,1e-03,1e-04,1e-05,1e-06]
epsilon = [(0.1)**(i) for i in range(1,5)]
ndim    = [2**i for i in range(3,11)]
maxdim  = max(ndim)
times   = np.zeros((len(ndim), len(epsilon), len(methods)), dtype=float)
quasi_infty=1e+05
testing = False
dim = 231 # for 2d trajectories
if dim > maxdim:
    sys.exit("ERROR: dim should be smaller than maxdim!")


#dealing with command line arguments and constructing diag[]
if len(sys.argv) == 1:
    sys.exit("Usage: at least one argument required to determine the Matrix Q.")
else:
    if sys.argv[1] in ['1','2','3']:
        if sys.argv[1] == '1':
            # fibonacci
            diag = [0,1]
            for i in range(2,maxdim+1):
                diag.append(diag[i-2] + diag[i-1])
            diag.remove(0)
        elif sys.argv[1] == '2':
            # first alternative
            # 1st value is log(2), bacause log(1)=0 causes problems
            diag = [np.log(i+1) for i in range(1,maxdim+1)]
        elif sys.argv[1] == '3':
            # second alternative
            diag = [1. + 1./i for i in range(1,maxdim+1)]
    else:
        sys.exit("Usage: first argument should be either 1, 2 or 3.")
    if len(sys.argv) > 2:
        if sys.argv[2] == "testing":
            testing = True
        else:
            sys.exit("Usage: second argument should be either 'testing' or left empty.")


# function to be optimized - quadratic form  xQx/2 + bx
# with q_ij = diag(i) if i == j else 0 and b_i = i^(-1)
def func(x):
    value = 0.
    for i in range(len(x)):
        value += diag[i] * x[i]**2
    value /= 2
    for i in range(len(x)):
        value += (i+1)**(-1) * x[i]
    return value

# gradient Qx + b
def grad(x):
    value = []
    for i in range(len(x)):
        value.append(diag[i] * x[i] + (i+1)**(-1))
    return np.array(value) # as required by scipy (array_like / - operation)
    
# curvature matrix / hessian Q
def hesse(x):
    value = []
    for i in range(len(x)):
        value.append([])
        for j in range(len(x)):
            if i == j:
                value[i].append(diag[i])
            else:   
                value[i].append(0)
    return np.array(value) # as required by scipy (dtype information)    


# verify results with an easy example only if testing is enabled
if testing == True:
    dim = 3    
    x = dim*[1.]
    print "Value of f at ",x,":"
    print func(x)
    print "Value of grad(f) at ",x,":"
    print grad(x)
    print "Value of hesse(f) at ",x,":"    
    print hesse(x)
    print "Results:"
    for m in range(len(methods)):
        result = minimize(func,x,method=methods[m],jac=grad,hess=hesse,tol=min(epsilon))
        print result.x,"\t:",methods[m]
    sys.exit(0)


# measurement
for i in range(len(ndim)):
    # define arbitrary starting point
    x = ndim[i]*[179]
    for  j in range(len(epsilon)):
        for m in range(len(methods)):
            tmp = timer()
            result = minimize(func,x,method=methods[m],jac=grad,hess=hesse,tol=epsilon[j],options={'maxiter':quasi_infty,'disp':False})
            tmp = timer() - tmp
            times[i][j][m] = tmp
# plot results time(dim) for every eps
for i in range(len(epsilon)):
    fig, ax =   plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    for m in range(len(methods)):  
        ax.plot(ndim,times[:,i,m], label = "method="+str(methods[m]))
    ax.set_xlabel("Number of Dimensions")
    ax.set_ylabel("elapsed time in seconds")
    ax.legend(loc='best')
    fig.suptitle("epsilon=" + str(epsilon[i]))
    fig.savefig(fname="eps="+str(epsilon[i]), format="png", dpi=400)



# 2d plot of the trajectory via projection onto the 01-coordinate plane, fixed eps=eps_const
plt.figure()
x = [np.log(i+1) for i in range(dim)]
eps_const = 1e-06
for m in range(len(methods)):
    # callback function for intermediate values
    niter_cb = 0
    interm_results = []
    def cback(xk):
        global niter_cb
        global interm_results
        interm_results.append(xk)
        niter_cb += 1
    result = minimize(func,x,method=methods[m],jac=grad,hess=hesse,tol=eps_const,callback=cback,options={'maxiter':quasi_infty,'disp':False})
    listx = []
    listy = []
    listx.append(x[0])
    listy.append(x[1])
    for i in range(niter_cb):
        listx.append(interm_results[i][0])
        listy.append(interm_results[i][1]) 
    plt.plot(listx,listy, label = "method="+str(methods[m]))
    #plt.scatter(listx,listy, marker='x')

plt.ylabel("Y")
plt.xlabel("X")
plt.legend(loc='best')
plt.title("Trajectories in the 01-plane; epsilon="+str(eps_const)+"; dimensions:"+str(dim))
plt.savefig(fname="trajectories", format="png", dpi=400)



