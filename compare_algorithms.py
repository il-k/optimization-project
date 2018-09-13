#! /usr/bin/python

from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
import os
import cmath

#omit warnings about unneeded hessians
import warnings
warnings.filterwarnings("ignore")

#remove old plots
os.system("rm eps=*")
os.system("rm trajectories")

#setup
methods = ['BFGS','Newton-CG','CG', 'trust-exact']
#epsilon = [1e-01,1e-02,1e-03,1e-04,1e-05,1e-06]
epsilon = [(0.1)**(i) for i in range(1,9)]
ndim    = [2**i for i in range(3,11)]
maxdim  = ndim[len(ndim)-1] # hier max() verwenden!
times   = np.zeros((len(ndim), len(epsilon), len(methods)), dtype=float)
quasi_infty=1e+05

if ndim[0] != 2:
    print "ERROR: smallest dimension should be 2!"
#construct table with values of the fibonacci series beforehand
#fib = [0,1]
#for i in range(2,maxdim+1):
#    fib.append(fib[i-2] + fib[i-1])
#test new Matrix Q
fib = [cmath.log(i) for i in range(1,maxdim+2)]

# function to be optimized - quadratic form  xQx/2 + bx
# with q_ij = fib(i) if i == j else 0 and b_i = i^(-1)
def func(x):
    value = 0.
    for i in range(len(x)):
        value += fib[i+1] * x[i]**2
    value /= 2
    for i in range(len(x)):
        value += (i+1)**(-1) * x[i]
    return value

# gradient Qx + b
def grad(x):
    value = []
    for i in range(len(x)):
        value.append(fib[i+1] * x[i] + (i+1)**(-1))
    return np.array(value) # as required by scipy (array_like / - operation)
    
# curvature matrix / hessian Q
def hesse(x):
    value = []
    for i in range(len(x)):
        value.append([])
        for j in range(len(x)):
            if i == j:
                value[i].append(fib[i+1])
            else:   
                value[i].append(0)
    return np.array(value) # as required by scipy (dtype information)    

#measurement
for i in range(len(ndim)):
    # define arbitrary starting point
    x = ndim[i]*[1.79]
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
    fig.savefig(fname="eps="+str(epsilon[i]), format="png")


'''
# beispiel in 2dim (ndim[0]) mit callback, plotte fortschritt, eps = 1e-04
plt.figure()
x = [1.792+i for i in range(5)]
eps_c = 1e-05
for m in range(len(methods)):
    # callback function for intermediate values
    niter_cb = 0
    interm_results = []
    def cback(xk):
        global ninit_cb
        global interm_results
        interm_results.append(xk)
        niter_cb += 1
    result = minimize(func,x,method=methods[m],jac=grad,hess=hesse,tol=eps_c,callback=cback)
    listx = listy = []
    for i in range(niter_cb):
        listx.append(interm_result[i][0])
        listy.append(interm_result[i][1]) 
    plt.plot(listx,listy, label = "method="+str(methods[m]))


plt.ylabel("Y")
plt.xlabel("X")
plt.legend(loc='best')
plt.title("trajectory; epsilon=" + str(eps_c))
plt.savefig(fname="trajectories", format="png")
'''

'''
# testing
x = ndim[0]*[1.]
#print func(x)
#print grad(x)
#print hesse(x)
# solution to x+1=0,y+1/2=0,2z+1/3=0 <-> Ax+b=0
# result = -1,-1/2,-1/6
result = minimize(func,x,method='BFGS',jac=grad,tol=eps[0])
print result.x
result = minimize(func,x,method='Newton-CG',jac=grad,hess=hesse,tol=eps[0])
print result.x
result = minimize(func,x,method='CG',jac=grad,tol=eps[0])
print result.x
# trust-exact macht probleme ab ndim=94:  'long' object has no attribute 'sqrt'
#result = minimize(func,x,method='trust-exact',jac=grad,hess=hesse,tol=eps[0])
#print result.x 
'''

