# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:43:23 2019

@author: Max
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import copy


def gradDesc(w, errorfn, gradfn,  eta=1e-3, verbose=True, maxit=100):
    """Find a minimum of a function using Gradient Descent
    
    Inputs:
        w: Initialisation of the weights
        errorfn: the function of w to minimise
        gradfn:  The gradient of that function w.r.t. w
        eta:     The initial value of the step size
        verbose: Whether to print out progress information
        
    Outputs:
        - The value of w at which the minimum is achieved
        - a list errors for the different iterations
        - a list of the corresponding time steps
    
    """
    err = errorfn(w)             # Compute the error function
    errs = np.zeros(maxit)       # Keep track of the error function
    ts = np.zeros(maxit)         # Keep track of time stamps
    start = time.time()

    newError = errorfn(w) + 1.
    for n in range(maxit):
        grad = gradfn(w)
        if verbose:
            print("##", n, "err:", newError, "eta:", eta, "w:", w)
            print("  Gradient:", grad)
            print("  FD grad: ", gradfd(w,errorfn))
        pw = copy.copy(w)
        w -= grad * eta     # Update the weights
        
        pastError = newError
        newError = errorfn(w)
        while eta > 0. and pastError - newError < 0: # If the error increases, eta is too large
            eta /= 2.0                                # halve it, and
            w = pw - eta * grad                      # try again from the original value of the weights
            newError = errorfn(w)
        else:                                        # If the error goes down,
            eta *= 1.2                               # try to increase eta a little, to speed up things.
                
        if pastError-newError < 1e-5:                # If we couldn't decrease the error anymore, 
            return w, errs[:n], ts[:n]               # just give up
 
        errs[n] = newError                           # Keep track of how the errors evolved
        ts[n] = time.time()-start

    return w, errs, ts


def sigma(a):
    """Numerically stable implementation of the logistic function"""
    return 0. if a<-40. else 1. if a>40. else 1./(1.+np.exp(-a))
sigma = np.vectorize(sigma) # make sigma work elementwise on vectors too

def gradSigma(a):
    """Return the derivative of the logistic function w.r.t. its function parameter"""
    s = sigma(a)
    return s*(1.-s)

#ANSQ3
def sse(output, target):
    return (output-target)**2
   
def gradSse(output,target):
    return -2*(output-target)

def neuron(weights,inputs):
    y = np.dot(weights,inputs)
    return sigma(y)
 
def gradNeuron(weights,inputs):
    a = np.dot(weights,inputs)
    GradZ = gradSigma(a)
    return [i*GradZ for i in inputs]
#/ANSQ3
    
#ANSQ4
import random as rd

delta = 1e-5

#Finit difference method to calc derivative
def finDiffGrad(weights,inputs,delta):
   
    for iFD in range(len(weights)):
        Fweights = weights[iFD]+delta
        Bweights = weights[iFD]-delta
       
    forwardGrad = neuron(Fweights,inputs)
    backwardGrad = neuron(Bweights,inputs)    
    Grad = (forwardGrad-backwardGrad)/(2*delta)
    return Grad

def verifyGradient(n,verbose):
   
    Diff=FDGrad=TestGrad=[]
   
    for i in range(n):
        #random weights inputs generation
        weights = [rd.uniform(0, 1) for _ in range(0, 5)]
        inputs = [rd.uniform(0, 1) for _ in range(0, 5)]
     
        FDGradi = finDiffGrad(weights,inputs,delta)
        TestGradi = gradNeuron(weights,inputs)
        Diffi = FDGradi - TestGradi
       
        FDGrad.append(FDGradi)
        TestGrad.append(TestGradi)
        Diff.append(Diffi)
       
        if verbose == 1:
            print('FD:       ',FDGradi)
            print('Analytical',np.round(TestGradi,8))
            print('DIFF      ',Diffi,'\n')
       
    meanDiff = np.mean(Diff)
    varDiff = np.var(Diff)
    if verbose == 1:
        print('MeanDiff',meanDiff)
        print('Variance',varDiff)
    return meanDiff, varDiff

verifyGradient(100,0)
#/ANSQ4

data = np.load("curve.npz")
train = np.array([data['train'][:,0]]).T
traint = data['train'][:,1]

trainones = np.ones((len(train),2))
trainones[:,1] = train[:,0]

def errorfunc(w):
    error = 0
    for i in range(len(trainones)):
        a = np.dot(w,trainones[i])
        error += 0.5*(sigma(a)-traint[i])**2
    return error

def graderrorfunc(w):
   graderror = 0
   
   for i in range(len(trainones)):
       graderror += gradNeuron(w,trainones[i])
       
   return graderror

# Test errorfunc and graderrorfunc
w = [0,0]
print("Error is:",errorfunc(w))
print("Gradient error is:",graderrorfunc(w))

[wfinal, errs, ts] = gradDesc(w, errorfunc, graderrorfunc,1e-3,True,100)
#commented out the gradfd function, ask what how why this is and why it's not defined

plt.plot(ts,errs)
plt.xlabel('Iterations or epochs?')  #X-axis label
plt.ylabel('Error') #Y-axis label 
plt.title("Training of network, evolution of the error") #Chart title. 

axis = np.linspace(0,1,10)

output = np.zeros(len(trainones))
for i in range (len(trainones)):
    output[i] = neuron(wfinal,trainones[i])

plt.figure()
plt.scatter(train,traint)
plt.plot(axis,output)
plt.xlabel('Attribute 1')  #X-axis label
plt.ylabel('Attribute 2') #Y-axis label 
plt.title("Neuron implemented function overlayed with the training data") #Chart title. 
