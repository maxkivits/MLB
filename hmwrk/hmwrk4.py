# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:39:36 2019

@author: Max
"""

import numpy as np

maxexp=40 # Python floats cannot represent exp(maxexp) and larger


#weights
w11 = [1,1,-1]
w21 = [0,-1,-1]
w31 = [-1,0,1]
w41 = [1,1,1]

w1 = [w11]
w1.append(w21)
w1.append(w31)
w1.append(w41)

w12 = [[1,-1,1,-1,1]]

x = [1,1,1]
t = 0



Layer = [2,4,1]


'sigmoid activation fnction + gradient'
def logitvgf(a):
    if a<-40.:
        s=0.
    if a>40.:
        s=1
    else: 
        s = 1./(1.+np.exp(-a))
    return s,s*(1.-s)

'output of a single neuron'
def activation(x,w):
    a = np.dot(w,x)
    return a


'Activation of a single neuron' 
def neur_out(a):
    z = logitvgf(a)[0]
    return z

'single forward pass through a layer'
def forward(Layer,x,w):
    
    z =[]
    
    for i in range(Layer):
        z1 = neur_out(activation(np.transpose(w[i]),x))
        z.append(z1)
    
    return z

'add bias to input vector'
def addBias(xb):
    xb.insert(0,1)
    return xb


'delta of output neuron'
def deltaOut(y,t,a):
    d = -((t/y[0])-((1-t)/(1-y[0])))*logitvgf(a)[1]
    return d



xd=forward(Layer[1],x,w1)
print(xd)
addBias(xd)
yd=forward(Layer[2],xd,w12)
print(yd)


