# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:57:54 2019

@author: nagar
"""
from sensitivity_analysis import sigmoid
import pandas as pd

def sensitive_neuron(inpt,w,b,y):
    '''
    
    Parameters
    ----------
    inpt : 2-dimensional array
        This is the input provided to network.
    w : 2-dimensional array
        Weight matrix obtained after training the network for the entire network.
    b : 2-dimensional array
        Bias matrix obtained after training the network for the entire network.
    y : 1-dimensional array
        The desired output vector for given input.

    Returns
    -------
    neuron : int
        The index of neuron from input layer (1st layer) which is most sensitive to given input.

    '''
    a1,a2,a3,z1,z2,z3=[],[],[],[],[],[]
    for i in range(len(b[0])):
        z=sum(inpt*w[0][:,i])+b[0][i]
        z1.append(z)
        a=sigmoid.sigmoid(z)
        a1.append(a)  
    for i in range(len(b[1])):
        z=sum(a1[i]*w[1][:,i])+b[1][i]
        z2.append(z)
        a=sigmoid.sigmoid(z)
        a2.append(a)    
    for i in range(len(b[2])):
        z=sum(a2[i]*w[2][:,i])+b[2][i]
        z3.append(z)
        a=sigmoid.sigmoid(z)
        a3.append(a)
    e=sum(y-a3)
    d1=[sigmoid.sigmoid_prime(z3[i]) for i in range(len(a3))]
    d2=[sum(w[2][i,:]) for i in range(len(z3))]
    d3=[sigmoid.sigmoid_prime(z2[i]) for i in range(len(a2))]
    d4=[sum(w[1][i,:]) for i in range(len(z2))]
    d5=[sigmoid.sigmoid_prime(z1[i]) for i in range(len(a1))]
    d6=[sum(w[0][i,:]) for i in range(len(z1))]
    p1,p2,p3=[],[],[]
    p1=[e*d1[i]*d2[i] for i in range(len(d1))]
    p2=[d3[i]*d4[i] for i in range(len(d3))]
    p3=[d5[i]*d6[i] for i in range(len(d5))]
    l=[]
    row={}
    row['p1']=0.0
    row['p2']=0.0
    row['p3']=0.0
    row['product']=0.0
    for i in range(len(p1)):
        for j in range(len(p2)):
            for k in range(len(p3)):
                row['p1']=float(p1[i])
                row['p2']=float(p2[j])
                row['p3']=float(p3[k])
                row['product']=row['p3']*row['p2']*row['p1']
                row['indexo']=i
                row['indexh2']=j
                row['indexh1']=k
                l.append(row)
                row={}
    df=pd.DataFrame(l)
    neuron=df['indexh1'][df['p3'].idxmax()]
    return neuron


def image_pixels(inpt,wt,n):
    '''
    

    Parameters
    ----------
    inpt : 2-dimensional array
        This is the input provided to network.
    wt : 2-dimensional array
        The weight matrix corresponding to the most sensitive neuron of the input layer.
    n : int
        The number of pixels to nullify.

    Returns
    -------
    inpt : 2-dimensional array
        This is the modified input which should be provided to a trained network.

    '''
    prod=inpt*wt
    dat=pd.DataFrame()
    dat['prod']=list(prod)
    index1=dat.sort_values(['prod'],ascending=False)    
    list_of_indexes=index1.index.tolist()
    y=list_of_indexes[0:n]
    for i in range(len(y)):
        inpt[y[i]]=0.0
    return inpt