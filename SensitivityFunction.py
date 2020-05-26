# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:57:54 2019

@author: nagar
"""
from sensitivity_analysis import sigmoid
import pandas as pd

def sensitive_neuron(inpt,w,b,y):
    a1=[]
    a2=[]
    a3=[]
    z1=[]
    z2=[]
    z3=[]
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
    d1=[]
    d3=[]
    d5=[]
    d2=[]
    d4=[]
    d6=[]
    for i in range(len(a3)):
        d=sigmoid.sigmoid_prime(z3[i])
        d1.append(d)
    for i in range(len(a2)):
        d=sigmoid.sigmoid_prime(z2[i])
        d3.append(d)
    for i in range(len(a1)):
        d=sigmoid.sigmoid_prime(z1[i])
        d5.append(d)
    for i in range(len(z3)):
        d=sum(w[2][i,:])
        d2.append(d)
    for i in range(len(z2)):
        d=sum(w[1][i,:])
        d4.append(d)
    for i in range(len(z1)):
        d=sum(w[0][i,:])
        d6.append(d)
    p1=[]
    p2=[]
    p3=[]
    for i in range(len(d1)):
        p=e*d1[i]*d2[i]
        p1.append(p)
    for i in range(len(d3)):
        p=d3[i]*d4[i]
        p2.append(p)
    for i in range(len(d5)):
        p=d5[i]*d6[i]
        p3.append(p)
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
    prod=inpt*wt
    dat=pd.DataFrame()
    dat['prod']=list(prod)
    index1=dat.sort_values(['prod'],ascending=False)    
    list_of_indexes=index1.index.tolist()
    y=list_of_indexes[0:n]
    for i in range(len(y)):
        inpt[y[i]]=0.0
    return inpt