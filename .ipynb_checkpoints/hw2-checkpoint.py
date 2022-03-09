#!/usr/bin/env python
# coding: utf-8

# In[3]:


import random
import numpy as np
import pandas as pd


# In[4]:


df=pd.read_csv('hw2dataNorm.csv')
df=df.iloc[:,1:]
data=df.to_numpy()
# class values
y=data[:,-1]
X=data[:,:-1]
w=np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
stepSize=0.01
w0=np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])


# # Q 1

# In[5]:


def sigmoidLikelihood(X,y,w):
   
    n,m=X.shape
        
    tem=np.ones((n,1))
    
    X=np.append(X,tem,1)
    g =np.array( 1 / (1 +np.exp( - (X.dot(w) ) )))
    
    pred=[]
    for i in range(0,len(X)): 
        ML= ((1-g[i])**(1-y[i]))*((g[i])**y[i])    
        pred.append(ML)
        
    return np.array(pred)
    


# In[8]:


LVector = sigmoidLikelihood(X, y, w)


# # Q2
# # a)
# # b)-5991.464547107981

# a) If LVector contains all values of 0.05 ([0.05, 0.05, 0.05, ..., 0.05]), how many data points (elements in LVector) are needed for np.prod to estimate the pseudo-likelihood as perfectly 0?

# In[11]:


#


# b) What is the pseudo-log-likelihood equivalent given the number of data points from part (a)?

# In[2]:


# n,m=X.shape
# x=np.full((n),0.05)
# np.sum(np.log(x))


# In[433]:


# np.prod(x)


# # Q3

# In[32]:


def learnLogistic(w0,X, y,K):

    n,m=X.shape
    #creating a n rows of 1 to accomodate b
    
    tem=np.ones((n,1))

    X1=np.append(X,tem,1)
    
   
    update=np.zeros(m+1)
    count=0
    log_Lvector=[]
    w1=np.copy(w0)
    prob =np.array( 1 / (1 +np.exp( - (X1.dot(w1) ) )))
    while count!=K:
        for i,datapt in enumerate(X1):
            for j,feature in enumerate(datapt):
                update[j] += stepSize *(X1[i,j])*( y[i]-prob[i])

        for index,i in enumerate(update):
            w1[index] += update[index]
        

        count+=1
        Lvector=sigmoidLikelihood(X,y,w1)
        log_Lvector.append(np.sum(np.log(Lvector)))
   
        
    return np.array(w1),np.array(log_Lvector)


# In[33]:


K=3
w,LHistory=learnLogistic(w0,X,y,K)


# # Q4

# In[24]:


def learnLogisticFast(w0,X,y,K):
     
    n,m=X.shape
   
    
    tem=np.ones((n,1))
    
    X1=np.append(X,tem,1)
   
    
    dataPt=X1

    update=np.zeros(m+1)
    count=0
    log_Lvector=[]
    w1=np.copy(w0)
    
    prob =np.array( 1 / (1 +np.exp( - (dataPt.dot(w1) ) )))
    
    while count!=K:
    
        for i in range(len(X1)):

             update+= stepSize *(X1[i,:])*( y[i]-prob[i])

        for index,i in enumerate(update):
            w1[index] += update[index]

        count+=1
        Lvector=sigmoidLikelihood(X,y,w1)
        log_Lvector.append(np.sum(np.log(Lvector)))
        
    return np.array(w1),np.array(log_Lvector)


# In[31]:


K=3
w,LHistory=learnLogisticFast(w0,X,y,K)


# # Part b
# K=10
# # learnLogistic(w0,X,y,K) = 0.2756929397583008
# # learnLogisticFast(w0,X,y,K)=0.14236879348754883

# In[43]:


# df=pd.read_csv('hw2dataNorm.csv')
# df=df.iloc[:,1:]
# data=df.to_numpy()
# # class values
# y=data[:,-1]
# X=data[:,:-1]
# w0=np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])


# In[47]:


# import time
# timeStart=time.time()
# K=10
# w,LHistory=learnLogistic(w0,X,y,K)
# timeEnd=time.time()
# print(timeEnd-timeStart)


# In[46]:



# import time
# timeStart=time.time()
# K=10

# w0=np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
# w,LHistory=learnLogisticFast(w0,X,y,K)
# timeEnd=time.time()
# print(timeEnd-timeStart)


# # Q5

# In[87]:


# df=pd.read_csv('hw2dataNorm.csv')
# df=df.iloc[:,1:]
# data=df.to_numpy()
# y=data[:,-1]
# x=data[:,:-1]
# w=np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])


# In[88]:


def logisticClassify(x,w):
   

    n,m=x.shape

    tem=np.ones((n,1))
    
    x=np.append(x,tem,1)
    
    Y_pred =np.array( 1 / (1 +np.exp( - (x.dot(w) ) )))
    
    Y_pred = np.where( Y_pred < 0.5,0,1)
    # this is pseudo-code!
    return np.array(Y_pred)


# In[89]:


classLabels=logisticClassify(x,w)


# In[ ]:




