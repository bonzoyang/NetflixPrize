#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import pickle as pkl

# To create plots
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix, lil_matrix
from sklearn.model_selection import train_test_split


# In[3]:


smp_train = pkl.load(open('./pkl/smp_train.pkl', 'rb'))
smp_test = pkl.load(open('./pkl/smp_test.pkl', 'rb'))
probe = pkl.load(open('./pkl/probe_.pkl', 'rb'))


# In[4]:


smp_test_marked = smp_test.copy()
smp_test_marked.Rating = np.NaN
smp = pd.concat([smp_train, smp_test_marked])
pvt = smp.pivot_table(index='CustomerID', columns='MovieID', values='Rating')
print('Shape User-Item matrix:\t{}'.format(pvt.shape))
pvt.head()


# In[5]:


print(f'there are {(~np.isnan(pvt.values)).sum()} element that is not nan')


# In[6]:


"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension k x M
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""
def matrix_factorization(R, P, Q, B, K, steps=5000, alpha=0.002, beta=1e-5):
    print(f'R:{R.shape}, P:{P.shape}, Q:{Q.shape}, B:{B.shape}')
    rows, cols = np.where(~np.isnan(R))
    narows, nacols =  np.where(np.isnan(R))
    N = rows.shape[0]
    
    Mask = np.zeros(R.shape)#lil_matrix(R.shape, )
    Mask[rows, cols] = 1
    R[narows, nacols] = 0
    errorHistory = []
    
    p = 0
    for step in range(steps):
        import time
        s = time.time()
        R_hat = B + np.dot(P, Q)
        E = R - R_hat
        gradP = -np.dot(E, Q.T) + beta*P
        gradQ = -np.dot(E.T, P).T + beta*Q
        gradB = -E + beta*B
        
        # SGD
        P = P - alpha * gradP
        Q = Q - alpha * gradQ
        B = B - alpha * gradB
        
        error = np.sqrt( np.square(
            np.multiply(E, Mask)).sum()/N 
                       )
        errorHistory.append(error)
        
        p += 1
        print(f'\nprogress {p} time:{time.time()-s:.2f} s step:{step} Error:{error} ')
        
        if error < 1e-5:
            break
    else:
        print(f'step:{step} Error:{error}')
    
    return P, Q, errorHistory


# In[7]:


nLatentFactor = 20
R = pvt.values
P = np.random.rand(R.shape[0], nLatentFactor)
Q = np.random.rand(nLatentFactor, R.shape[1])
B = np.random.rand(R.shape[0], R.shape[1])

P_, Q_, H_ = matrix_factorization(R, P, Q, B, K=nLatentFactor, steps=3)


# In[393]:


P_


# In[394]:


Q_


# In[383]:


R


# In[384]:


P


# In[385]:


Q


# In[364]:


np.square(
    np.multiply(E, Mask)).sum()/N 


# In[363]:


Error = np.sqrt( np.square(
    np.multiply(E, Mask)).sum()/N 
               )


# In[324]:


type(gradP)


# In[319]:


alphaP = 0
alphaQ = 0
alphaB = 0
alphaP = alphaP+ gradP**2
alphaQ = gradQ**2
alphaB = gradB**2


# In[298]:


np.dot(P, Q)
R_hat = B + np.dot(P, Q)
E = R - R_hat
E.shape


# In[318]:


beta = 0.02
R_hat = B + np.dot(P, Q)
E = R - R_hat
#         gradP = -np.dot(E, Q.T) + beta*P
#         gradQ = -np.dot(E.T, P) + beta*Q
gradP = -E @ Q.T + beta*P
gradQ = (-(E.T) @ P).T + beta*Q
gradB = -E


# In[53]:





# In[52]:


P


# In[ ]:




