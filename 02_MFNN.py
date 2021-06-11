#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle as pkl
import asyncio
import time

# sklearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import dump_svmlight_file

# tensorflow
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam

# create plots
import matplotlib.pyplot as plt

# load dataframes
probe = pkl.load(open('./pkl/probe_.pkl', 'rb'))
probe.info()
print('_'*50)
ts = pkl.load(open('./pkl/trainingSet_.pkl', 'rb')) # 20.01 s, 9G
ts.info()


# In[2]:


probe


# In[3]:


ts


# In[4]:


# concat
# concatCustomerID = concat.CustomerID.astype('category').cat.codes.values
# concatMovieID = concat.MovieID.astype('category').cat.codes.values
# ts['catCustomerID'] = concatCustomerID[:ts.shape[0]]
# probe['catCustomerID'] = concatCustomerID[ts.shape[0]:]
# ts['catMovieID'] = concatMovieID[:ts.shape[0]]
# probe['catMovieID'] = concatMovieID[ts.shape[0]:]
# del concat


# In[4]:


# train, test = train_test_split(ts, test_size=0.2)
# pkl.dump(train, open('NN_train.pkl', 'wb'))
# pkl.dump(test, open('NN_test.pkl', 'wb'))
# pkl.dump(probe, open('NN_probe.pkl', 'wb'))
train = pkl.load(open('./pkl/NN_train.pkl', 'rb'))
test = pkl.load(open('./pkl/NN_test.pkl', 'rb'))
probe = pkl.load(open('./pkl/NN_probe.pkl', 'rb'))


# In[8]:


# concat = pd.concat([ts, probe])
# nCustomer, nMovie = len(concat.CustomerID.unique()), len(concat.MovieID.unique())
# del concat


# In[5]:


nCustomer, nMovie =(958804, 17770)


# # latent vector size 10

# In[10]:


nLatentFactor = 10

movieInput = keras.layers.Input(shape=[1],name='movie')
movieEmbedding10 = keras.layers.Embedding(nMovie, nLatentFactor, name='movieEmbedding', 
                                        embeddings_regularizer = keras.regularizers.l2(1e-5))(movieInput)
movieBias10 = keras.layers.Embedding(nMovie, 1, name='movieBias',
                                   embeddings_regularizer = keras.regularizers.l2(1e-5))(movieInput)
movieVec10 = keras.layers.Flatten(name='flattenMovie')(movieEmbedding10)

customerInput = keras.layers.Input(shape=[1],name='customer')
customerEmbedding10 = keras.layers.Embedding(nCustomer, nLatentFactor,name='customerEmbedding', 
                                           embeddings_regularizer = keras.regularizers.l2(1e-5))(customerInput)
customerBias10 = keras.layers.Embedding(nCustomer, 1, name='customerBias', 
                                      embeddings_regularizer = keras.regularizers.l2(1e-5))(customerInput)
customerVec10 = keras.layers.Flatten(name='flattenCustomer')(customerEmbedding10)
 
iProduct10 = keras.layers.dot([customerVec10, movieVec10], axes=1,name='innerProduct')
add10 = keras.layers.add([iProduct10, movieBias10, customerBias10], name='add')
model10 = keras.Model([customerInput, movieInput], add10)

model10.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
model10.summary()


# In[11]:


tf.keras.utils.plot_model(model10, to_file='model10.png')


# In[12]:


history10 = model10.fit([train.catCustomerID, train.catMovieID], train.Rating, 
                     batch_size=16384, 
                     epochs=40)


# In[6]:


# model10.save('model10')
# pkl.dump(pd.DataFrame(history10.history), open('./pkl/history10.pkl', 'wb'))
model10 = keras.models.load_model('model10')
history10 = pkl.load(open('./pkl/history10.pkl', 'rb'))


# In[7]:


pd.Series(history10['loss']).plot(logy=True, figsize=(8, 6))
plt.xlabel("Epoch")
plt.ylabel("Training Error")


# In[8]:


loss10 = model10.evaluate([test.catCustomerID, test.catMovieID], test.Rating, batch_size=32768)


# In[9]:


resultProbe10 = model10.predict([probe.catCustomerID, probe.catMovieID], batch_size=32768)


# In[10]:


probe['Rating10'] = resultProbe10.reshape((-1,))
probe
pkl.dump(probe, open('./pkl/probe_10.pkl', 'wb'))


# In[11]:


movie_embedding_learnt = model10.get_layer(name='movieEmbedding').get_weights()[0]
pd.set_option('precision', 2)
pd.DataFrame(movie_embedding_learnt).describe()


# # latent vector size 20

# In[12]:


nLatentFactor = 20

movieInput = keras.layers.Input(shape=[1],name='movie')
movieEmbedding20 = keras.layers.Embedding(nMovie, nLatentFactor, name='movieEmbedding', 
                                        embeddings_regularizer = keras.regularizers.l2(1e-5))(movieInput)
movieBias20 = keras.layers.Embedding(nMovie, 1, name='movieBias',
                                   embeddings_regularizer = keras.regularizers.l2(1e-5))(movieInput)
movieVec20 = keras.layers.Flatten(name='flattenMovie')(movieEmbedding20)

customerInput = keras.layers.Input(shape=[1],name='customer')
customerEmbedding20 = keras.layers.Embedding(nCustomer, nLatentFactor,name='customerEmbedding', 
                                           embeddings_regularizer = keras.regularizers.l2(1e-5))(customerInput)
customerBias20 = keras.layers.Embedding(nCustomer, 1, name='customerBias', 
                                      embeddings_regularizer = keras.regularizers.l2(1e-5))(customerInput)
customerVec20 = keras.layers.Flatten(name='flattenCustomer')(customerEmbedding20)
 
iProduct20 = keras.layers.dot([customerVec20, movieVec20], axes=1,name='innerProduct')
add20 = keras.layers.add([iProduct20, movieBias20, customerBias20], name='add')
model20 = keras.Model([customerInput, movieInput], add20)

model20.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
model20.summary()


# In[13]:


tf.keras.utils.plot_model(model20, to_file='model20.png')


# In[14]:


history20 = model20.fit([train.catCustomerID, train.catMovieID], train.Rating, 
                     batch_size=16384, 
                     epochs=40)


# In[2]:


# model20 = keras.models.load_model('model20')
# history20 = pkl.load(open('./pkl/history20.pkl', 'rb'))
model20.save('model20')
pkl.dump(pd.DataFrame(history20.history), open('./pkl/history20.pkl', 'wb'))


# In[17]:


pd.Series(history20.history['loss']).plot(logy=True, figsize=(8, 6))
plt.xlabel("Epoch")
plt.ylabel("Training Error")


# In[5]:


loss20 = model20.evaluate([test.catCustomerID, test.catMovieID], test.Rating, batch_size=32768)


# In[ ]:


resultProbe20 = model20.predict([probe.catCustomerID, probe.catMovieID], batch_size=32768)


# In[25]:


probe['Rating20'] = resultProbe20.reshape((-1,))
pkl.dump(probe, open('./pkl/probe_20.pkl', 'wb'))
probe


# In[21]:


movie_embedding_learnt = model20.get_layer(name='movieEmbedding').get_weights()[0]
pd.DataFrame(movie_embedding_learnt).describe()


# In[8]:


probe = probe[['CustomerID', 'Date', 'MovieID', 'Rating20']]
probe.columns = ['CustomerID', 'Date', 'MovieID', 'Rating']
probe.to_csv('probe_NN.csv')

