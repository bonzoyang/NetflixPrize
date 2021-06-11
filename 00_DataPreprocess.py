#!/usr/bin/env python
# coding: utf-8

# # Environment

# In[1]:


# version: Python 3.8.5
# requirements.txt


# # Imports and settings

# In[2]:


import glob
import pandas as pd
import numpy as np
import pickle as pkl
import asyncio
from scipy.sparse import csr_matrix

path = './download/'

qf, _, _, mt, _, ts, pb = glob.glob(path+'*')
qf, mt, ts, pb


# # Data cleaning

# In[3]:


def addMovieIDColumn(df):
    movieId = df[df.CustomerID.str.contains(':')].copy()
    lastRowIndex = df.shape[0]
    movieIdStart = movieId.index.to_list()
    movieIdEnd = movieIdStart[1:]+[lastRowIndex]

    movieIdRepeat = []
    for s, e in zip(movieIdStart,movieIdEnd):
        mid = int(movieId.loc[s,'CustomerID'].replace(':',''))
        movieIdRepeat += [mid]*(e-s)
    else:
        df['MovieID'] = pd.Series(movieIdRepeat)
        df_ = df[~df.CustomerID.str.contains(':')].copy()
        
    return df_, movieId


# ### qualifying.txt

# In[4]:


dfqf = pd.read_csv(qf, header = None, names = ['CustomerID', 'Date'], usecols = [0,1])
dfqf_, movieId = addMovieIDColumn(dfqf)
dfqf_.reset_index(drop=True, inplace=True)
movieId.reset_index(drop=True, inplace=True)
pkl.dump(dfqf_, open(f'./pkl/qualifying_.pkl', 'wb')) # for prediction
pkl.dump(movieId, open(f'./pkl/qualifyingMovieId.pkl', 'wb')) # cache movieId info
dfqf_


# ### probe.txt

# In[5]:


dfpb = pd.read_csv(pb, header = None, names = ['CustomerID'], usecols = [0])
dfpb_, movieId = addMovieIDColumn(dfpb)
dfpb_.reset_index(drop=True, inplace=True)
movieId.reset_index(drop=True, inplace=True)
pkl.dump(dfqf_, open(f'./pkl/probe_.pkl', 'wb')) # for prediction
pkl.dump(movieId, open(f'./pkl/probeMovieId.pkl', 'wb')) # cache movieId info
dfpb_


# ### movie_titles.txt

# In[6]:


dfmt = pd.read_csv(mt, header = None, 
                   names = ['MovieID', 'YearOfRelease', 'Title'], 
                   usecols = [0,1,2], 
                   encoding = "ISO-8859-1")
dfmt.Title = dfmt.Title.apply(lambda x : x.replace('\n',''))
dfmt[dfmt.YearOfRelease.isna()]


# In[7]:


_ = dfmt[dfmt.YearOfRelease.isna()].index
dfmt.loc[_, 'YearOfRelease'] = (2001, 2001, 2001, 1974, 1999, 1994, 1999) # hand fix
dfmt.YearOfRelease.astype('int', copy=True).astype('str')
dfmt.YearOfRelease = dfmt.YearOfRelease.apply(lambda x: pd.to_datetime(x, format='%Y', errors='ignore'))
pkl.dump(dfmt, open(f'./pkl/movie_titles.pkl', 'wb'))
dfmt


# # training_set

# In[8]:


# procedural process (2552.06 s)
tsFiles = glob.glob(ts+'/*')
tsFiles.sort()

dfts_ = []
movieId = []
for _ in tsFiles:
    dftsi = pd.read_csv(_, header = None, names = ['CustomerID', 'Rating', 'Date'], usecols = [0, 1, 2])
    dftsi_, movieIdi = addMovieIDColumn(dftsi)
    dfts_.append(dftsi_)
    movieId.append(movieIdi)
else:
    dfts_ = pd.concat(dfts_).reset_index(drop=True)
    movieId = pd.concat(movieId).reset_index(drop=True)
    dfts_.CustomerID = dfts_.CustomerID.astype('int32')
    dfts_.MovieID = dfts_.MovieID.astype('int32')
    dfts_.Rating = dfts_.Rating.astype('int32')
    
    pkl.dump(dfts_, open(f'./pkl/trainingSet_.pkl', 'wb')) # for prediction
    pkl.dump(movieId, open(f'./pkl/trainingSetMovieId.pkl', 'wb')) # cache movieId info
dfts_ 


# In[20]:


# multiprocess process (3586.15 s)(not improved)
from multiprocessing import Process, Queue
import os
tsFiles = glob.glob(ts+'/*')
tsFiles.sort()

cpus = os.cpu_count()

def part(f, Qdft_, QmovieId):
    dftsi = pd.read_csv(_, header = None, names = ['CustomerID', 'Rating', 'Date'], usecols = [0, 1, 2])
    dftsi_, movieIdi = addMovieIDColumn(dftsi)
    Qdft_.put(dftsi_)
    QmovieId.put(movieIdi)


Qdft_ = Queue()
QmovieId = Queue()
jobs = []

dfts_ = []
movieId = []
for _ in tsFiles:
    p = Process(target=part, args=(_, Qdft_, QmovieId))
    jobs.append(p)
    p.start()
    dfts_.append(Qdft_.get())
    movieId.append(QmovieId.get())
    
    if len(jobs) >= cpus:
        [job.join() for job in jobs] # block until every job finish
        jobs = []
else:
    [job.join() for job in jobs] # block until every job finish
    dfts_ = pd.concat(dfts_).reset_index(drop=True)
    movieId = pd.concat(movieId).reset_index(drop=True)
    pkl.dump(dfts_, open(f'./pkl/trainingSet_.pkl', 'wb')) # for prediction
    pkl.dump(movieId, open(f'./pkl/trainingSetMovieId.pkl', 'wb')) # cache movieId info
dfts_

