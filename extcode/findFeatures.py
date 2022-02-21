#!/usr/bin/env python
# coding: utf-8


import os 
import pandas as pd
import numpy as np
import pdb
import glob
#from natsort import natsorted, ns #, #os_sorted

featureori = pd.read_csv('./extfile/features.csv',index_col = 0)

ckpt_dir = '/scratch/project_2001668/primasan/ckpt/'

listdir = os.listdir(ckpt_dir)

for i in range(0,len(listdir)):
    #print(listdir[i])
    try:
        readfeature = pd.read_csv(ckpt_dir + listdir[i] + '/features.csv',index_col=0)
        if featureori.equals(readfeature):
            print(ckpt_dir + listdir[i])
    except:
        pass