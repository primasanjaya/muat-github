#!/usr/bin/env python
# coding: utf-8


import os 
import pandas as pd
import numpy as np
import pdb
import glob
import re

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)
#from natsort import natsorted, ns #, #os_sorted


ckptdir = '/scratch/project_2001668/primasan/ckpt/'

folname = os.listdir(ckptdir)

for i in range(len(folname)):

    globptx = glob.glob(ckptdir + folname[i] + '/*.pth*')

    if len(globptx) >= 4:
        #pdb.set_trace()
        sortmodel = natural_sort(globptx)

        for j in range (len(sortmodel)-3):
            os.remove(sortmodel[j])