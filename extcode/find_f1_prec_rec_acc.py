#!/usr/bin/env python
# coding: utf-8

import os 
import pandas as pd
import numpy as np
import pdb
import glob

ckpt_dir = '/scratch/project_2001668/primasan/ckpt/'

pd_all = pd.DataFrame()

dictAll = {}

foldstart = 2
foldend = 11

for i in range(foldstart,foldend):
    pd_read = pd.read_csv('./topfolds' + str(i) + '.csv',index_col=0)

    pd_all = pd_all.append(pd_read)

model = pd_read['model']
totmodel = len(model)

allmodel = []

for x in model:
    split = x.split('_')

    tag = split[0]

    removefold = ''.join([i for i in tag if not i.isdigit()])

    split.append(removefold)

    allmodel.append(split)


pd_sortpermodel = pd.DataFrame()

for i in range(len(allmodel)):
    sr_model = allmodel[i]
    for fo in range(foldstart,foldend):
        joinstr = '_'.join(sr_model[1:-1])
        src_model = sr_model[-1] + str(fo) + '_' +joinstr

        src_pd_all = pd_all.loc[pd_all['model']==src_model]
        pd_sortpermodel = pd_sortpermodel.append(src_pd_all)

pd_sortpermodel.to_csv('./notebookpcawg/sortpermodel.csv')

for i in range(int(len(pd_sortpermodel)/(foldend - foldstart))):

    start_row = i * (foldend - foldstart)
    end_row = start_row + (foldend - foldstart)

    print(str(start_row) + ' ' + str(end_row))

    pd_sum = pd_sortpermodel.iloc[start_row:end_row]

    splits = pd_sum['model'].iloc[0]

    splits = splits.split('_')
    tag = splits[0]
    removefold = ''.join([i for i in tag if not i.isdigit()])

    splits.append(removefold)

    joinstr = '_'.join(splits[1:-1])

    joinstr = removefold + joinstr + '.csv'

    pd_sum.to_csv('./notebookpcawg/allresults/' + joinstr)




        


