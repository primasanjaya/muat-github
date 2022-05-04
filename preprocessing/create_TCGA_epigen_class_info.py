#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import os
import pdb
import math


#pcawg dir --> new pcawg directory
pcawg_dir = '/mnt/e/puhti/data/tcga/tokenized/'
#output dir --> to projectdir/dataset_utils
output_dir = '../dataset_utils/'


#scan all samples per tumour types

tumour_types = os.listdir(pcawg_dir)
tumour_types = [i for i in tumour_types if len(i.split('.'))==1]
tumour_types.sort()

#scan all samples
pd_allsamples = []
for i in range(len(tumour_types)):
    all_samples = os.listdir(pcawg_dir + tumour_types[i])
    #filter
    all_samples = [i[6:] for i in all_samples if i[0:6]=='token_']
    one_tuple = (tumour_types[i],i,len(all_samples))
    pd_allsamples.append(one_tuple)
pd_allsamples = pd.DataFrame(pd_allsamples)
pd_allsamples.columns = ['class_name','class_index','n_samples']

pd_allsamples_filtered = pd_allsamples.loc[pd_allsamples['n_samples']>=100]

pd_allsamples_filtered['class_index'] = np.arange(len(pd_allsamples_filtered))

pd_allsamples_filtered = pd_allsamples_filtered.reset_index(drop=True)

pd_allsamples_filtered.to_csv(output_dir + 'classinfo_tcga_epi.csv')


