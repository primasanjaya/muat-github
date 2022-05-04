import numpy as np
import pandas as pd
import os
import pdb

#requirements:
total_fold = 10

#pcawg dir --> new pcawg directory
pcawg_dir = '/csc/epitkane/projects/PCAWG20191001/data/modified_data/train/new24classes/'

#output dir --> to projectdir/dataset_utils
output_dir = '../dataset_utils/'

#scan all samples per tumour types
tumour_types = os.listdir(pcawg_dir)
tumour_types = [i for i in tumour_types if len(i.split('.'))==1]
tumour_types.sort()

#scan all samples
pd_allsamples = pd.DataFrame()
for i in range(len(tumour_types)):
    all_samples = os.listdir(pcawg_dir + tumour_types[i])
    #filter
    all_samples = [i[5:] for i in all_samples if i[0:4]=='new_']
    pd_samp = pd.DataFrame(all_samples)
    pd_samp['nm_class'] = tumour_types[i]
    
    pd_allsamples = pd_allsamples.append(pd_samp)
pd_allsamples.columns = ['samples','nm_class']


#slicing data
get_10slices = []
startslice=0
for i in range(0,len(pd_allsamples)):
    startslice = startslice + 1    
    if startslice > 10:
        startslice = 1
        get_10slices.append(startslice)
    else:
        get_10slices.append(startslice)
pd_allsamples['slices'] = get_10slices


#create_train_val_test split,
trainAll = pd.DataFrame()
valAll = pd.DataFrame()

for valfold in range(1,11):
    val = pd_allsamples.loc[pd_allsamples['slices']==valfold]
    train = pd_allsamples.loc[pd_allsamples['slices']!=valfold]
    
    train['fold'] = valfold
    val['fold'] = valfold
    
    trainAll = trainAll.append(train)
    valAll = valAll.append(val)
trainAll.to_csv(output_dir + 'pcawg_train.csv')
valAll.to_csv(output_dir + 'pcawg_val.csv')

#create class_info
pd_allsamples = []
for i in range(len(tumour_types)):
    all_samples = os.listdir(pcawg_dir + tumour_types[i])
    #filter
    all_samples = [i[5:] for i in all_samples if i[0:4]=='new_']
    one_tuple = (tumour_types[i],i,len(all_samples))
    pd_allsamples.append(one_tuple)
pd_allsamples = pd.DataFrame(pd_allsamples)
pd_allsamples.columns = ['class_name','class_index','n_samples']
pd_allsamples.to_csv(output_dir + 'classinfo_pcawg.csv')




