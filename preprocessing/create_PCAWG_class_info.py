#!/usr/bin/env python
# coding: utf-8

# In[55]:


import numpy as np
import pandas as pd
import os
import pdb
import math


# In[ ]:





# In[56]:


#pcawg dir --> new pcawg directory
pcawg_dir = 'G:/experiment/data/new24classes/'

#output dir --> to projectdir/dataset_utils
output_dir = '../dataset_utils/'


# In[ ]:





# In[63]:


#scan all samples per tumour types

tumour_types = os.listdir(pcawg_dir)
tumour_types = [i for i in tumour_types if len(i.split('.'))==1]
tumour_types.sort()


# In[64]:


#scan all samples
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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




