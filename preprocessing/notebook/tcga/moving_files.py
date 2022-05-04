import numpy as np 
import pandas as pd
import os 
import pdb
import shutil


tcga_dir = '/scratch/project_2001668/data/tcga/'
shuffled_dir = tcga_dir + 'shuffled/'
alltcgaclass = tcga_dir + 'alltcga/'
allicd10 = tcga_dir + 'allicd10/'

pd_metadata = pd.read_csv(tcga_dir + 'mc3.clinical.tsv',sep='\t')

list_samples = os.listdir(shuffled_dir)

for i in list_samples:

    getone = pd_metadata.loc[pd_metadata['sample_id']==i[:-4]]

    diseasecode = getone['disease_code']

    if len(diseasecode)>0:
        if not diseasecode.isnull().values.any():
            path = alltcgaclass + diseasecode.values[0]
            os.makedirs(path, exist_ok = True)

            src = shuffled_dir + i
            dst = path + '/' + i

            shutil.copyfile(src, dst)
    
    icd_code = getone['icd_10']

    if len(icd_code)>0:
        if not diseasecode.isnull().values.any():
            path = allicd10 + icd_code.values[0]
            os.makedirs(path, exist_ok = True)

            src = shuffled_dir + i
            dst = path + '/' + i

            shutil.copyfile(src, dst)




