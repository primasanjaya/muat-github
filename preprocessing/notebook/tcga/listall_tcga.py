import numpy as np
import pandas as pd
import pdb
import os
import math
import argparse

if __name__ == '__main__':

    #edit the directory
    muat_dir = '/users/primasan/projects/muat/'
    metadata = pd.read_csv(muat_dir + 'extfile/metadata_icgc_pcawg.tsv',sep='\t',index_col=0) 
    dictMutation = pd.read_csv(muat_dir + 'extfile/dictMutation.csv',index_col=0)
    dictChpos = pd.read_csv(muat_dir + 'extfile/dictChpos.csv',index_col=0)
    dictGES = pd.read_csv(muat_dir + 'extfile/dictGES.csv',index_col=0)

    pcawg_dir = '/scratch/project_2001668/data/tcga/alltcga/'
    simplified_data = '/scratch/project_2001668/data/tcga/simplified/'

    tokenized_data = '/scratch/project_2001668/data/tcga/tokenized/'

    all_class = os.listdir(pcawg_dir)

    pd_all = []

    for i in all_class:
        pcawg_histology = i
        allsamples = os.listdir(simplified_data + pcawg_histology)

        for j in allsamples:
            onesamples = j

            onerow = (pcawg_histology,onesamples)
            pd_all.append(onerow)

    pd_allsamp = pd.DataFrame(pd_all)
    pd_allsamp.columns = ['tumour_type','sample']

    pd_allsamp.to_csv(muat_dir +'extfile/all_tumoursamples_tcga.tsv', sep = '\t')