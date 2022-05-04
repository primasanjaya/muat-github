import numpy as np
import pandas as pd
import pdb
import os
import math
import argparse

'''
how to use
ex:
python3 /users/primasan/projects/muat/preprocessing/notebook/tcga/tcga_create_simplified_data.py --muat-dir '/users/primasan/projects/muat/' --tcga-dir '/scratch/project_2001668/data/tcga/alltcga/' --simplified-dir '/scratch/project_2001668/data/tcga/simplified/'

'''

def get_args():
        parser = argparse.ArgumentParser(description='preprocessing args')

        parser.add_argument('--data-dir', type=str,help='tcga directory: all .csv per samples per class')

        parser.add_argument('--muat-dir', type=str,help='muat project directory')

        parser.add_argument('--simplified-dir', type=str,help='output directory for simplification of mutation (3 bp)')

        args = parser.parse_args()
        
        return args

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