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

        parser.add_argument('--tcga-dir', type=str,help='tcga directory: all .csv per samples per class')

        parser.add_argument('--muat-dir', type=str,help='muat project directory')

        parser.add_argument('--simplified-dir', type=str,help='output directory for simplification of mutation (3 bp)')

        args = parser.parse_args()
        
        return args

if __name__ == '__main__':

    args = get_args()

    #muat_dir = '/users/primasan/projects/muat/'

    muat_dir = args.muat_dir
    metadata = pd.read_csv(muat_dir + 'extfile/metadata_icgc_pcawg.tsv',sep='\t',index_col=0) 
    dictMutation = pd.read_csv(muat_dir + 'extfile/dictMutation.csv',index_col=0)
    dictChpos = pd.read_csv(muat_dir + 'extfile/dictChpos.csv',index_col=0)
    dictGES = pd.read_csv(muat_dir + 'extfile/dictGES.csv',index_col=0)

    #pcawg dir : directory of all .csv
    pcawg_dir = args.tcga_dir

    #export directory (the files here will be combined with epigenetics data)
    simplified_data = args.simplified_dir

    all_class = os.listdir(pcawg_dir)

    for i in all_class:
        pcawg_histology = i
        os.makedirs(simplified_data + pcawg_histology, exist_ok=True)
        allsamples = os.listdir(pcawg_dir + pcawg_histology)

        for j in allsamples:

            onesamples = j

            read_sample = pd.read_csv(pcawg_dir + pcawg_histology + '/' + onesamples)
                
            somatic_files_persamples = read_sample
            pcawg_sample = onesamples[:-4]
            
            #1) simplifing mutation information
            simplified_mutation = []
            for k in range(0,len(somatic_files_persamples)):
                onemut = somatic_files_persamples.iloc[k]
        
                trp = onemut['seq'][int(len(onemut['seq'])/2)-1:int(len(onemut['seq'])/2)+2]
                ps = math.floor(onemut['pos'] / 1000000)

                chompos = str(onemut['chrom']) + '_' + str(ps)
                rt = onemut['rt']
                ges = str(onemut['genic']) + '_' + str(onemut['exonic']) + '_' + str(onemut['strand'])
                try:
                    typ = dictMutation.loc[dictMutation['triplet']==trp]['mut_type'].values[0]
                except:
                    typ = 'Neg'

                tuple_onerow = (trp,chompos,rt,ges,typ,onemut['chrom'],onemut['pos'])
                simplified_mutation.append(tuple_onerow)
            pd_new = pd.DataFrame(simplified_mutation)
            pd_new.columns = ['seq','chrompos','rt','ges','mut_type','chrom','pos']
            pd_new.to_csv(simplified_data + pcawg_histology + '/' + onesamples )