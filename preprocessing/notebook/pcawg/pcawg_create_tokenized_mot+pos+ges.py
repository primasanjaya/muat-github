import numpy as np
import pandas as pd
import pdb
import os
import math
import argparse


'''
create token from simplified data

Ex: python pcawg_create_tokenized_mot+pos+ges.py
todo: args
'''

def get_args():
        parser = argparse.ArgumentParser(description='preprocessing args')

        parser.add_argument('--data-dir', type=str,help='tcga directory: all .csv per samples per class')

        parser.add_argument('--sample-file', type=str,help='sample file')

        parser.add_argument('--epigen-file', type=str,help='epigenetic state file')

        parser.add_argument('--muat-dir', type=str,help='muat project directory')

        parser.add_argument('--output-dir', type=str,help='output directory for simplification of mutation (3 bp)')

        args = parser.parse_args()
        
        return args

if __name__ == '__main__':

    #args = get_args()

    #edit the directory
    muat_dir = '/users/primasan/projects/muat/'
    metadata = pd.read_csv(muat_dir + 'extfile/metadata_icgc_pcawg.tsv',sep='\t',index_col=0) 
    dictMutation = pd.read_csv(muat_dir + 'extfile/dictMutation.csv',index_col=0)
    dictChpos = pd.read_csv(muat_dir + 'extfile/dictChpos.csv',index_col=0)
    dictGES = pd.read_csv(muat_dir + 'extfile/dictGES.csv',index_col=0)

    pcawg_dir = '/scratch/project_2001668/data/pcawg/allclasses/newformat/'

    simplified_data = '/scratch/project_2001668/data/pcawg/allclasses/simplified/'
    tokenized_data = '/scratch/project_2001668/data/pcawg/allclasses/tokenized/'

    pd_tumoursamples = pd.read_csv('/users/primasan/projects/muat/extfile/all_tumoursamples_pcawg.tsv',sep='\t',index_col=0)

    for i in range(len(pd_tumoursamples)):
        pcawg_histology = pd_tumoursamples.iloc[i]['tumour_type']

        os.makedirs(tokenized_data + pcawg_histology, exist_ok=True)

        onesamples = pd_tumoursamples.iloc[i]['sample']
        read_sample = pd.read_csv(simplified_data + pcawg_histology + '/' + onesamples,index_col=0)
            
        somatic_files_persamples = read_sample
        pcawg_sample = onesamples[:-4]

        pd_new = read_sample
        print(onesamples)
        
        #2) tokenization from simplified mutation files
        mergetriplet = pd_new.merge(dictMutation, left_on='seq', right_on='triplet', how='left',suffixes=('', '_y'))
        mergetriplet.drop(mergetriplet.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
        mergeges = mergetriplet.merge(dictGES, left_on='ges', right_on='ges', how='left',suffixes=('', '_y'))
        mergeges.drop(mergeges.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
        mergeges = mergeges.rename(columns={"token": "gestoken"})
        mergechrompos = mergeges.merge(dictChpos, left_on='chrompos', right_on='chrompos', how='left',suffixes=('', '_y'))
        mergechrompos.drop(mergechrompos.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
        mergechrompos = mergechrompos.rename(columns={"token": "postoken"})
        token_data = mergechrompos[['triplettoken','postoken','gestoken','rt','mut_type']]
        
        #save token data
        #pdb.set_trace()
        token_files = 'token_' + pcawg_sample + '.csv'
        token_data.to_csv(tokenized_data + pcawg_histology + '/' + token_files)
        
        #3) create data utilities (required)
        SNVonly = token_data.loc[token_data['mut_type']=='SNV']
        SNVonly = SNVonly.drop(columns=['mut_type'])
        MNVonly = token_data.loc[token_data['mut_type']=='MNV']
        MNVonly = MNVonly.drop(columns=['mut_type'])
        indelonly = token_data.loc[token_data['mut_type']=='indel']
        indelonly = indelonly.drop(columns=['mut_type'])
        MEISVonly = token_data.loc[token_data['mut_type'].isin(['MEI','SV'])]
        Negonly = token_data.loc[token_data['mut_type']=='Normal']
        Negonly = Negonly.drop(columns=['mut_type'])

        #have to be exported
        SNVonly.to_csv(tokenized_data + pcawg_histology + '/' + 'SNV_' + pcawg_sample + '.csv')
        MNVonly.to_csv(tokenized_data + pcawg_histology + '/' + 'MNV_' + pcawg_sample + '.csv')
        indelonly.to_csv(tokenized_data + pcawg_histology + '/' + 'indel_' + pcawg_sample + '.csv')
        MEISVonly.to_csv(tokenized_data + pcawg_histology + '/' + 'MEISV_' + pcawg_sample + '.csv')
        Negonly.to_csv(tokenized_data + pcawg_histology + '/' + 'Normal_' + pcawg_sample + '.csv')
        pd_count = pd.DataFrame([len(SNVonly),len(MNVonly),len(indelonly),len(MEISVonly),len(Negonly)])
        pd_count.to_csv(tokenized_data + pcawg_histology + '/' + 'count_' + pcawg_sample + '.csv')