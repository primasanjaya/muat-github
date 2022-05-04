import numpy as np
import pandas as pd
import pdb
import os
import math
import argparse

'''
how to run
ex:
python3 /users/primasan/projects/muat/preprocessing/notebook/tcga/tcga_create_epigen_data_slurm.py --class-name 'HNSC' --sample-file 'TCGA-CN-4739-01A-02D-1512-08.csv' --muat-dir '/users/primasan/projects/muat/' --epigen-file '/scratch/project_2001668/data/pcawg/allclasses/epigenetic_token.tsv' --simplified-dir '/scratch/project_2001668/data/pcawg/allclasses/simplified_data/' --tokenized-dir '/scratch/project_2001668/data/pcawg/allclasses/tokenized_data/'
'''

def get_args():
        parser = argparse.ArgumentParser(description='preprocessing args')

        parser.add_argument('--class-name', type=str,help='class-name')

        parser.add_argument('--sample-file', type=str,help='sample file')

        parser.add_argument('--epigen-file', type=str,help='epigenetic state file')

        parser.add_argument('--muat-dir', type=str,help='muat project directory')

        parser.add_argument('--simplified-dir', type=str,help='3bp simplified data directory')

        parser.add_argument('--tokenized-dir', type=str,help='final token output for training')

        args = parser.parse_args()
        
        return args

if __name__ == '__main__':

    args = get_args()

    #requirements:
    metadata = pd.read_csv(args.muat_dir + 'extfile/metadata_icgc_pcawg.tsv',sep='\t',index_col=0) 
    dictMutation = pd.read_csv(args.muat_dir + 'extfile/dictMutation.csv',index_col=0)
    dictChpos = pd.read_csv(args.muat_dir + 'extfile/dictChpos.csv',index_col=0)
    dictGES = pd.read_csv(args.muat_dir + 'extfile/dictGES.csv',index_col=0)

    epigene = pd.read_csv(args.epigen_file,sep='\t')
    epigene = epigene.drop(columns=['Unnamed: 0'])

    pcawg_dir = args.simplified_dir
    tokenized_data = args.tokenized_dir

    pcawg_histology = args.class_name

    os.makedirs(tokenized_data + pcawg_histology, exist_ok=True)

    onesamples = args.sample_file

    pd_new = pd.read_csv(pcawg_dir + pcawg_histology + '/' + onesamples,index_col=0)

    mergetriplet = pd_new.merge(dictMutation, left_on='seq', right_on='triplet', how='left',suffixes=('', '_y'))
    mergetriplet.drop(mergetriplet.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
    mergeges = mergetriplet.merge(dictGES, left_on='ges', right_on='ges', how='left',suffixes=('', '_y'))
    mergeges.drop(mergeges.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
    mergeges = mergeges.rename(columns={"token": "gestoken"})
    mergechrompos = mergeges.merge(dictChpos, left_on='chrompos', right_on='chrompos', how='left',suffixes=('', '_y'))
    mergechrompos.drop(mergechrompos.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
    mergechrompos = mergechrompos.rename(columns={"token": "postoken"})

    pcawg_sample = onesamples[:-4]

    #1) simplifing mutation information
    epigenetics_list = []
    pd_epiall = pd.DataFrame()
    for k in range(0,len(mergechrompos)):
        onemut = mergechrompos.iloc[k]
        chrom = onemut['chrom']
        pos = onemut['pos']
        all_chrom = epigene.loc[epigene['chrom']==str('chr'+str(chrom))]
        all_pos = all_chrom.loc[all_chrom['start']<=pos]
        get_epi = all_pos.loc[all_chrom['end']>=pos]

        if len(get_epi) == 0:
            try:
                pd_epi = pd.DataFrame(np.zeros(145),dtype=int).T
                pd_epi.columns = get_epi.columns[3:]
            except:
                pdb.set_trace()
        else:
            pd_epi = pd.DataFrame(get_epi.iloc[0,3:]).T

        pd_epiall = pd_epiall.append(pd_epi)

    pd_epiall = pd_epiall.reset_index(drop=True)
    mergechrompos = mergechrompos.reset_index(drop=True)
    mergechrompos['rt'] = mergechrompos['rt'].fillna(0)
    mergechrompos = pd.concat([mergechrompos, pd_epiall], axis=1)
    get_columns = ['triplettoken','postoken','gestoken','rt','mut_type']+pd_epiall.columns.to_list()

    token_data = mergechrompos[get_columns]

    SNVonly = token_data.loc[token_data['mut_type']=='SNV']
    SNVonly = SNVonly.drop(columns=['mut_type'])
    MNVonly = token_data.loc[token_data['mut_type']=='MNV']
    MNVonly = MNVonly.drop(columns=['mut_type'])
    indelonly = token_data.loc[token_data['mut_type']=='indel']
    indelonly = indelonly.drop(columns=['mut_type'])
    MEISVonly = token_data.loc[token_data['mut_type'].isin(['MEI','SV'])]
    Negonly = token_data.loc[token_data['mut_type']=='Normal']
    Negonly = Negonly.drop(columns=['mut_type'])

    SNVonly.to_csv(tokenized_data + pcawg_histology + '/' + 'SNV_' + pcawg_sample + '.csv')
    MNVonly.to_csv(tokenized_data + pcawg_histology + '/' + 'MNV_' + pcawg_sample + '.csv')
    indelonly.to_csv(tokenized_data + pcawg_histology + '/' + 'indel_' + pcawg_sample + '.csv')
    MEISVonly.to_csv(tokenized_data + pcawg_histology + '/' + 'MEISV_' + pcawg_sample + '.csv')
    Negonly.to_csv(tokenized_data + pcawg_histology + '/' + 'Normal_' + pcawg_sample + '.csv')
    pd_count = pd.DataFrame([len(SNVonly),len(MNVonly),len(indelonly),len(MEISVonly),len(Negonly)])
    pd_count.to_csv(tokenized_data + pcawg_histology + '/' + 'count_' + pcawg_sample + '.csv')