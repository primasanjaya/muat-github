import numpy as np
import pandas as pd
import pdb
import os
import math

#requirements:
metadata = pd.read_csv('/csc/epitkane/projects/muat/extfile/metadata_icgc_pcawg.tsv',sep='\t',index_col=0) 
dictMutation = pd.read_csv('/csc/epitkane/projects/muat/extfile//dictMutation.csv',index_col=0)
dictChpos = pd.read_csv('/csc/epitkane/projects/muat/extfile/dictChpos.csv',index_col=0)
dictGES = pd.read_csv('/csc/epitkane/projects/muat/extfile/dictGES.csv',index_col=0)

epigene = pd.read_csv('/csc/epitkane/projects/PCAWG20191001/data/modified_data/train/allclasses/epigenetic_token.tsv',sep='\t',index_col=0)

pcawg_dir = '/csc/epitkane/projects/PCAWG20191001/data/modified_data/train/allclasses/simplified_data/'
sourcedir = '/csc/epitkane/projects/PCAWG20191001/data/modified_data/raw/indelmeisubsvperclass/'
tokenized_data = '/csc/epitkane/projects/PCAWG20191001/data/modified_data/train/allclasses/tokenized_data/'

all_class = os.listdir(sourcedir)

for i in range(0,len(all_class)):
    all_samples = os.listdir(sourcedir + all_class[i])
    print(all_class[i])

    pcawg_histology = all_class[i]
    os.makedirs(tokenized_data + pcawg_histology, exist_ok=True)
    for j in range(0,len(all_samples)):
        pd_new = pd.read_csv(pcawg_dir + all_class[i] + '/' + all_samples[j],index_col=0)

        mergetriplet = pd_new.merge(dictMutation, left_on='seq', right_on='triplet', how='left',suffixes=('', '_y'))
        mergetriplet.drop(mergetriplet.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
        mergeges = mergetriplet.merge(dictGES, left_on='ges', right_on='ges', how='left',suffixes=('', '_y'))
        mergeges.drop(mergeges.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
        mergeges = mergeges.rename(columns={"token": "gestoken"})
        mergechrompos = mergeges.merge(dictChpos, left_on='chrompos', right_on='chrompos', how='left',suffixes=('', '_y'))
        mergechrompos.drop(mergechrompos.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
        mergechrompos = mergechrompos.rename(columns={"token": "postoken"})

        pcawg_sample = all_samples[j][:-4]

        #1) simplifing mutation information
        epigenetics_list = []
        pd_epiall = pd.DataFrame()
        for k in range(0,len(mergechrompos)):
            onemut = mergechrompos.iloc[k]
            chrom = onemut['chrom']
            pos = onemut['pos']
            all_chrom = epigene.loc[epigene['chrom']==str('chr'+chrom)]
            all_pos = all_chrom.loc[all_chrom['start']<=pos]
            get_epi = all_pos.loc[all_chrom['end']>=pos]

            if len(get_epi) == 0:
                pd_epi = pd.DataFrame(np.zeros(145),dtype=int).T
                pd_epi.columns = pd_epiall.columns
            else:
                pd_epi = pd.DataFrame(get_epi.iloc[0,3:]).T

            pd_epiall = pd_epiall.append(pd_epi)

        pd_epiall = pd_epiall.reset_index(drop=True)
        mergechrompos = mergechrompos.reset_index(drop=True)
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
        Negonly.to_csv(pcawg_dir + pcawg_histology + '/' + 'Normal_' + pcawg_sample + '.csv')
        pd_count = pd.DataFrame([len(SNVonly),len(MNVonly),len(indelonly),len(MEISVonly),len(Negonly)])
        pd_count.to_csv(tokenized_data + pcawg_histology + '/' + 'count_' + pcawg_sample + '.csv')