
import numpy as np
import pandas as pd
import os
import pdb
import math

#requirements:
metadata = pd.read_csv('/csc/epitkane/projects/muat/extfile/metadata_icgc_pcawg.tsv',sep='\t',index_col=0) 
dictMutation = pd.read_csv('/csc/epitkane/projects/muat/extfile//dictMutation.csv',index_col=0)
dictChpos = pd.read_csv('/csc/epitkane/projects/muat/extfile/dictChpos.csv',index_col=0)
dictGES = pd.read_csv('/csc/epitkane/projects/muat/extfile/dictGES.csv',index_col=0)

icgc_data_dir = '/csc/epitkane/data/ICGC/release_28/Projects/'

#output dir --> new pcawg directory
pcawg_dir = '/csc/epitkane/data/ICGC/release_28/pcawg_recreate/'

all_project = list(set(metadata['project_code'].to_list()))
all_project.sort()

count_proj = 0
#loop every project code
#pdb.set_trace()
for i in range(0,len(all_project)):
#for i in range(0,len(all_project)):
    
    count_proj = count_proj + 1
    print(curr_project + ' ' + str(count_proj) + '/' + str(len(all_project)))
    somatic_files = pd.read_csv(icgc_data_dir + curr_project+ '/somatic_mutations.' + curr_project + '.dmm.k256.annotated.tsv.gz',sep='\t')
    
    row = metadata.loc[metadata['project_code'] == curr_project]
    
    #loop every samples
    for j in range(0,len(row)):
        try:
            sample_loop = row.iloc[j]['submitted_sample_id']
            pcawg_sample = row.iloc[j]['tumor_wgs_aliquot_id']
            pcawg_histology = row.iloc[j]['histology']
            #create pcawg histology dir
            os.makedirs(pcawg_dir + pcawg_histology, exist_ok = True)
            
            somatic_files_persamples = somatic_files.loc[somatic_files['sample']==sample_loop]
            
            #rename icgc_sample with pcawg sample
            somatic_files_persamples['sample'] = pcawg_sample
            
            #tokenization process

            #1) simplifing mutation information
            simplified_mutation = []
            for k in range(0,len(somatic_files_persamples)):
                onemut = somatic_files_persamples.iloc[k]
                trp = onemut['seq'][int(len(onemut['seq'])/2)-1:int(len(onemut['seq'])/2)+2]
                ps = math.floor(onemut['pos'] / 1000000)
                chompos = str(onemut['chrom']) + '_' + str(ps)
                rt = onemut['gc1kb']
                ges = str(onemut['genic']) + '_' + str(onemut['exonic']) + '_' + str(onemut['strand'])
                try:
                    typ = dictMutation.loc[dictMutation['triplet']==trp]['mut_type'].values[0]
                except:
                    typ = 'Neg'

                tuple_onerow = (trp,chompos,rt,ges,typ)
                simplified_mutation.append(tuple_onerow)
            pd_new = pd.DataFrame(simplified_mutation)
            pd_new.columns = ['seq','chrompos','rt','ges','mut_type'] 
            #save simplified mutation
            new_files = 'new_' + pcawg_sample + '.csv'
            pd_new.to_csv(pcawg_dir + pcawg_histology + '/' + new_files)
            
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
            token_files = 'token_' + pcawg_sample + '.csv'
            token_data.to_csv(pcawg_dir + pcawg_histology + '/' + token_files)
            
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
            SNVonly.to_csv(pcawg_dir + pcawg_histology + '/' + 'SNV_' + pcawg_sample + '.csv')
            MNVonly.to_csv(pcawg_dir + pcawg_histology + '/' + 'MNV_' + pcawg_sample + '.csv')
            indelonly.to_csv(pcawg_dir + pcawg_histology + '/' + 'indel_' + pcawg_sample + '.csv')
            MEISVonly.to_csv(pcawg_dir + pcawg_histology + '/' + 'MEISV_' + pcawg_sample + '.csv')
            Negonly.to_csv(pcawg_dir + pcawg_histology + '/' + 'Normal_' + pcawg_sample + '.csv')
            pd_count = pd.DataFrame([len(SNVonly),len(MNVonly),len(indelonly),len(MEISVonly),len(Negonly)])
            pd_count.to_csv(pcawg_dir + pcawg_histology + '/' + 'count_' + pcawg_sample + '.csv')
        except:
            pass