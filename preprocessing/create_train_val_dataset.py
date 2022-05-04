import numpy as np
import pandas as pd
import os
import pdb
import argparse


#ex: python create_train_val_dataset.py --total-fold 10 --train-split 9 --val-split 1 --class-info-file '/mnt/g/experiment/muat/dataset_utils/classinfo_tcga_epi.csv' --output-dir '/mnt/g/experiment/muat/dataset_utils/' --output-prefix 'tcgaepi' --alltumour-sample '/mnt/g/experiment/muat/extfile/all_tumoursamples_tcga.tsv'

def get_args():
        parser = argparse.ArgumentParser(description='create train-val split')

        parser.add_argument('--total-fold', type=int, default=10,
                        help='total fold')
        parser.add_argument('--train-split', type=int, default=9,
                        help='training split')
        parser.add_argument('--val-split', type=int, default=1,
                        help='validation split')
        parser.add_argument('--class-info-file', type=str, default=None,
                        help='class info file')
        parser.add_argument('--output-dir', type=str, default=None,
                        help='saving training-val files')
        parser.add_argument('--output-prefix', type=str, default=None,
                        help='saving training-val files')
        parser.add_argument('--alltumour-sample', type=str, default=None,
                        help='list of tumour samples')
        args = parser.parse_args()
        return args

if __name__ == '__main__':

    args = get_args()

    #requirements:
    total_fold = args.total_fold

    all_tumour_samples = args.alltumour_sample

    #output dir --> to projectdir/dataset_utils
    output_dir = args.output_dir

    allsamples = pd.read_csv(all_tumour_samples, sep='\t',index_col=0)
    selected_tumour = pd.read_csv(args.class_info_file, index_col=0)
    selected_tumour = selected_tumour['class_name'].to_list()
    
    allsamples = allsamples.loc[allsamples['tumour_type'].isin(selected_tumour)]
    allsamples = allsamples.sort_values(by=['tumour_type'])
    allsamples = allsamples.reset_index(drop=True)

    allsamples.columns = ['nm_class','samples']
    allsamples['samples'] = allsamples['samples']
    allsamples[['samples','ext']] = allsamples['samples'].str.split('.',expand=True)
    pd_allsamples = allsamples[['nm_class','samples']]

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
    
    trainAll.to_csv(output_dir + args.output_prefix + '_train.csv')
    valAll.to_csv(output_dir + args.output_prefix + '_val.csv')






