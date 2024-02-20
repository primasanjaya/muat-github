#!/usr/bin/env python
# coding: utf-8

import os 
import pandas as pd
import numpy as np
import pdb
import glob
import pdb
import math

import os
import glob

from preprocessing.dmm.preprocess3 import *

def preprocessing_fromdmm_all(args):

    #load all dictionary
    dictMutation = pd.read_csv(args.cwd + '/extfile/dictMutation.csv',index_col=0,low_memory=False)
    dictChpos = pd.read_csv(args.cwd + '/extfile/dictChpos.csv',index_col=0,low_memory=False)
    dictGES = pd.read_csv(args.cwd + '/extfile/dictGES.csv',index_col=0,low_memory=False)

    fns  = pd.read_csv(args.predict_filepath,sep='\t')['path'].to_list()

    for i in range(len(fns)):

        try:
            if args.gel:
                fn = fns[i]
                sample_name = fn[:-7]
                sample_name = sample_name.split('/')
                sample_name = sample_name[0] + '_'.join(sample_name[1:])
                filename = sample_name
            else:
                fn = fns[i]
                get_ext = fn[-4:]
                if get_ext == '.vcf':
                    sample_name = fn[:-4]
                else:
                    sample_name = fn[:-7]

                filename = sample_name.split('/')
                filename = filename[-1]

            #pdb.set_trace()

            readfile = filename + '.gc.genic.exonic.cs.tsv.gz'
            #pdb.set_trace()

            pd_data = pd.read_csv(args.tmp_dir + readfile,sep='\t',low_memory=False)

            ps = (pd_data['pos'] / 1000000).apply(np.floor).astype(int).astype(str)

            chrom = pd_data['chrom'].astype(str)

            chrompos = chrom + '_' + ps

            pd_data['chrompos'] = chrompos

            pd_data['ges'] = pd_data['genic'].astype(str) + '_' + pd_data['exonic'].astype(str) + '_' + pd_data['strand'].astype(str)

            mergetriplet = pd_data.merge(dictMutation, left_on='seq', right_on='triplet', how='left')
            mergeges = mergetriplet.merge(dictGES, left_on='ges', right_on='ges', how='left')
            mergeges = mergeges.rename(columns={"token": "gestoken"})
            mergechrompos = mergeges.merge(dictChpos, left_on='chrompos', right_on='chrompos', how='left')
            mergechrompos = mergechrompos.rename(columns={"token": "postoken"})

            mergeAlltoken =  mergechrompos[['triplettoken', 'postoken','gestoken','gc1kb','mut_type']]
            mergeAlltoken = mergeAlltoken.rename(columns={"mut_type" : "type"})

            NiSionly = mergeAlltoken.loc[mergeAlltoken['type']=='SNV']
            NiSionly = NiSionly.drop(columns=['type'])

            SNVonly = mergeAlltoken.loc[mergeAlltoken['type']=='MNV']
            SNVonly = SNVonly.drop(columns=['type'])

            indelonly = mergeAlltoken.loc[mergeAlltoken['type']=='indel']
            indelonly = indelonly.drop(columns=['type'])

            MEISVonly = mergeAlltoken.loc[mergeAlltoken['type'].isin(['MEI','SV'])]

            Normalonly = mergeAlltoken.loc[mergeAlltoken['type']=='Neg']
            Normalonly = Normalonly.drop(columns=['type'])

            NiSionly.to_csv(args.tmp_dir + 'SNV_' + filename + '.csv')
            SNVonly.to_csv(args.tmp_dir + 'MNV_' + filename + '.csv')
            indelonly.to_csv(args.tmp_dir + 'indel_' + filename + '.csv')
            MEISVonly.to_csv(args.tmp_dir + 'MEISV_' + filename + '.csv')
            Normalonly.to_csv(args.tmp_dir + 'Neg_' + filename + '.csv')

            pd_count = pd.DataFrame([len(NiSionly),len(SNVonly),len(indelonly),len(MEISVonly),len(Normalonly)])

            pd_count.to_csv(args.tmp_dir + 'count_' + filename + '.csv')
        except:
            pass

def preprocessing_fromdmm(args):

    #load all dictionary
    dictMutation = pd.read_csv(args.cwd + '/extfile/dictMutation.csv',index_col=0,low_memory=False)
    dictChpos = pd.read_csv(args.cwd + '/extfile/dictChpos.csv',index_col=0,low_memory=False)
    dictGES = pd.read_csv(args.cwd + '/extfile/dictGES.csv',index_col=0,low_memory=False)


    filename = args.output.split('/')
    filename = filename[-1]
    filename = strip_suffixes(filename, ['.tsv.gz'])

    readfile = filename + '.gc.genic.exonic.cs.tsv.gz'

    pd_data = pd.read_csv(args.tmp_dir + readfile,sep='\t',low_memory=False)

    ps = (pd_data['pos'] / 1000000).apply(np.floor).astype(int).astype(str)

    chrom = pd_data['chrom'].astype(str)

    chrompos = chrom + '_' + ps

    pd_data['chrompos'] = chrompos

    pd_data['ges'] = pd_data['genic'].astype(str) + '_' + pd_data['exonic'].astype(str) + '_' + pd_data['strand'].astype(str)

    mergetriplet = pd_data.merge(dictMutation, left_on='seq', right_on='triplet', how='left')
    mergeges = mergetriplet.merge(dictGES, left_on='ges', right_on='ges', how='left')
    mergeges = mergeges.rename(columns={"token": "gestoken"})
    mergechrompos = mergeges.merge(dictChpos, left_on='chrompos', right_on='chrompos', how='left')
    mergechrompos = mergechrompos.rename(columns={"token": "postoken"})

    mergeAlltoken =  mergechrompos[['triplettoken', 'postoken','gestoken','gc1kb','mut_type']]
    mergeAlltoken = mergeAlltoken.rename(columns={"mut_type" : "type"})

    NiSionly = mergeAlltoken.loc[mergeAlltoken['type']=='SNV']
    NiSionly = NiSionly.drop(columns=['type'])

    SNVonly = mergeAlltoken.loc[mergeAlltoken['type']=='MNV']
    SNVonly = SNVonly.drop(columns=['type'])

    indelonly = mergeAlltoken.loc[mergeAlltoken['type']=='indel']
    indelonly = indelonly.drop(columns=['type'])

    MEISVonly = mergeAlltoken.loc[mergeAlltoken['type'].isin(['MEI','SV'])]

    Normalonly = mergeAlltoken.loc[mergeAlltoken['type']=='Neg']
    Normalonly = Normalonly.drop(columns=['type'])

    NiSionly.to_csv(args.tmp_dir + 'SNV_' + filename + '.csv')
    SNVonly.to_csv(args.tmp_dir + 'MNV_' + filename + '.csv')
    indelonly.to_csv(args.tmp_dir + 'indel_' + filename + '.csv')
    MEISVonly.to_csv(args.tmp_dir + 'MEISV_' + filename + '.csv')
    Normalonly.to_csv(args.tmp_dir + 'Neg_' + filename + '.csv')

    pd_count = pd.DataFrame([len(NiSionly),len(SNVonly),len(indelonly),len(MEISVonly),len(Normalonly)])

    pd_count.to_csv(args.tmp_dir + 'count_' + filename + '.csv')

if __name__ == '__main__':

    test = preprocessing_fromvcf('G:/experiment/muat/data/raw/vcf/00b9d0e6-69dc-4345-bffd-ce32880c8eef.consensus.20160830.somatic.snv_mnv.vcf')



