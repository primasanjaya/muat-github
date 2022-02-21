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

def preprocessing_fromdmm(args):

    #load all dictionary
    dictMutation = pd.read_csv(args.cwd + '/extfile/dictMutation.csv',index_col=0)
    dictChpos = pd.read_csv(args.cwd + '/extfile/dictChpos.csv',index_col=0)
    dictGES = pd.read_csv(args.cwd + '/extfile/dictGES.csv',index_col=0)


    filename = args.output.split('/')
    filename = filename[-1]
    filename = strip_suffixes(filename, ['.tsv.gz'])

    readfile = filename + '.gc.genic.exonic.cs.tsv.gz'

    pd_data = pd.read_csv(args.tmp_dir + readfile,sep='\t')

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

    NiSionly.to_csv(args.tmp_dir + 'SNV_new_' + filename + '.csv')
    SNVonly.to_csv(args.tmp_dir + 'MNV_new_' + filename + '.csv')
    indelonly.to_csv(args.tmp_dir + 'indel_new_' + filename + '.csv')
    MEISVonly.to_csv(args.tmp_dir + 'MEISV_new_' + filename + '.csv')
    Normalonly.to_csv(args.tmp_dir + 'Neg_new_' + filename + '.csv')

    pd_count = pd.DataFrame([len(NiSionly),len(SNVonly),len(indelonly),len(MEISVonly),len(Normalonly)])

    pd_count.to_csv(args.tmp_dir + 'count_new_' + filename + '.csv')

if __name__ == '__main__':

    test = preprocessing_fromvcf('G:/experiment/muat/data/raw/vcf/00b9d0e6-69dc-4345-bffd-ce32880c8eef.consensus.20160830.somatic.snv_mnv.vcf')



