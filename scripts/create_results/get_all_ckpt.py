import numpy as np
import pandas as pd
import pdb
import os
import math
import argparse
import glob


def get_args():
    parser = argparse.ArgumentParser(description='preprocessing args')
    parser.add_argument('--ckpt-dir', type=str,help='ckpt dir')
    parser.add_argument('--output-file', type=str,help='output-dir')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = get_args()

    #edit the directory

    ckpt_folder = os.listdir(args.ckpt_dir)

    experiment_list = []

    for i in range(len(ckpt_folder)):

        allfiles = os.listdir(args.ckpt_dir + ckpt_folder[i])

        listprf = [x for x in allfiles if x[-7:] == "prf.csv"]

        listprf.sort(reverse=True)

        try:
            prf = pd.read_csv(args.ckpt_dir + ckpt_folder[i] + '/' + listprf[0],index_col=0)
            meanall = prf.mean().values.tolist()
            model_run =  ckpt_folder[i]

            allrow = [model_run] + meanall
            tupl_mean = tuple(allrow)
            experiment_list.append(tupl_mean)
        except:
            #pdb.set_trace()
            pass

pd_exp = pd.DataFrame(experiment_list)
pd_exp.columns =  ['model_run'] + prf.columns.tolist()

pd_exp = pd_exp.sort_values(by=['top1'],ascending=False)

pd_exp.to_csv(args.output_file)
