import numpy as np
import torch
import os
import pdb

import argparse

#fixing nonF model

'''
weight type:
1: refixing from plain model.pthx (from full training in puhti)
2: training from new code (including args and type)
'''

def get_args():
        parser = argparse.ArgumentParser(description='PCAWG / TCGA experiment')

        # DATASET
        parser.add_argument('--cwd', type=str,help='project dir')
        parser.add_argument('--dataloader', type=str, default='pcawg',
                        help='dataloader setup, option: pcawg or tcga')
        # MODEL
        parser.add_argument('--arch', type=str, default=None,
                        help='architecture')
        # DIRECTORY
            #INPUT DATA
        parser.add_argument('--input-data-dir', type=str, default=None,
                        help='input data directory')
        parser.add_argument('--input-crossdata-dir', type=str, default=None,
                        help='output data directory')
        parser.add_argument('--input-newdata-dir', type=str, default=None,
                        help='input new data directory')

            #OUTPUT DATA
        parser.add_argument('--output-data-dir', type=str, default=None,
                        help='output data directory')
        parser.add_argument('--output-crossdata-dir', type=str, default=None,
                        help='output data directory')
        parser.add_argument('--output-newdata-dir', type=str, default=None,
                        help='output data directory')
        parser.add_argument('--tmp-dir', type=str, default=None,
                        help='temporary data directory')
        parser.add_argument('--output-prefix', type=str, default=None,
                        help='prefix of output data')
        
            # FILENAMES
        parser.add_argument('--input-filename', type=str, default=None,
                        help='input filename')
        parser.add_argument('--output-filename', type=str, default=None,
                        help='output filename')
            #CKPT SAVE
        parser.add_argument('--save-ckpt-dir', type=str, default=None,
                        help='save checkpoint dir')
            #CKPT LOAD
        parser.add_argument('--load-ckpt-dir', type=str, default=None,
                        help='load checkpoint dir')

        # HYPER PARAMS 
        parser.add_argument('--n-class', type=int, default=None,
                        help='number of class')
        parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size')
        parser.add_argument('--block-size', type=int, default=1000,
                        help='block of sequence')
        parser.add_argument('--context-length', type=int, default=3,
                        help='length of sequence')
        parser.add_argument('--n-layer', type=int, default=1,
                        help='attention layer')
        parser.add_argument('--n-head', type=int, default=8,
                        help='attention head')
        parser.add_argument('--n-emb', type=int, default=128,
                        help='embedding dimension')
        parser.add_argument('--tag', type=str, default='myexperiment',
                        help='tensorboardX tag')
        parser.add_argument('--fold', type=int, default=1, 
                            help='fold')

        #EXECUTION
        
        parser.add_argument('--train', action='store_true', default=False,
                            help='execute training')
        parser.add_argument('--predict', action='store_true', default=False,
                            help='execute prediction')

        parser.add_argument('--predict-new-data', action='store_true', default=False,
                            help='execute prediction from new data (PCAWG training-ready format)')
        
        parser.add_argument('--single-pred-vcf', action='store_true', default=False)

        parser.add_argument('--motif', action='store_true', default=False)
        parser.add_argument('--motif-pos', action='store_true', default=False)
        parser.add_argument('--motif-pos-ges', action='store_true', default=False)

        parser.add_argument('--get-muat-features', action='store_true', default=False)

        parser.add_argument('--num-mut', type=int, default=0,
                        help='sampling number of mutation')
        parser.add_argument('--frac', type=float, default=0,
                        help='sampling number of mutation based on data fraction')

        parser.add_argument('--mut-type', type=str, default='',
                        help='mutation type, only [SNV,SNV+MNV,SNV+MNV+indel,SNV+MNV+indel+SV/MEI,SNV+MNV+indel+SV/MEI+Neg] can be applied')
    
        parser.add_argument('--mutratio', type=str, default='',
                        help='mutation ratio per mutation type, sum of them must be one')
        parser.add_argument('--vis-attention', type=str, default='',
                        help='visualize attention values')

        #dmm_parser
        parser.add_argument('-v', '--verbose', type=int, help='Try to be more verbose')

        parser.add_argument('--mutation-coding', type=int, default=None,
                                help='Mutation coding table ("ref alt code"/line)')
        parser.add_argument('--config', help='Read parameters from a JSON file')
        parser.add_argument('--data-config',
                                help='Column specification for --input, --validation and --aux-data  [{}]')
        parser.add_argument('--random-seed', default=None, type=int, metavar='seed')
        parser.add_argument('--tmp')
        
        parser.add_argument('-i', '--input', action='append', metavar='dir(s)',
                                help='Either a directory with vcf/maf[.gz] files or a vcf/maf[.gz] file (-i may be given more than once)')
        parser.add_argument('-o', '--output', metavar='fn', help='Preprocessed mutation data')
        parser.add_argument('-r', '--reference', metavar='ref', help='Reference genome (fasta) [{}]')
        parser.add_argument('-k', '--context', help='Sequence context length (power of 2) [{}]', metavar='bp', type=int,default=8)
        parser.add_argument('-e', '--errors', metavar='fn',
                                help='File where to log errors [{}]')
        parser.add_argument('--no-ref-preload', help='Use samtools to read reference on demand (slow but fast startup) [false]',
                                action='store_true')
        parser.add_argument('--no-filter', help='Process all variants [default=only PASS/. variants]',
                                action='store_true')
        parser.add_argument('--sample-id', help='Sample identifier column name in MAF file')
        parser.add_argument('-n', '--generate_negatives', help='Ratio of negative to positive examples [{}]. Two passes on data are required for n>0.', type=float)
        parser.add_argument('--median-variant-type-negatives', action='store_true',
                                help='Generate median number of each variant type as negative examples for each sample')
        parser.add_argument('--median-variant-type-file', help='Load median variant numbers from a file')
        parser.add_argument('--negative-generation-mode', help='[generate] output in one go (default), [augment] input files or [process] augmented files', default='generate')
        parser.add_argument('--info-column', help='Input column name to write toutputo output (MAF input only). May be specified more than once.', action='append')
        parser.add_argument('--report-interval', help='Interval to report number of variants processed',
                                type=int)
        parser.add_argument('--array-jobs', help='How many array jobs in total', type=int)
        parser.add_argument('--array-index', help='Index of this job', type=int)
        parser.add_argument('--nope', help='Only one variant per output sequence', action='store_true')
        parser.add_argument('--no-overwrite', help='Do not overwrite if output exists', action='store_true')

        args = parser.parse_args()
        return args

def fixing_args(splitmdl,args):
    args.arch = splitmdl[3]

    if "pcawg" in splitmdl[0]:
        args.dataset = 'pcawg'
        args.n_class = 24
    elif "tcga" in splitmdl[0]:
        args.dataset = 'tcga'    
        args.n_class = 23

    if splitmdl[1] == '10000':
        args.mut_type = 'SNV'
        args.mutratio = '1-0-0-0-0'
    elif splitmdl[1] == '11000':
        args.mut_type = 'SNV+MNV'
        args.mutratio = '0.5-0.5-0-0-0'
    elif splitmdl[1] == '11100':
        args.mut_type = 'SNV+MNV+indel'
        args.mutratio = '0.4-0.3-0.3-0-0'
    elif splitmdl[1] == '11110':
        args.mut_type = 'SNV+MNV+indel+SV/MEI'
        args.mutratio = '0.3-0.3-0.2-0.2-0'
    elif splitmdl[1] == '11111':
        args.mut_type = 'SNV+MNV+indel+SV/MEI+Neg'
        args.mutratio = '0.25-0.25-0.25-0.15-0.1'

    if splitmdl[2] == 'tripkon':
        args.motif = True
    elif splitmdl[2] == 'wpos':
        args.motif_pos = True        
        #pdb.set_trace()
    elif splitmdl[2] == 'wposges':
        args.motif_pos_ges = True

    block_size = int(splitmdl[4][2:])
    layer = int(splitmdl[5][2:])
    head = int(splitmdl[6][2:])
    emb = int(splitmdl[7][2:])
    cl = int(splitmdl[8][2:])

    args.block_size = block_size
    args.n_layer = layer
    args.n_head = head
    args.n_emb = emb
    args.context_length = cl

    return args

def update_args(args,old_args):
    args.arch = old_args.arch
    args.block_size = old_args.block_size
    args.dataloader = old_args.dataloader
    args.fold = old_args.fold
    args.mut_type = old_args.mut_type
    args.motif = old_args.motif
    args.motif_pos = old_args.motif_pos
    args.motif_pos_ges = old_args.motif_pos_ges
    args.mutratio = old_args.mutratio
    args.n_class =  old_args.n_class
    args.n_emb = old_args.n_emb
    args.n_head = old_args.n_head
    args.n_layer =  old_args.n_layer

    return args

args = get_args()

ckpt_dir = '/mnt/g/experiment/muat/bestckpt/raw/'

all_ckptdir = os.listdir(ckpt_dir)

device = 'cpu'
if torch.cuda.is_available():
    device = torch.cuda.current_device()

for mdl in all_ckptdir:
    if device == 'cpu':
        allckpt = torch.load(ckpt_dir + mdl +'/model.pthx',map_location=device)
    else:
        allckpt = torch.load(ckpt_dir + mdl +'/model.pthx')

    if len(allckpt) == 2:
        print('this ' + mdl + 'is already in new format')
    else:
        weight = allckpt

        #fix args 
        splitmdl = mdl.split('_')
        args = fixing_args(splitmdl,args)

        concateweight = [weight,args,1]

        torch.save(concateweight, (ckpt_dir + mdl  + '/renew_model.pthx'))

