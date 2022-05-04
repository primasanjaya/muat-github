# make deterministic
from mingpt.utils import set_seed
set_seed(42)
#frompc

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset

from mingpt.model import *

from mingpt.trainer import Trainer, TrainerConfig
from mingpt.utils import sample

import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)
import pdb

from dataset.tcga_dataset import TCGA
from dataset.tcga_conv_dataset import TCGAConv
from dataset.pcawg_conv_dataset import *
from dataset.pcawg_dataset import PCAWG
from dataset.pcawg_emb_dataset import PCAWGEmb
from dataset.pcawg_sepdataset import PCAWGSep
from dataset.pcawg_2stream import PCAWG2Stream
from dataset.tcgadisttabletoemb_dataset import TCGADist
from dataset.tcgamutdist_dataset import TCGAMutDist
from dataset.tcgamutdistasone_dataset import TCGAMutDistasOne
from dataset.tcgapcawg_dataset import TCGAPCAWG
from dataset.newtcgapcawg_dataset import NewTCGAPCAWG
from dataset.finaltcgapcawg_dataset import FinalTCGAPCAWG

from mingpt.bert import *
from preprocessing.dmm.dmm import *
from preprocessing.fromvcffiles import *

import argparse
import os
import pandas as pd

def translate_args(args):

    cwd = os.getcwd()
    args.cwd = cwd

    args.mutation_coding = cwd + '/preprocessing/dmm/data/mutation_codes_sv.tsv'
    args.input = args.data_dir

    args.output = cwd + '/data/raw/out/00b9d0e6-69dc-4345-bffd-ce32880c8eef.consensus.20160830.somatic.snv_mnv.tsv.gz' 

    args.reference = '/csc/epitkane/data/ref_genomes/hs37d5_1000GP/hs37d5_1000GP.fa'
    args.context = 8

    args.sample_id = 'submitted_sample_id'

    args.tmp = cwd + '/data/raw/tmp/'
    args.verbose = 1
    args.generate_negatives = 1
    args.report_interval = 100000

    return args


def get_args():
        parser = argparse.ArgumentParser(description='TCGA / PEACOCK experiment')

        # DATASET
        parser.add_argument('--cwd', type=str,help='project dir')

        parser.add_argument('--dataset', type=str, default='pcawg',
                        help='dataset')
        # MODEL
        parser.add_argument('--arch', type=str, default=None,
                        help='architecture')
        # DIRECTORY
        parser.add_argument('--data-dir', type=str, default=None,
                        help='data directory')
        parser.add_argument('--crossdata-dir', type=str, default=None,
                        help='data directory')
        parser.add_argument('--adddata-dir', type=str, default=None,
                        help='data directory')

        parser.add_argument('--n-class', type=int, default=None,
                        help='number of class')

        parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size')

        parser.add_argument('--block-size', type=int, default=1000,
                        help='block of sequence')

        parser.add_argument('--context-length', type=int, default=256,
                        help='length of sequence')
        parser.add_argument('--n-layer', type=int, default=1,
                        help='attention layer')
        parser.add_argument('--n-head', type=int, default=8,
                        help='attention head')
        parser.add_argument('--n-emb', type=int, default=128,
                        help='embedding dimension')
        parser.add_argument('--n-vocab-type', type=int, default=1,
                        help='embedding dimension')

        parser.add_argument('--tag', type=str, default='myexperiment',
                        help='dataset')
        
        parser.add_argument('--train', action='store_true', default=False)
        parser.add_argument('--predict', action='store_true', default=False)
        parser.add_argument('--trainbp', action='store_true', default=False)
        parser.add_argument('--vis-weight', action='store_true', default=False)
        parser.add_argument('--top-weight', action='store_true', default=False)

        parser.add_argument('--visval', action='store_true', default=False)


        parser.add_argument('--single-predict', action='store_true', default=False)
        parser.add_argument('--create-dataset', action='store_true', default=False)
        parser.add_argument('--two-streams', action='store_true', default=False)
        parser.add_argument('--three-streams', action='store_true', default=False)

        parser.add_argument('--filter', action='store_true', default=False)

        parser.add_argument('--bert', action='store_true', default=False)
        parser.add_argument('--withclass', action='store_true', default=False)
        parser.add_argument('--default', action='store_true', default=False)
        parser.add_argument('--addposition', action='store_true', default=False)
        parser.add_argument('--oneDhot', action='store_true', default=False)
        parser.add_argument('--addorder', action='store_true', default=False)
        parser.add_argument('--addtoken', action='store_true', default=False)
        parser.add_argument('--addtriplet', action='store_true', default=False)
        parser.add_argument('--addtriplettoken', action='store_true', default=False)
        parser.add_argument('--addgestoken', action='store_true', default=False)
        parser.add_argument('--addrt', action='store_true', default=False)
        parser.add_argument('--addlongcontext', action='store_true', default=False)
        parser.add_argument('--tokenizedlongcontext', action='store_true', default=False)
        parser.add_argument('--ohlongcontext', action='store_true', default=False)
        parser.add_argument('--flattenohlongcontext', action='store_true', default=False)
        parser.add_argument('--addpostoken', action='store_true', default=False)
        parser.add_argument('--addrttoken', action='store_true', default=False)
        parser.add_argument('--balance', action='store_true', default=False)

        parser.add_argument('--l1', action='store_true', default=False)

        parser.add_argument('--fold', type=int, default=1, 
                            help='number of mutation')

        parser.add_argument('--output-mode', type=str, default='token',help='dataset')

        parser.add_argument('--rbm', action='store_true', default=False)

        parser.add_argument('--newtraining', action='store_true', default=False)
        parser.add_argument('--newpredict', action='store_true', default=False)
        parser.add_argument('--newpredict2', action='store_true', default=False)
        parser.add_argument('--normal', action='store_true', default=False)

        parser.add_argument('--freezeemb', action='store_true', default=False)

        parser.add_argument('--predictvis', action='store_true', default=False)

        parser.add_argument('--crossdata', action='store_true', default=False)

        parser.add_argument('--nummut', type=int, default=0,
                        help='number of mutation')
        parser.add_argument('--frac', type=float, default=0,
                        help='frac')

        parser.add_argument('--mutratio', type=str, default='',
                        help='mutation ratio')

        parser.add_argument('--spectral', action='store_true', default=False)
        parser.add_argument('--finalpredict', action='store_true', default=False)

        parser.add_argument('--finalpredictnewdata', action='store_true', default=False)
        parser.add_argument('--single-pred-vcf', action='store_true', default=False)


        parser.add_argument('--vis-attention', action='store_true', default=False)


        #dmm_parser
        parser.add_argument('-v', '--verbose', type=int, help='Try to be more verbose')
        parser.add_argument('--mutation-coding', help='Mutation coding table ("ref alt code"/line) [{}]'.format(\
                                defaults['mutation_coding']), metavar='fn')
        parser.add_argument('--config', help='Read parameters from a JSON file')
        parser.add_argument('--data-config',
                                help='Column specification for --input, --validation and --aux-data  [{}]'.format(\
                                defaults['data_config']))
        parser.add_argument('--random-seed', default=None, type=int, metavar='seed')
        parser.add_argument('--tmp')
        
        parser.add_argument('-i', '--input', action='append', metavar='dir(s)',
                                help='Either a directory with vcf/maf[.gz] files or a vcf/maf[.gz] file (-i may be given more than once)')
        parser.add_argument('-o', '--output', metavar='fn', help='Preprocessed mutation data')
        parser.add_argument('-r', '--reference', metavar='ref', help='Reference genome (fasta) [{}]'.format(\
                                defaults['reference']))
        parser.add_argument('-k', '--context', help='Sequence context length (power of 2) [{}]'.format(\
                                defaults['context']), metavar='bp', type=int,default=8)
        parser.add_argument('-e', '--errors', metavar='fn',
                                help='File where to log errors [{}]'.format(defaults['errors']))
        parser.add_argument('--no-ref-preload', help='Use samtools to read reference on demand (slow but fast startup) [false]',
                                action='store_true')
        parser.add_argument('--no-filter', help='Process all variants [default=only PASS/. variants]',
                                action='store_true')
        parser.add_argument('--sample-id', help='Sample identifier column name in MAF file')
        parser.add_argument('-n', '--generate_negatives', help='Ratio of negative to positive examples [{}]. Two passes on data are required for n>0.'.format(\
                                defaults['negative_ratio']), type=float)
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

def get_dataloader(args,train_val,load):

    if args.dataset == 'finalpcawg' or args.dataset == 'wgspcawg':
        if train_val=='training':
            dataloader_class = FinalTCGAPCAWG(dataset_name = args.dataset, 
                                data_dir=args.data_dir, 
                                mode='training', 
                                curr_fold=args.fold, 
                                block_size=args.block_size, 
                                load=False,
                                mutratio = args.mutratio,
                                addtriplettoken=args.addtriplettoken,
                                addpostoken=args.addpostoken,
                                addgestoken=args.addgestoken,
                                addrt=args.addrt,
                                nummut = args.nummut,
                                frac = args.frac,
                                crossdata = args.crossdata,
                                crossdatadir = args.crossdata_dir,
                                adddatadir = args.adddata_dir
                                )

        elif train_val=='validation':
            dataloader_class = FinalTCGAPCAWG(dataset_name = args.dataset, 
                                data_dir=args.data_dir, 
                                mode='validation', 
                                curr_fold=args.fold, 
                                block_size=args.block_size, 
                                load=False,
                                mutratio = args.mutratio,
                                addtriplettoken=args.addtriplettoken,
                                addpostoken=args.addpostoken,
                                addgestoken=args.addgestoken,
                                addrt=args.addrt,
                                nummut = args.nummut,
                                frac = args.frac,
                                crossdata = args.crossdata,
                                crossdatadir = args.crossdata_dir,
                                adddatadir = args.adddata_dir)

    elif args.dataset == 'finaltcga' or args.dataset == 'westcga':
        if train_val=='training':
            dataloader_class = FinalTCGAPCAWG(dataset_name = args.dataset, 
                                data_dir=args.data_dir, 
                                mode='training', 
                                curr_fold=args.fold, 
                                block_size=args.block_size, 
                                load=False,
                                mutratio = args.mutratio,
                                addtriplettoken=args.addtriplettoken,
                                addpostoken=args.addpostoken,
                                addgestoken=args.addgestoken,
                                addrt=args.addrt,
                                nummut = args.nummut,
                                frac = args.frac,
                                crossdata = args.crossdata,
                                crossdatadir = args.crossdata_dir,
                                adddatadir = args.adddata_dir)

        elif train_val=='validation':
            dataloader_class = FinalTCGAPCAWG(dataset_name = args.dataset, 
                                data_dir=args.data_dir, 
                                mode='validation', 
                                curr_fold=args.fold, 
                                block_size=args.block_size, 
                                load=False,
                                mutratio = args.mutratio,
                                addtriplettoken=args.addtriplettoken,
                                addpostoken=args.addpostoken,
                                addgestoken=args.addgestoken,
                                addrt=args.addrt,
                                nummut = args.nummut,
                                frac = args.frac,
                                crossdata = args.crossdata,
                                crossdatadir = args.crossdata_dir,
                                adddatadir = args.adddata_dir)
                    
    return dataloader_class

def get_model(args,mconf):
        if args.arch == 'GPTConv':
                model = GPTConv(mconf)
        elif args.arch == 'GPTConvDeeper':
                model = GPTConvDeeper(mconf)
        elif args.arch == 'GPTNonPosition':
                model = GPTNonPosition(mconf)
        elif args.arch == 'CTransformer':
                model = CTransformer(mconf)
        elif args.arch == 'ConvTransformer':
                model = ConvTransformer(mconf)
        elif args.arch == 'Conv2DTransformer':
                model = Conv2DTransform
        elif args.arch == 'Transformer2Stream':
                model = Transformer2Stream(mconf)
        elif args.arch == 'CTransformerDNN':
                model = CTransformerDNN(mconf)
        elif args.arch == 'CTransformerMutDist':
                model = CTransformerMutDist(mconf)
        elif args.arch == 'SimpleAttention':
                model = SimpleAttention(mconf)
        elif args.arch == 'BertForSequenceClassification':
                model = BertForSequenceClassification(mconf)
        elif args.arch == 'BertwithPosition':
                model = BertwithPosition(mconf)
        elif args.arch == 'CTransformerWithPaddingIDX':
                model = CTransformerWithPaddingIDX(mconf)
        elif args.arch == 'Conv2DTransformerOnehot':
                model = Conv2DTransformerOnehot(mconf)
        elif args.arch == 'CTransformerWithPaddingIDXandfirstvec':
                model = CTransformerWithPaddingIDXandfirstvec(mconf)
        elif args.arch == 'Conv2DTransformerOnehotDeeper':
                model = Conv2DTransformerOnehotDeeper(mconf)
        elif args.arch == 'DNNTransformerOnehotDeeper':
                model = DNNTransformerOnehotDeeper(mconf)
        elif args.arch == 'CTransformerWithPosition':
                model = CTransformerWithPosition(mconf)
        elif args.arch == 'CTransformerWithPositionConcate':
                model = CTransformerWithPositionConcate(mconf)
        elif args.arch == 'DNNTransformerOnehotDeeperwithPosition':
                model = DNNTransformerOnehotDeeperwithPosition(mconf)
        elif args.arch == 'DNNTransformerOnehotDeeperwithPositionwithOrder':
            model = DNNTransformerOnehotDeeperwithPositionwithOrder(mconf)
        elif args.arch == 'CTransformerDNNWithPositionConcateToken':
            model = CTransformerDNNWithPositionConcateToken(mconf)
        elif args.arch == 'CTransformerDNNWithPositionConcateTokenSep':
            model = CTransformerDNNWithPositionConcateTokenSep(mconf)
        elif args.arch == 'CTransformerRBMWithPositionConcate':
            model = CTransformerRBMWithPositionConcate(mconf)
        elif args.arch == 'TripletPositionTokenandOnehot':
            model = TripletPositionTokenandOnehot(mconf) 
        elif args.arch == 'PositionToken':
            model = PositionToken(mconf) 
        elif args.arch == 'TripletPositionTokenandOnehotConcAfter':
            model = TripletPositionTokenandOnehotConcAfter(mconf)
        elif args.arch == 'TripletPositionRTToken':
            model = TripletPositionRTToken(mconf)
        elif args.arch == 'FullConvTransformer':
            model = FullConvTransformer(mconf)
        elif args.arch == 'TripletPositionTokenBest':
            model = TripletPositionTokenBest(mconf)
        elif args.arch == 'TripletPositionTokenRT':
            model = TripletPositionTokenRT(mconf)
        elif args.arch == 'EmbFC':
            model = EmbFC(mconf)    
        elif args.arch == 'TripletPositionTokenOldBest':
            model = TripletPositionTokenOldBest(mconf)
        elif args.arch == 'CTransformerPCAWGtoTCGA_TPGES':
            model = CTransformerPCAWGtoTCGA_TPGES(mconf)
        elif args.arch == 'CTransformerPCAWGtoTCGA_T':
            model = CTransformerPCAWGtoTCGA_T(mconf)
        elif args.arch == 'TripletPosition':
            model = TripletPosition(mconf)    
        elif args.arch == 'TripletPositionGES':
            model = TripletPositionGES(mconf)
        elif args.arch == 'TripletPositionGESRT':
            model = TripletPositionGESRT (mconf)   
        elif args.arch == 'TripletPositionF':
            model = TripletPositionF(mconf)    
        elif args.arch == 'TripletPositionGESF':
            model = TripletPositionGESF(mconf)
        elif args.arch == 'CTransformerF':
            model = CTransformerF(mconf)
        elif args.arch == 'EmbFCPos':
            model = EmbFCPos(mconf)
        elif args.arch == 'EmbFCPosGES':
            model = EmbFCPosGES(mconf)

        return model

def fold_split(args):

        num_class = os.listdir(args.data_dir)
        class_name = [i for i in num_class if len(i.split('.'))==1]
        class_name = sorted(class_name)

        num_samples = []

        for i in class_name:
                ns = len(os.listdir(args.data_dir+i))
                num_samples.append(ns)

        d = {'class_name':class_name,'n_samples':num_samples}
        pd_class_info = pd.DataFrame(d)
        
        folds=10

        class_used = pd_class_info.loc[pd_class_info['n_samples']>=folds]
        class_used = class_used.rename_axis('class_index').reset_index()
        class_used.to_csv(args.data_dir + 'sample_info_' + args.dataset + '.csv', index=False)

        num_class=len(class_used)

        tuple_list = []

        for nm_class in class_used['class_name']:
                num_sample = class_used.loc[class_used['class_name']==nm_class]['n_samples'].values[0]
                class_idx = class_used.loc[class_used['class_name']==nm_class]['class_index'].values[0]
                samples = os.listdir(args.data_dir+nm_class)
                count_split = 0

                for i in range(0,num_sample):
                        count_split = count_split+1
                        if count_split > folds:
                                count_split = 1

                        tuple_onerow = tuple([nm_class,class_idx,samples[i],count_split])
                        tuple_list.append(tuple_onerow)
                
        all_split = pd.DataFrame(tuple_list,columns = ['class_name','class_index','name_samples','split'])

        test_split = pd.DataFrame(columns = all_split.columns)
        train_split = pd.DataFrame(columns = all_split.columns)
        validation_split = pd.DataFrame(columns = all_split.columns)

        for i in range(1,folds):
            test = all_split.loc[all_split['split']==i]
            train = all_split.loc[all_split['split']!=i]
            split_min = i + 1
            if split_min >= folds:
                split_min = 1
            validation = train.loc[train['split']==split_min]
            train = train.loc[train['split']!=split_min]
            train['split'] = i
            validation['split'] = i

            test_split = test_split.append(test)
            validation_split = validation_split.append(validation)
            train_split = train_split.append(train)

        train_split.to_csv(args.data_dir + 'train_split.csv', index=False)
        validation_split.to_csv(args.data_dir + 'validation_split.csv', index=False)
        test_split.to_csv(args.data_dir + 'test_split.csv', index=False)


if __name__ == '__main__':

        best_accuracy=0

        args = get_args()

        if args.train:

                #class_info = fold_split(args)

                block_size = args.block_size # spatial extent of the model for its context
                train_dataset = get_dataloader(args=args,train_val='training',load= not args.create_dataset)
                validation_dataset = get_dataloader(args=args,train_val='validation',load= not args.create_dataset)

                if args.bert:
                    if args.default:
                        mconf = BertConfig(vocab_size_or_config_json_file = train_dataset.vocab_size,num_class=args.n_class)
                    else:
                        if args.addposition:
                            mconf = BertConfig(vocab_size_or_config_json_file = train_dataset.vocab_size,num_class=args.n_class,num_hidden_layers=args.n_layer,hidden_size=args.n_emb,num_attention_heads=args.n_head,type_vocab_size=args.n_vocab_type,position_size=train_dataset.position_size)
                        else:    
                            mconf = BertConfig(vocab_size_or_config_json_file = train_dataset.vocab_size,num_class=args.n_class,num_hidden_layers=args.n_layer,hidden_size=args.n_emb,num_attention_heads=args.n_head,type_vocab_size=args.n_vocab_type)

                    model = get_model(args,mconf)

                    string_logs = f"{args.tag}_{args.arch}_bs{args.block_size:.0f}_nl{args.n_layer:.0f}_nh{args.n_head:.0f}_ne{args.n_emb:.0f}_cl{args.context_length:.0f}/"
                    tconf = TrainerConfig(max_epochs=150, batch_size=1, learning_rate=0.001,
                            lr_decay=True, warmup_tokens=1*150, final_tokens=150*len(train_dataset)*block_size,
                            num_workers=1,string_logs=string_logs, args=args)
                    trainer = Trainer(model, train_dataset, validation_dataset, tconf)
                    trainer.bert_train()

                if args.rbm:
                    num_class=args.n_class
                    mconf = GPTConfig(vocab_size=train_dataset.vocab_size, block_size=block_size,num_class=num_class,
                    n_layer=args.n_layer,n_head=args.n_head, n_embd=args.n_emb,context_length=args.context_length,conv_filter=256)

                    if args.addposition:
                        mconf = GPTConfig(vocab_size=train_dataset.vocab_size, block_size=block_size,num_class=num_class,
                    n_layer=args.n_layer,n_head=args.n_head, n_embd=args.n_emb,context_length=args.context_length,conv_filter=256,position_size=train_dataset.position_size)

                    model = get_model(args,mconf)

                    string_logs = f"{args.tag}_{args.arch}_bs{mconf.block_size:.0f}_nl{mconf.n_layer:.0f}_nh{mconf.n_head:.0f}_ne{mconf.n_embd:.0f}_cl{mconf.context_length:.0f}/"

                    tconf = TrainerConfig(max_epochs=150, batch_size=1, learning_rate=6e-4,
                            lr_decay=True, warmup_tokens=1*150, final_tokens=150*len(train_dataset)*block_size,
                            num_workers=1,string_logs=string_logs, args=args)
                    trainer = Trainer(model, train_dataset, validation_dataset, tconf)

                    output_mode = args.output_mode.split("_")

                    if len(output_mode)>1:
                        trainer.multi_stream_rbm(len(output_mode))
                    else:
                        trainer.basic_train()
                else:
                    num_class=args.n_class
                    mconf = GPTConfig(vocab_size=train_dataset.vocab_size, block_size=block_size,num_class=num_class,
                    n_layer=args.n_layer,n_head=args.n_head, n_embd=args.n_emb,context_length=args.context_length,conv_filter=256)

                    if args.addposition:
                        mconf = GPTConfig(vocab_size=train_dataset.vocab_size, block_size=block_size,num_class=num_class,
                    n_layer=args.n_layer,n_head=args.n_head, n_embd=args.n_emb,context_length=args.context_length,conv_filter=256,position_size=train_dataset.position_size)

                    model = get_model(args,mconf)

                    string_logs = f"{args.tag}_{args.arch}_bs{mconf.block_size:.0f}_nl{mconf.n_layer:.0f}_nh{mconf.n_head:.0f}_ne{mconf.n_embd:.0f}_cl{mconf.context_length:.0f}/"

                    tconf = TrainerConfig(max_epochs=150, batch_size=1, learning_rate=6e-4,
                            lr_decay=True, warmup_tokens=1*150, final_tokens=150*len(train_dataset)*block_size,
                            num_workers=1,string_logs=string_logs, args=args)
                    trainer = Trainer(model, train_dataset, validation_dataset, tconf)

                    output_mode = args.output_mode.split("_")

                    if len(output_mode)>1:
                        trainer.multi_stream(len(output_mode))
                    else:
                        trainer.basic_train()

        if args.newtraining:
            block_size = args.block_size # spatial extent of the model for its context
            train_dataset = get_dataloader(args=args,train_val='training',load= not args.create_dataset)
            validation_dataset = get_dataloader(args=args,train_val='validation',load= not args.create_dataset)

            mconf = GPTConfig(vocab_size=train_dataset.vocab_size, block_size=block_size,num_class=args.n_class,
                    n_layer=args.n_layer,n_head=args.n_head, n_embd=args.n_emb,context_length=args.context_length,conv_filter=256)
                    
            if args.addpostoken:
                mconf = GPTConfig(vocab_size=train_dataset.vocab_size, block_size=block_size,num_class=args.n_class,
                    n_layer=args.n_layer,n_head=args.n_head, n_embd=args.n_emb,context_length=args.context_length,conv_filter=256,position_size=train_dataset.position_size,rt_size = train_dataset.rt_size)

            if args.addgestoken:
                mconf = GPTConfig(vocab_size=train_dataset.vocab_size, block_size=block_size,num_class=args.n_class,
                n_layer=args.n_layer,n_head=args.n_head, n_embd=args.n_emb,context_length=args.context_length,conv_filter=256,position_size=train_dataset.position_size, ges_size =  train_dataset.ges_size,rt_size = train_dataset.rt_size)

            model = get_model(args,mconf)

            string_logs = f"{args.tag}_{args.arch}_bs{mconf.block_size:.0f}_nl{mconf.n_layer:.0f}_nh{mconf.n_head:.0f}_ne{mconf.n_embd:.0f}_cl{mconf.context_length:.0f}/"

            tconf = TrainerConfig(max_epochs=150, batch_size=1, learning_rate=6e-4,
                    lr_decay=True, warmup_tokens=1*150, final_tokens=150*len(train_dataset)*block_size,
                    num_workers=1,string_logs=string_logs, args=args)
            trainer = Trainer(model, train_dataset, validation_dataset, tconf)

            output_mode = args.output_mode.split("_")

            trainer.dynamic_stream()

        
        if args.predict:

                class_info = fold_split(args)
                block_size = args.block_size # spatial extent of the model for its context

                training_dataset = get_dataloader(args=args,train_val='training',load=True)
                validation_dataset = get_dataloader(args=args,train_val='validation',load=True)
                test_dataset = get_dataloader(args=args,train_val='testing',load=True)

                num_class=args.n_class

                mconf = GPTConfig(vocab_size=validation_dataset.vocab_size, block_size=block_size,num_class=num_class,
                n_layer=args.n_layer,n_head=args.n_head, n_embd=args.n_emb,context_length=args.context_length,conv_filter=256)

                if args.addposition:
                        mconf = GPTConfig(vocab_size=validation_dataset.vocab_size, block_size=block_size,num_class=num_class,
                        n_layer=args.n_layer,n_head=args.n_head, n_embd=args.n_emb,context_length=args.context_length,conv_filter=256,position_size=validation_dataset.position_size)

                model = get_model(args,mconf)

                string_logs = f"{args.tag}_{args.arch}_bs{mconf.block_size:.0f}_nl{mconf.n_layer:.0f}_nh{mconf.n_head:.0f}_ne{mconf.n_embd:.0f}_cl{mconf.context_length:.0f}/"

                tconf = TrainerConfig(max_epochs=200, batch_size=1, learning_rate=6e-3,
                        lr_decay=True, warmup_tokens=1*200, final_tokens=200*len(validation_dataset)*block_size,
                        num_workers=20,string_logs=string_logs, args=args)

                trainer = Trainer(model, None,[validation_dataset], tconf)

                output_mode = args.output_mode.split("_")

                if len(output_mode)>1:
                    trainer.predict_multi_stream(len(output_mode))
                else:
                    trainer.predict()

        if args.newpredict:

                class_info = fold_split(args)
                block_size = args.block_size # spatial extent of the model for its context

                training_dataset = get_dataloader(args=args,train_val='training',load=True)
                validation_dataset = get_dataloader(args=args,train_val='validation',load=True)
                test_dataset = get_dataloader(args=args,train_val='testing',load=True)

                num_class=args.n_class

                mconf = GPTConfig(vocab_size=validation_dataset.vocab_size, block_size=block_size,num_class=num_class,
                n_layer=args.n_layer,n_head=args.n_head, n_embd=args.n_emb,context_length=args.context_length,conv_filter=256,position_size=validation_dataset.position_size, rt_size = validation_dataset.rt_size)

                model = get_model(args,mconf)

                string_logs = f"{args.tag}_{args.arch}_bs{mconf.block_size:.0f}_nl{mconf.n_layer:.0f}_nh{mconf.n_head:.0f}_ne{mconf.n_embd:.0f}_cl{mconf.context_length:.0f}/"

                tconf = TrainerConfig(max_epochs=200, batch_size=1, learning_rate=6e-3,
                        lr_decay=True, warmup_tokens=1*200, final_tokens=200*len(validation_dataset)*block_size,
                        num_workers=20,string_logs=string_logs, args=args)

                trainer = Trainer(model, None,[validation_dataset], tconf)

                if args.visval:
                    trainer.vis_embed()

                if args.crossdata:
                    trainer.newpredict_dynamic_streamc(args.predictvis)
                else:
                    trainer.newpredict_dynamic_stream(args.predictvis)

        if args.finalpredict:

                class_info = fold_split(args)
                block_size = args.block_size # spatial extent of the model for its context

                validation_dataset = get_dataloader(args=args,train_val='validation',load=True)
                train_dataset = get_dataloader(args=args,train_val='training',load=True)

                num_class=args.n_class
           
                mconf = GPTConfig(vocab_size=validation_dataset.vocab_size, block_size=block_size,num_class=args.n_class, n_layer=args.n_layer,n_head=args.n_head, n_embd=args.n_emb,context_length=args.context_length,conv_filter=256,
                position_size=validation_dataset.position_size, ges_size =  validation_dataset.ges_size,rt_size = validation_dataset.rt_size)

                model = get_model(args,mconf)

                string_logs = f"{args.tag}_{args.arch}_bs{mconf.block_size:.0f}_nl{mconf.n_layer:.0f}_nh{mconf.n_head:.0f}_ne{mconf.n_embd:.0f}_cl{mconf.context_length:.0f}/"

                tconf = TrainerConfig(max_epochs=200, batch_size=1, learning_rate=6e-3,
                        lr_decay=True, warmup_tokens=1*200, final_tokens=200*len(validation_dataset)*block_size,
                        num_workers=20,string_logs=string_logs, args=args)

                trainer = Trainer(model, None,[validation_dataset], tconf)

                if args.vis_attention:
                    trainer = Trainer(model, None,[train_dataset, validation_dataset], tconf) 

                    trainer.visualize_attention(args.vis_attention)

                else:                    
                    if args.visval:                   
                        trainer.vis_embed()
                    
                    if args.predictvis:
                        trainer = Trainer(model, None,[train_dataset,validation_dataset], tconf) 

                    trainer.finalpredict_dynamic_stream(args.predictvis,args.adddata_dir)

        if args.finalpredictnewdata:

                class_info = fold_split(args)
                block_size = args.block_size # spatial extent of the model for its context

                validation_dataset = get_dataloader(args=args,train_val='validation',load=True)
           
                mconf = GPTConfig(vocab_size=validation_dataset.vocab_size, block_size=block_size,num_class=args.n_class, n_layer=args.n_layer,n_head=args.n_head, n_embd=args.n_emb,context_length=args.context_length,conv_filter=256,
                position_size=validation_dataset.position_size, ges_size =  validation_dataset.ges_size,rt_size = validation_dataset.rt_size)

                model = get_model(args,mconf)

                string_logs = f"{args.tag}_{args.arch}_bs{mconf.block_size:.0f}_nl{mconf.n_layer:.0f}_nh{mconf.n_head:.0f}_ne{mconf.n_embd:.0f}_cl{mconf.context_length:.0f}/"

                tconf = TrainerConfig(max_epochs=200, batch_size=1, learning_rate=6e-3,
                        lr_decay=True, warmup_tokens=1*200, final_tokens=200*len(validation_dataset)*block_size,
                        num_workers=20,string_logs=string_logs, args=args)

                trainer = Trainer(model, None,[validation_dataset], tconf)

                if args.vis_attention:
                    trainer = Trainer(model, None,[validation_dataset], tconf) 
                    trainer.visualize_attention(args.vis_attention)

                else:                    
                    if args.visval:                   
                        trainer.vis_embed()
                    
                    if args.predictvis:
                        trainer = Trainer(model, None,[validation_dataset], tconf) 

                    trainer.finalpredict_newdata(args.predictvis,args.adddata_dir)

        if args.newpredict2:

                class_info = fold_split(args)
                block_size = args.block_size # spatial extent of the model for its context

                training_dataset = get_dataloader(args=args,train_val='training',load=True)
                validation_dataset = get_dataloader(args=args,train_val='validation',load=True)

                num_class=args.n_class

                mconf = GPTConfig(vocab_size=validation_dataset.vocab_size, block_size=block_size,num_class=num_class,
                n_layer=args.n_layer,n_head=args.n_head, n_embd=args.n_emb,context_length=args.context_length,conv_filter=256,position_size=validation_dataset.position_size, ges_size =  validation_dataset.ges_size, rt_size = validation_dataset.rt_size)

                model = get_model(args,mconf)

                string_logs = f"{args.tag}_{args.arch}_bs{mconf.block_size:.0f}_nl{mconf.n_layer:.0f}_nh{mconf.n_head:.0f}_ne{mconf.n_embd:.0f}_cl{mconf.context_length:.0f}/"

                tconf = TrainerConfig(max_epochs=200, batch_size=1, learning_rate=6e-3,
                        lr_decay=True, warmup_tokens=1*200, final_tokens=200*len(validation_dataset)*block_size,
                        num_workers=20,string_logs=string_logs, args=args)

                trainer = Trainer(model, None,[validation_dataset], tconf)

                if args.visval:
                    trainer.vis_embed2()

                trainer.newpredict_dynamic_stream(args.predictvis)

        if args.single_predict:

                block_size = args.block_size
                num_class=args.n_class

                validation_dataset = get_dataloader(args=args,train_val='validation',load=True)

                test_dataset = SinglePrediction(data_dir = args.data_dir)

                mconf = GPTConfig(vocab_size=validation_dataset.vocab_size, block_size=block_size,num_class=num_class,
                n_layer=args.n_layer,n_head=args.n_head, n_embd=args.n_emb,context_length=args.context_length,conv_filter=256)

                model = get_model(args,mconf)
                string_logs = f"{args.tag}_{args.arch}_bs{mconf.block_size:.0f}_nl{mconf.n_layer:.0f}_nh{mconf.n_head:.0f}_ne{mconf.n_embd:.0f}_cl{mconf.context_length:.0f}/"

                tconf = TrainerConfig(max_epochs=200, batch_size=1, learning_rate=6e-4,
                        lr_decay=True, warmup_tokens=1*200, final_tokens=200*len(validation_dataset)*block_size,
                        num_workers=20,string_logs=string_logs, args=args)

                trainer = Trainer(model, None,[test_dataset], tconf)

                trainer.single_predict()

        if args.vis_weight:
            class_info = fold_split(args)
            block_size = args.block_size # spatial extent of the model for its context

            training_dataset = get_dataloader(args=args,train_val='training',load=True)
            validation_dataset = get_dataloader(args=args,train_val='validation',load=True)
            test_dataset = get_dataloader(args=args,train_val='testing',load=True)

            num_class=args.n_class

            mconf = GPTConfig(vocab_size=validation_dataset.vocab_size, block_size=block_size,num_class=num_class,
            n_layer=args.n_layer,n_head=args.n_head, n_embd=args.n_emb,context_length=args.context_length,conv_filter=256)

            if args.addposition:
                    mconf = GPTConfig(vocab_size=validation_dataset.vocab_size, block_size=block_size,num_class=num_class,
                    n_layer=args.n_layer,n_head=args.n_head, n_embd=args.n_emb,context_length=args.context_length,conv_filter=256,position_size=validation_dataset.position_size)

            model = get_model(args,mconf)

            string_logs = f"{args.tag}_{args.arch}_bs{mconf.block_size:.0f}_nl{mconf.n_layer:.0f}_nh{mconf.n_head:.0f}_ne{mconf.n_embd:.0f}_cl{mconf.context_length:.0f}/"

            tconf = TrainerConfig(max_epochs=200, batch_size=1, learning_rate=6e-3,
                    lr_decay=True, warmup_tokens=1*200, final_tokens=200*len(validation_dataset)*block_size,
                    num_workers=20,string_logs=string_logs, args=args)

            trainer = Trainer(model, None,[validation_dataset,test_dataset], tconf)

            output_mode = args.output_mode.split("_")

            if len(output_mode)>1:
                trainer.predict_vis(len(output_mode))
            else:
                trainer.predict()

        if args.top_weight:
            class_info = fold_split(args)
            block_size = args.block_size # spatial extent of the model for its context

            training_dataset = get_dataloader(args=args,train_val='training',load=True)
            validation_dataset = get_dataloader(args=args,train_val='validation',load=True)
            test_dataset = get_dataloader(args=args,train_val='testing',load=True)

            num_class=args.n_class

            mconf = GPTConfig(vocab_size=validation_dataset.vocab_size, block_size=block_size,num_class=num_class,
            n_layer=args.n_layer,n_head=args.n_head, n_embd=args.n_emb,context_length=args.context_length,conv_filter=256)

            if args.addposition:
                    mconf = GPTConfig(vocab_size=validation_dataset.vocab_size, block_size=block_size,num_class=num_class,
                    n_layer=args.n_layer,n_head=args.n_head, n_embd=args.n_emb,context_length=args.context_length,conv_filter=256,position_size=validation_dataset.position_size)

            model = get_model(args,mconf)

            string_logs = f"{args.tag}_{args.arch}_bs{mconf.block_size:.0f}_nl{mconf.n_layer:.0f}_nh{mconf.n_head:.0f}_ne{mconf.n_embd:.0f}_cl{mconf.context_length:.0f}/"

            tconf = TrainerConfig(max_epochs=200, batch_size=1, learning_rate=6e-3,
                    lr_decay=True, warmup_tokens=1*200, final_tokens=200*len(validation_dataset)*block_size,
                    num_workers=20,string_logs=string_logs, args=args)

            trainer = Trainer(model, None,[training_dataset,validation_dataset,test_dataset], tconf)

            output_mode = args.output_mode.split("_")

            if len(output_mode)>1:
                trainer.topweight_vis(len(output_mode))
            else:
                trainer.predict()

        if args.single_pred_vcf:

            args = translate_args(args)

            #cmd_preprocess(args)
            preprocessing_fromdmm(args)

            pdb.set_trace()

            class_info = fold_split(args)
            block_size = args.block_size # spatial extent of the model for its context

            training_dataset = get_dataloader(args=args,train_val='training',load=True)
            validation_dataset = get_dataloader(args=args,train_val='validation',load=True)
            test_dataset = get_dataloader(args=args,train_val='testing',load=True)

            num_class=args.n_class

            mconf = GPTConfig(vocab_size=validation_dataset.vocab_size, block_size=block_size,num_class=num_class,
            n_layer=args.n_layer,n_head=args.n_head, n_embd=args.n_emb,context_length=args.context_length,conv_filter=256)

            if args.addposition:
                    mconf = GPTConfig(vocab_size=validation_dataset.vocab_size, block_size=block_size,num_class=num_class,
                    n_layer=args.n_layer,n_head=args.n_head, n_embd=args.n_emb,context_length=args.context_length,conv_filter=256,position_size=validation_dataset.position_size)

            model = get_model(args,mconf)

            string_logs = f"{args.tag}_{args.arch}_bs{mconf.block_size:.0f}_nl{mconf.n_layer:.0f}_nh{mconf.n_head:.0f}_ne{mconf.n_embd:.0f}_cl{mconf.context_length:.0f}/"

            tconf = TrainerConfig(max_epochs=200, batch_size=1, learning_rate=6e-3,
                    lr_decay=True, warmup_tokens=1*200, final_tokens=200*len(validation_dataset)*block_size,
                    num_workers=20,string_logs=string_logs, args=args)

            trainer = Trainer(model, None,[training_dataset,validation_dataset,test_dataset], tconf)

            output_mode = args.output_mode.split("_")

            if len(output_mode)>1:
                trainer.topweight_vis(len(output_mode))
            else:
                trainer.predict()





        