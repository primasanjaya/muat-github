# make deterministic
from models.utils import set_seed
set_seed(42)
#frompc

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset

from models.model import *

from models.trainer import *
from models.predict import *

from models.utils import sample

import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)
import pdb

from dataset.pcawgtcga_dataloader import TCGAPCAWG_Dataloader
from dataset.singlepredictvcf import SinglePredictVCF
from dataset.predictfolder_dataset import *

from preprocessing.dmm.dmm import *
from preprocessing.fromvcffiles import *
from preprocessing.dmm.preprocess3 import *
from preprocessing.dmm.annotate_mutations_all import *

from models.utils import *

import argparse
import os
import pandas as pd

import subprocess

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
        parser.add_argument('--gx-dir', type=str, default=None,
                        help='input gene expression data')
        parser.add_argument('--predict-filepath', type=str, default=None,
                        help='all samples paths that will be predicted')
        parser.add_argument('--predict-inputlist', type=list)

            #OUTPUT DATA
        parser.add_argument('--output-train-dir', type=str, default=None,
                        help='output data directory')
        parser.add_argument('--output-crossdata-dir', type=str, default=None,
                        help='output cross data directory')
        parser.add_argument('--output-newdata-dir', type=str, default=None,
                        help='output new data directory')
        parser.add_argument('--tmp-dir', type=str, default=None,
                        help='temporary data directory')
        parser.add_argument('--output-prefix', type=str, default=None,
                        help='prefix of output data')
        parser.add_argument('--output-pred-dir', type=str, default=None,
                        help='output of prediction directory')
        parser.add_argument('--output-pred-file', type=str, default=None,
                        help='output prediction file')
        parser.add_argument('--output-pred-filename', type=str, default=None,
                        help='output prediction filename')

            # FILENAMES
        parser.add_argument('--input-filename', type=str, default=None,
                        help='input filename')
        parser.add_argument('--input-file', type=str, default=None,
                        help='input file')
        parser.add_argument('--output-filename', type=str, default=None,
                        help='output filename')
        parser.add_argument('--output-file', type=str, default=None,
                        help='output file')
        parser.add_argument('--trainsplit-file', type=str, default=None,
                        help='train and validation file split')
        parser.add_argument('--valsplit-file', type=str, default=None,
                        help='train and validation file split')
        parser.add_argument('--classinfo-file', type=str, default=None,
                        help='class info file')

            #CKPT SAVE
        parser.add_argument('--save-ckpt-dir', type=str, default=None,
                        help='save checkpoint dir')
        parser.add_argument('--save-ckpt-filename', type=str, default=None,
                        help='save checkpoint filename')
            #CKPT LOAD
        parser.add_argument('--load-ckpt-dir', type=str, default=None,
                        help='load checkpoint dir')
        parser.add_argument('--load-ckpt-filename', type=str, default=None,
                        help='load checkpoint filename')
        parser.add_argument('--load-ckpt-file', type=str, default=None,
                        help='load checkpoint complete path file')
        # HYPER PARAMS 
        parser.add_argument('--epoch', type=int, default=1,
                        help='number of epoch')
        parser.add_argument('--l-rate', type=float, default=6e-4,
                        help='learning rate')
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
        parser.add_argument('--epi-emb', type=int, default=2, 
                            help='epigenetic embedding')


        #EXECUTION
        
        parser.add_argument('--train', action='store_true', default=False,
                            help='execute training')
        parser.add_argument('--generative', action='store_true', default=False,
                            help='execute generative training (dimensional reduction)')
        
        parser.add_argument('--predict', action='store_true', default=False,
                            help='execute prediction')

        parser.add_argument('--predict-new-data', action='store_true', default=False,
                            help='execute prediction from new data (PCAWG training-ready format)')
        
        parser.add_argument('--single-pred-vcf', action='store_true', default=False)
        parser.add_argument('--multi-pred-vcf', action='store_true', default=False)


        parser.add_argument('--get-motif', action='store_true', default=False)
        parser.add_argument('--get-position', action='store_true', default=False)
        parser.add_argument('--get-ges', action='store_true', default=False)
        parser.add_argument('--get-epi', action='store_true', default=False)

        parser.add_argument('--motif', action='store_true', default=False)
        parser.add_argument('--motif-pos', action='store_true', default=False)
        parser.add_argument('--motif-pos-ges', action='store_true', default=False)
        parser.add_argument('--motif-pos-ges-epi', action='store_true', default=False)

        parser.add_argument('--ensemble', action='store_true', default=False)
        parser.add_argument('--predict-all', action='store_true', default=False)

        parser.add_argument('--get-features', action='store_true', default=False)

        parser.add_argument('--num-mut', type=int, default=0,
                        help='sampling number of mutation')
        parser.add_argument('--frac', type=float, default=0,
                        help='sampling number of mutation based on data fraction')

        parser.add_argument('--mut-type', type=str, default='',
                        help='mutation type, only [SNV,SNV+MNV,SNV+MNV+indel,SNV+MNV+indel+SV/MEI,SNV+MNV+indel+SV/MEI+Neg] can be applied')
    
        parser.add_argument('--mutratio', type=str, default=None,
                        help='mutation ratio per mutation type, sum of them must be one')
        parser.add_argument('--vis-attention', type=str, default='',
                        help='visualize attention values')

        parser.add_argument('--genomic-tracks', type=str, default=None, 
                            help='Genomic tracks directory')

        parser.add_argument('--convert-hg38-hg19',action='store_true', default=False)

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


def execute_annotation(args,only_input_filename):

    #gc content
    syntax_gc = 'python3 preprocessing/dmm/annotate_mutations_with_gc_content.py \
    -i ' + args.tmp_dir + only_input_filename + '.tsv.gz \
    -o ' + args.tmp_dir + only_input_filename + '.gc.tsv.gz \
    -n 1001 \
    -l gc1kb \
    --reference ' + args.reference + ' \
    --verbose'
    subprocess.run(syntax_gc, shell=True)
    os.remove(args.tmp_dir + only_input_filename + '.tsv.gz')
    
    #pdb.set_trace()

    if args.convert_hg38_hg19:
        from pyliftover import LiftOver

        #lo = LiftOver('/mnt/g/experiment/redo_muat/muat-github/preprocessing/genomic_tracks/hg38ToHg19.over.chain.gz')
        #lo = LiftOver('/genomic_tracks/GRCh37_to_GRCh38.chain.gz')
        lo = LiftOver('hg38', 'hg19')
        
        pd_hg38 = pd.read_csv(args.tmp_dir + only_input_filename + '.gc.tsv.gz',sep='\t') 
        chrom_pos = []

        for i in range(len(pd_hg38)):
            try:
                row = pd_hg38.iloc[i]
                chrom = str('chr') + str(row['chrom'])
                pos = row['pos']
                ref = row['ref']
                alt = row['alt']
                sample = row['sample']
                seq = row['seq']
                gc1kb = row['gc1kb']
                hg19chrompos = lo.convert_coordinate(chrom, pos)
                chrom = hg19chrompos[0][0][3:]
                pos = hg19chrompos[0][1]

                chrom_pos.append((chrom,pos,ref,alt,sample,seq,gc1kb))
            except:
                pass
        pd_hg19 = pd.DataFrame(chrom_pos)
        pd_hg19.columns = pd_hg38.columns.tolist()

        pd_hg38.to_csv(args.tmp_dir + only_input_filename + '.gc.tsv.gz',sep='\t',index=False, compression="gzip")

    # Genic regions
    syntax_genic = 'preprocessing/dmm/annotate_mutations_with_bed.sh \
    ' + args.tmp_dir + only_input_filename + '.gc.tsv.gz \
    ' + args.genomic_tracks + 'Homo_sapiens.GRCh37.87.genic.genomic.bed.gz \
    '+ args.tmp_dir + only_input_filename + '.gc.genic.tsv.gz \
    genic'
    subprocess.run(syntax_genic, shell=True)
    os.remove(args.tmp_dir + only_input_filename + '.gc.tsv.gz')

    #exon regions
    syntax_exonic = 'preprocessing/dmm/annotate_mutations_with_bed.sh \
    ' + args.tmp_dir + only_input_filename + '.gc.genic.tsv.gz \
    ' + args.genomic_tracks + 'Homo_sapiens.GRCh37.87.exons.genomic.bed.gz \
    ' + args.tmp_dir + only_input_filename + '.gc.genic.exonic.tsv.gz \
    exonic'
    subprocess.run(syntax_exonic, shell=True)
    #pdb.set_trace()
    os.remove(args.tmp_dir + only_input_filename + '.gc.genic.tsv.gz')

    # Annotate dataset with gene orientation information

    syntax_geneorientation = 'python3 preprocessing/dmm/annotate_mutations_with_coding_strand.py \
    -i '+ args.tmp_dir + only_input_filename + '.gc.genic.exonic.tsv.gz \
    -o '+ args.tmp_dir + only_input_filename + '.gc.genic.exonic.cs.tsv.gz \
    --annotation ' + args.genomic_tracks + 'Homo_sapiens.GRCh37.87.transcript_directionality.bed.gz \
    --ref ' + args.reference

    #pdb.set_trace()
    subprocess.run(syntax_geneorientation, shell=True)
    os.remove(args.tmp_dir + only_input_filename + '.gc.genic.exonic.tsv.gz')
    

if __name__ == '__main__':
        best_accuracy=0

        args = get_args()
        args = fix_path(args)

        #simplified args
        args = simplified_args(args)

        if args.train:
            train_dataset = get_simplified_dataloader(args=args,train_val='training',input_filename=None)
            validation_dataset = get_simplified_dataloader(args=args,train_val='validation',input_filename=None)

            mconf = ModelConfig(vocab_size=train_dataset.vocab_size, block_size=args.block_size,num_class=args.n_class,
                    n_layer=args.n_layer,n_head=args.n_head, n_embd=args.n_emb,position_size=train_dataset.position_size, ges_size =  train_dataset.ges_size,dnn_input=train_dataset.dnn_input, epi_size=train_dataset.epi_size, epi_emb = args.epi_emb)

            model = get_model(args,mconf)

            tconf = TrainerConfig(max_epochs=args.epoch, batch_size=args.batch_size, learning_rate=args.l_rate,
                    lr_decay=True, num_workers=1, args=args)
            trainer = Trainer(model, train_dataset, validation_dataset, tconf)

            #trainer.dynamic_stream()
            trainer.batch_train()


        if args.predict:

            device = 'cpu'
            if torch.cuda.is_available():
                device = torch.cuda.current_device()

            #load ckpt
            if device == 'cpu':
                allckpt = torch.load(args.load_ckpt_dir + args.load_ckpt_filename,map_location=device)
            else:
                allckpt = torch.load(args.load_ckpt_dir + args.load_ckpt_filename)

            if len(allckpt) == 2:
                old_args = allckpt[1]
                weight = allckpt[0]

                args = update_args(args,old_args)

            else:
                weight = allckpt

            validation_dataset = get_simplified_dataloader(args=args,train_val='validation')
            train_dataset = get_simplified_dataloader(args=args,train_val='training')

            try:
                mconf = ModelConfig(vocab_size=validation_dataset.vocab_size, block_size=args.block_size,num_class=args.n_class, n_layer=args.n_layer,n_head=args.n_head, n_embd=args.n_emb,position_size=validation_dataset.position_size, ges_size =  validation_dataset.ges_size)
                model = get_model(args,mconf)

                #load weight to the model
                model = model.to(device)
                model.load_state_dict(weight)
            except:
                #solving
                args = solving_arch(args)
                mconf = ModelConfig(vocab_size=validation_dataset.vocab_size, block_size=args.block_size,num_class=args.n_class, n_layer=args.n_layer,n_head=args.n_head, n_embd=args.n_emb,
                position_size=validation_dataset.position_size, ges_size =  validation_dataset.ges_size)
                model = get_model(args,mconf)

                #load weight to the model
                model = model.to(device)
                model.load_state_dict(weight)

            tconf = TrainerConfig(max_epochs=1, batch_size=1, learning_rate=6e-3,
                    lr_decay=True,num_workers=20, args=args)

            trainer = Trainer(model, None,[validation_dataset], tconf)

            if args.vis_attention:
                trainer = Trainer(model, None,[train_dataset, validation_dataset], tconf) 
                trainer.visualize_attention(args.vis_attention)
            else:                    
                if args.get_features:
                    trainer = Trainer(model, None,[train_dataset,validation_dataset], tconf) 

                trainer.predict(args.get_features,args.input_data_dir)

        if args.single_pred_vcf:

            args = translate_args(args)
            cmd_preprocess(args)

            only_input_filename = args.input_filename[:-4]
            execute_annotation(args,only_input_filename)
            preprocessing_fromdmm(args)

            device = 'cpu'
            if torch.cuda.is_available():
                device = torch.cuda.current_device()

            #load ckpt
            if device == 'cpu':
                allckpt = torch.load(args.load_ckpt_dir + args.load_ckpt_filename,map_location=device)
            else:
                allckpt = torch.load(args.load_ckpt_dir + args.load_ckpt_filename)

            #check weight
            #pdb.set_trace()
            if len(allckpt) == 3: #newformat
                old_args = allckpt[1]
                weight = allckpt[0]
                update_args(args,old_args)
            else:
                print('Warning: this model is depricated')

            validation_dataset = get_simplified_dataloader(args=args,train_val='validation',input_filename=args.input_filename)

            mconf = ModelConfig(vocab_size=validation_dataset.vocab_size, 
                                block_size=args.block_size,
                                num_class=args.n_class,
                                n_layer=args.n_layer,
                                n_head=args.n_head, 
                                n_embd=args.n_emb,
                                position_size=validation_dataset.position_size,
                                ges_size = validation_dataset.ges_size,
                                context_length=args.context_length,
                                args=args)

            #pdb.set_trace()

            model = get_model(args,mconf)

            #load weight to the model
            model = model.to(device)
            model.load_state_dict(weight)

            tconf = PredictorConfig(max_epochs=1, batch_size=1,num_workers=20, args=args)

            predictor = Predictor(model, None,[validation_dataset], tconf)

            predictor.predict(args.get_features,args.input_newdata_dir)


        if args.ensemble:
            if args.predict_all:
                args.ensemble = True
                args = translate_args(args)
                func_annotate_mutation_all(args)
                preprocessing_fromdmm_all(args)

                device = 'cpu'
                if torch.cuda.is_available():
                    device = torch.cuda.current_device()

                all_folder = os.listdir(args.load_ckpt_dir)
                
                #args.single_pred_vcf = True #carefull this is used in args.single pred. will be called twice if this is put above single_pred vcf
                
                for i in range(len(all_folder)):

                    try:
                        folder1 = all_folder[i]
                        splitfold = folder1.split('fold')
                        #pdb.set_trace()
                        splitfold = splitfold[1].split('_')
                        fold = int(splitfold[0])
                        args.output_prefix = 'model' + str(fold)
                        #load ckpt
                        if device == 'cpu':
                            allckpt = torch.load(args.load_ckpt_dir + str(folder1) + '/' + args.load_ckpt_filename,map_location=device)
                        else:
                            allckpt = torch.load(args.load_ckpt_dir + str(folder1) + '/' + args.load_ckpt_filename)
                    except:
                        print('can not load ckpt, plesae check the ckpt directory')

                    #check weight
                    #pdb.set_trace()
                    if len(allckpt) == 3: #newformat
                        old_args = allckpt[1]
                        weight = allckpt[0]
                        update_args(args,old_args)
                    else:
                        print('Warning: this model is depricated')

                    validation_dataset = get_simplified_dataloader(args=args,train_val='validation',input_filename=args.input_filename)

                    mconf = ModelConfig(vocab_size=validation_dataset.vocab_size, 
                                        block_size=args.block_size,
                                        num_class=args.n_class,
                                        n_layer=args.n_layer,
                                        n_head=args.n_head, 
                                        n_embd=args.n_emb,
                                        position_size=validation_dataset.position_size,
                                        ges_size = validation_dataset.ges_size,
                                        context_length=args.context_length,
                                        args=args)

                    #pdb.set_trace()

                    model = get_model(args,mconf)

                    #load weight to the model
                    model = model.to(device)
                    model.load_state_dict(weight)

                    tconf = PredictorConfig(max_epochs=1, batch_size=1,num_workers=20, args=args)

                    predictor = Predictor(model, None,[validation_dataset], tconf)

                    predictor.predict(args.get_features,args.input_newdata_dir)
            else:
                args = translate_args(args)
                cmd_preprocess(args)
                #pdb.set_trace()

                if args.predict_filepath is not None:
                    translate_args(args)
                    execute_annotation_all()
                else:
                    only_input_filename = args.input_filename[:-4]
                    execute_annotation(args,only_input_filename)
                    preprocessing_fromdmm(args)

                    device = 'cpu'
                    if torch.cuda.is_available():
                        device = torch.cuda.current_device()

                    all_folder = os.listdir(args.load_ckpt_dir)
                    
                    args.single_pred_vcf = True #carefull this is used in args.single pred. will be called twice if this is put above single_pred vcf
                    
                    for i in range(len(all_folder)):

                        try:
                            folder1 = all_folder[i]
                            splitfold = folder1.split('fold')
                            #pdb.set_trace()
                            splitfold = splitfold[1].split('_')
                            fold = int(splitfold[0])
                            args.output_prefix = 'model' + str(fold)
                            #load ckpt
                            if device == 'cpu':
                                allckpt = torch.load(args.load_ckpt_dir + str(folder1) + '/' + args.load_ckpt_filename,map_location=device)
                            else:
                                allckpt = torch.load(args.load_ckpt_dir + str(folder1) + '/' + args.load_ckpt_filename)
                        except:
                            print('can not load ckpt, plesae check the ckpt directory')

                        #check weight
                        #pdb.set_trace()
                        if len(allckpt) == 3: #newformat
                            old_args = allckpt[1]
                            weight = allckpt[0]
                            update_args(args,old_args)
                        else:
                            print('Warning: this model is depricated')

                        validation_dataset = get_simplified_dataloader(args=args,train_val='validation',input_filename=args.input_filename)

                        mconf = ModelConfig(vocab_size=validation_dataset.vocab_size, 
                                            block_size=args.block_size,
                                            num_class=args.n_class,
                                            n_layer=args.n_layer,
                                            n_head=args.n_head, 
                                            n_embd=args.n_emb,
                                            position_size=validation_dataset.position_size,
                                            ges_size = validation_dataset.ges_size,
                                            context_length=args.context_length,
                                            args=args)

                        #pdb.set_trace()

                        model = get_model(args,mconf)

                        #load weight to the model
                        model = model.to(device)
                        model.load_state_dict(weight)

                        tconf = PredictorConfig(max_epochs=1, batch_size=1,num_workers=20, args=args)

                        predictor = Predictor(model, None,[validation_dataset], tconf)

                        predictor.predict(args.get_features,args.input_newdata_dir)


        

        if args.multi_pred_vcf:

            #get all vcf files
            
            vcffiles = os.listdir(args.input_data_dir)
            vcffiles = [i for i in vcffiles if i[-4:] =='.vcf']

            for i in vcffiles:
                args.input_filename = i
                args = translate_args(args)
                cmd_preprocess(args)

                only_input_filename = i[:-4]
                execute_annotation(args,only_input_filename)
                preprocessing_fromdmm(args)
            
            #pdb.set_trace()

            device = 'cpu'
            if torch.cuda.is_available():
                device = torch.cuda.current_device()

            #load ckpt
            if device == 'cpu':
                allckpt = torch.load(args.load_ckpt_dir + args.load_ckpt_filename,map_location=device)
            else:
                allckpt = torch.load(args.load_ckpt_dir + args.load_ckpt_filename)

            #check weight
            if len(allckpt) == 3: #newformat
                old_args = allckpt[1]
                weight = allckpt[0]
                update_args(args,old_args)
            else:
                print('Warning: this model is depricated')

            validation_dataset = get_simplified_dataloader(args=args,train_val='validation',input_filename=vcffiles)

            mconf = ModelConfig(vocab_size=validation_dataset.vocab_size, 
                                block_size=args.block_size,
                                num_class=args.n_class,
                                n_layer=args.n_layer,
                                n_head=args.n_head, 
                                n_embd=args.n_emb,
                                position_size=validation_dataset.position_size,
                                ges_size = validation_dataset.ges_size,
                                context_length=args.context_length,
                                args=args)

            #pdb.set_trace()

            model = get_model(args,mconf)

            #load weight to the model
            model = model.to(device)
            model.load_state_dict(weight)

            tconf = PredictorConfig(max_epochs=1, batch_size=1,num_workers=20, args=args)

            predictor = Predictor(model, None,[validation_dataset], tconf)

            predictor.predict(args.get_features,args.input_newdata_dir)
        


        





        