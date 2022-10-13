import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from models.model import *
from dataset.pcawgtcga_dataloader import *
from dataset.pcawgtcgaepi_dataloader import *

def get_model(args,mconf):
    arch = "{n}".format(n=args.arch)

    if arch == 'MuAtMotif':
        model = MuAtMotif(mconf)
    elif arch == 'EmbFC':
        model = EmbFC(mconf)    
    elif arch == 'MuAtMotifPosition':
        model = MuAtMotifPosition(mconf)    
    elif arch == 'MuAtMotifPositionGES':
        model = MuAtMotifPositionGES(mconf)
    elif arch == 'MuAtMotifPositionGESRT':
        model = MuAtMotifPositionGESRT (mconf)   
    elif arch == 'MuAtMotifPositionF':
        model = MuAtMotifPositionF(mconf)    
    elif arch == 'MuAtMotifPositionGESF':
        model = MuAtMotifPositionGESF(mconf)
    elif arch == 'MuAtMotifF':
        model = MuAtMotifF(mconf)
    elif arch == 'EmbFCPos':
        model = EmbFCPos(mconf)
    elif arch == 'EmbFCPosGES':
        model = EmbFCPosGES(mconf)
    elif arch == 'DNN_GX':
        model = DNN_GX(mconf)
    elif arch == 'MuAtEpiF':
        model = MuAtEpiF(mconf)
    elif arch == 'MuAtMotifPositionGESEpiF':
        model = MuAtMotifPositionGESEpiF(mconf)
    elif arch == 'MuAtMotifPositionGESEpiF_OneEpiEmb':
        model = MuAtMotifPositionGESEpiF_OneEpiEmb(mconf)
    elif arch == 'MuAtMotifPositionGESEpiF_EpiCompress':
        model = MuAtMotifPositionGESEpiF_EpiCompress(mconf)
    elif arch == 'MuAtEpiF_EpiCompress':
        model = MuAtEpiF_EpiCompress(mconf)

    #pdb.set_trace()
    return model

def get_simplified_dataloader(args,train_val,input_filename):
    if args.dataloader == 'pcawg' or args.dataloader == 'wgspcawg':
        if args.multi_pred_vcf:
            dataloader_class = TCGAPCAWG_Dataloader(dataset_name = args.dataloader, 
                                    data_dir=args.tmp_dir,
                                    mode='testing', 
                                    curr_fold=args.fold, 
                                    block_size=args.block_size, 
                                    mutratio = args.mutratio,
                                    addtriplettoken=args.motif,
                                    addpostoken=args.motif_pos,
                                    addgestoken=args.motif_pos_ges,
                                    input_filename = input_filename,
                                    args = args
                                    )

        if args.single_pred_vcf:
            dataloader_class = TCGAPCAWG_Dataloader(dataset_name = args.dataloader, 
                                    data_dir=args.tmp_dir,
                                    mode='testing', 
                                    curr_fold=args.fold, 
                                    block_size=args.block_size, 
                                    mutratio = args.mutratio,
                                    addtriplettoken=args.motif,
                                    addpostoken=args.motif_pos,
                                    addgestoken=args.motif_pos_ges,
                                    input_filename = input_filename,
                                    args = args
                                    )

        if args.ensamble:
            dataloader_class = TCGAPCAWG_Dataloader(dataset_name = args.dataloader, 
                                    data_dir=args.tmp_dir,
                                    mode='testing', 
                                    curr_fold=args.fold, 
                                    block_size=args.block_size, 
                                    mutratio = args.mutratio,
                                    addtriplettoken=args.motif,
                                    addpostoken=args.motif_pos,
                                    addgestoken=args.motif_pos_ges,
                                    input_filename = input_filename,
                                    args = args
                                    )

        if args.train:
            dataloader_class = TCGAPCAWG_Dataloader(dataset_name = args.dataloader, 
                                    data_dir=args.input_data_dir, 
                                    mode='training', 
                                    curr_fold=args.fold, 
                                    block_size=args.block_size,
                                    mutratio = args.mutratio,
                                    addtriplettoken=args.motif,
                                    addpostoken=args.motif_pos,
                                    addgestoken=args.motif_pos_ges,
                                    args = args)
        if args.predict:
            dataloader_class = TCGAPCAWG_Dataloader(dataset_name = args.dataloader, 
                                    data_dir=args.input_data_dir, 
                                    mode='validation', 
                                    curr_fold=args.fold, 
                                    block_size=args.block_size,
                                    mutratio = args.mutratio,
                                    addtriplettoken=args.motif,
                                    addpostoken=args.motif_pos,
                                    addgestoken=args.motif_pos_ges,args = args)


    if args.dataloader == 'wgsgx':
        if args.train:
            #pdb.set_trace()
            dataloader_class = TCGAPCAWG_Dataloader(
                                    dataset_name = args.dataloader, 
                                    data_dir=args.input_data_dir, 
                                    mode='training', 
                                    curr_fold=args.fold, 
                                    block_size=args.block_size,
                                    mutratio = args.mutratio,
                                    addtriplettoken=args.motif,
                                    addpostoken=args.motif_pos,
                                    addgestoken=args.motif_pos_ges,
                                    args = args,
                                    gx_dir = args.gx_dir)
        
        if args.generative:
            #pdb.set_trace()
            dataloader_class = TCGAPCAWG_Dataloader(
                                    dataset_name = args.dataloader, 
                                    data_dir=args.input_data_dir, 
                                    mode='training', 
                                    curr_fold=args.fold, 
                                    block_size=args.block_size,
                                    mutratio = args.mutratio,
                                    addtriplettoken=args.motif,
                                    addpostoken=args.motif_pos,
                                    addgestoken=args.motif_pos_ges,
                                    args = args,
                                    gx_dir = args.gx_dir)

    if args.dataloader == 'tcga-mpg':
        if args.train:
            dataloader_class = TCGAPCAWGEPI_Dataloader(
                                dataset_name = args.dataloader, 
                                data_dir=args.input_data_dir, 
                                mode=train_val, 
                                curr_fold=args.fold, 
                                block_size=args.block_size,
                                mutratio = args.mutratio,
                                getmotif = args.get_motif,
                                getposition = args.get_position,
                                getges = args.get_ges,
                                getepi = args.get_epi,
                                args = args)

    return dataloader_class

def ensure_path(path, terminator="/"):
    if path == None:
        return path

    path = path.replace('\\', terminator)
    if path.endswith(terminator):
        return path
    else:
        path = path + terminator
    return path

def ensure_filepath(path, terminator="/"):
    if path == None:
        return path

    path = path.replace('\\', terminator)
    if path.endswith(terminator):
        return path
    else:
        path = path
    return path

def fix_path(args):
    cwd = os.getcwd()
    args.cwd =  ensure_path(cwd)

    try:
        args.input_data_dir = ensure_path(args.input_data_dir)
    except:
        pass
    try:
        args.output_data_dir = ensure_path(args.output_data_dir)
    except:
        pass  
    try:
        args.save_ckpt_dir = ensure_path(args.save_ckpt_dir)
    except:
        pass  
    try:
        args.load_ckpt_dir = ensure_path(args.load_ckpt_dir)
    except:
        pass 
    try:
        args.tmp_dir = ensure_path(args.tmp_dir)
    except:
        pass  
    try:
        args.output_pred_dir = ensure_path(args.output_pred_dir)
    except:
        pass  
    try:
        args.gx_dir = ensure_path(args.gx_dir)
    except:
        pass  

    return args


def ensure_terminator(path, terminator="/"):
    if path == None:
        return path
    if path.endswith(terminator):
        return path
    return path + terminator


def strip_suffixes(s, suffixes):
    loop = True
    while loop:
        loop = False
        for suf in suffixes:
            if s.endswith(suf):
                s = s[:-len(suf)]
                loop = True
    return s


def dir_filename_seperate(fullfile):

    try:
        sep_fol_file = fullfile.split('/')

        filename = sep_fol_file[-1]

        dirfol = ensure_path('/'.join(sep_fol_file[:-1]))
    except:
        pdb.set_trace()

    return dirfol,filename

def translate_args(args):

    cwd = ensure_path(os.getcwd())
    args.cwd = cwd

    args.mutation_coding = cwd + 'extfile/mutation_codes_sv.tsv'

    input_dir, inputfile = dir_filename_seperate(args.input_file)
    args.input_data_dir = input_dir
    args.input_filename = inputfile
    args.input = args.input_data_dir + args.input_filename

    '''
    output_dir, outputfile = dir_filename_seperate(args.output_pred_file)
    args.output_pred_dir = output_dir
    args.output_pred_filename = outputfile
    '''

    if args.ensamble:
        args.load_ckpt_filename = 'new_weight.pthx'
    else:
        ckpt_dir, ckpt_file = dir_filename_seperate(args.load_ckpt_file)
        args.load_ckpt_dir = ckpt_dir
        args.load_ckpt_filename = ckpt_file

    args.tmp_dir = cwd + 'data/raw/temp/'

    filename = strip_suffixes(args.input_filename, ['.vcf'])

    args.output = args.tmp_dir + filename + '.tsv.gz'

    args.reference = ensure_filepath(args.reference)
    args.context = 8
    args.sample_id = 'submitted_sample_id'

    args.verbose = 1
    args.generate_negatives = 1
    args.report_interval = 100000
    args.tmp = args.tmp_dir

    args.genomic_tracks = args.cwd + 'preprocessing/genomic_tracks/h37/'

    return args

def simplified_args(args):

    if args.motif:
        args.arch = 'MuAtMotifF'
    if args.motif_pos:
        args.arch = 'MuAtMotifPositionF'
    if args.motif_pos_ges:
        args.arch = 'MuAtMotifPositionGESF'
    if args.mut_type == 'SNV':
        args.mutratio = '1-0-0-0-0'
    elif args.mut_type == 'SNV+MNV':
        args.mutratio = '0.5-0.5-0-0-0'
    elif args.mut_type == 'SNV+MNV+indel':
        args.mutratio = '0.4-0.3-0.3-0-0'
    elif args.mut_type == 'SNV+MNV+indel+SV/MEI':
        args.mutratio = '0.3-0.3-0.2-0.2-0'
    elif args.mut_type == 'SNV+MNV+indel+SV/MEI+Neg':
        args.mutratio = '0.25-0.25-0.25-0.15-0.1'
    else:
        print('None of the option : SNV+MNV+indel+SV/MEI+Neg')

    return args

def solving_arch(args):
    if args.arch == 'MuAtPlainF':
        args.arch = 'MuAtPlain'
    if args.arch == 'TripletPositionF':
        args.arch = 'TripletPosition'
    if args.arch == 'TripletPositionGESF':
        args.arch = 'TripletPositionGES'
    return args

def update_args(args,old_args):

    if old_args.arch == 'CTransformer':
        rename_arch = 'MuAtMotif'
    elif old_args.arch == 'CTransformerF':
        rename_arch = 'MuAtMotifF'
    elif old_args.arch == 'TripletPosition':
        rename_arch = 'MuAtMotifPosition'
    elif old_args.arch == 'TripletPositionF':
        rename_arch = 'MuAtMotifPositionF'
    elif old_args.arch == 'TripletPositionGES':
        rename_arch = 'MuAtMotifPositionGES'
    elif old_args.arch == 'TripletPositionGESF':
        rename_arch = 'MuAtMotifPositionGESF'
    else:
        rename_arch = old_args.arch

    #check if mut type is same or not

    if args.mut_type == old_args.mut_type:
        args.mut_type = args.mut_type
    else:
        print('Warning: ckpt data type is ' + old_args.mut_type + ', different from input data type : ' + args.mut_type)
        print(old_args.mut_type + ' is selected')
        args.mut_type = old_args.mut_type

    args.arch = rename_arch
    args.block_size = old_args.block_size
    #args.dataloader = old_args.dataloader
    #args.fold = old_args.fold
    
    args.motif = old_args.motif
    args.motif_pos = old_args.motif_pos
    args.motif_pos_ges = old_args.motif_pos_ges
    args.mutratio = old_args.mutratio
    args.n_class =  old_args.n_class
    args.n_emb = old_args.n_emb
    args.n_head = old_args.n_head
    args.n_layer =  old_args.n_layer

    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x
