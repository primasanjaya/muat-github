"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

#from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)
import pdb
import os
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report, precision_recall_fscore_support
#from tensorboardX import SummaryWriter
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.nn import functional as F
from collections import Counter 
from torch.autograd import Variable
from tempfile import TemporaryFile


class PredictorConfig:

    def __init__(self, 
                    max_epochs=1,
                    batch_size=1,
                    learning_rate = 3e-4,
                    betas = (0.9, 0.95),
                    grad_norm_clip = 1.0,
                    weight_decay = 0.001,
                    lr_decay = False,
                    ckpt_path = None,
                    string_logs = None,
                    num_workers = 0,
                    ckpt_name = 'model',
                    args=None, 
                    **kwargs):

        self.max_epochs=max_epochs
        self.batch_size=batch_size
        self.learning_rate = learning_rate
        self.betas = betas
        self.grad_norm_clip = grad_norm_clip
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.ckpt_path = ckpt_path
        self.string_logs = string_logs
        self.num_workers = num_workers # for DataLoader
        self.ckpt_name = ckpt_name
        self.args=args

        self.ckpt_path = args.load_ckpt_dir

        for k,v in kwargs.items():
            setattr(self, k, v)

class Predictor:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.test_dataset = test_dataset
        self.config = config
        self.global_acc = 0

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()

    def target_handler(self,target_name, args):
        print('todo')

    def predict(self,predictvis,adddata_dir):
        model, config = self.model, self.config
        is_train = False

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('number of parameters = ' + f"{params:,}")

        pwd = os.getcwd()

        allloader = self.test_dataset
        self.class_handler = allloader[0].pd_class_info

        #predict all
        for loader in allloader:

            #pdb.set_trace()
            for it in range(0,len(loader)):
                try:
                    loader.block_size = self.config.args.block_size

                    npx,npy = loader.__getitem__(it)

                    #print(str(it + 1) + ' ' + str(npx[0]) + str(npy))

                    # place data on the correct device
                    x_all = []
                    for i in range(len(npx[1])):
                        xi = npx[1][i].to(self.device)
                        xi = xi.unsqueeze(0)
                        x_all.append(xi.to(self.device))
                        
                    if npy == '':
                        y = None
                    else:
                        y = npy.to(self.device)
                        y = y.unsqueeze(0)

                    with torch.set_grad_enabled(is_train):

                        try:
                            logits, loss = model(x_all)
                        except:
                            pdb.set_trace()

                        _, predicted = torch.max(logits.data, 1)
                        predicted = predicted.detach().cpu().numpy()[0]

                        cpu_prob = logits.detach().cpu().numpy()[0]

                        pd_cpu_prob = pd.DataFrame(cpu_prob).T

                        pd_cpu_prob.columns = self.class_handler['class_name'].to_list()
                        
                        class_name = loader.pd_class_info.loc[loader.pd_class_info['class_index']==predicted]['class_name'].values[0]
                        pd_cpu_prob['prediction'] = class_name
                        print(npx[0][:-4] + '.vcf is predicted as ' + str(class_name))

                        if self.config.args.output_prefix == None:
                            prefix = ''
                        else:
                            prefix = self.config.args.output_prefix + '_'
                    
                        try:
                            pd_cpu_prob.to_csv(self.config.args.output_pred_dir + prefix + npx[0][:-4] + '_probability.tsv',sep='\t')
                        except:
                            print('Error: Can not export the results, please specify --output-pred-dir')
                        
                        if predictvis:
                            try:
                                features = model(x_all, y,vis=predictvis)
                                features = features.detach().cpu().numpy()[0]

                                pd_cpu_features = pd.DataFrame(features).T

                                featurecolumn = []
                                count = 0

                                for i in range(len(pd_cpu_features.columns)):
                                    count = count + 1
                                    featurecolumn.append('M' + str(count))

                                pd_cpu_features.columns = featurecolumn

                                try:
                                    pd_cpu_features.to_csv(self.config.args.output_pred_dir + prefix + npx[0][:-4] + '_features.tsv',sep='\t')
                                except:
                                    print('Error: Can not export the results, please specify --output-pred-dir')  
                            except:
                                pass
                except:
                    pass
                
    def predict_core(self,predictvis,adddata_dir):
        model, config = self.model, self.config
        is_train = False

        pwd = os.getcwd()

        allloader = self.test_dataset

        self.class_handler = allloader[0].pd_class_info

        if self.device == 'cpu':
            weight = torch.load( self.config.ckpt_path + self.config.ckpt_name + '.pthx',map_location=self.device)
        else:
            weight = torch.load(self.config.ckpt_path + self.config.ckpt_name + '.pthx')

        model = model.to(self.device)
     
        model.load_state_dict(weight)

        #predict all

        pd_all_prob = pd.DataFrame()

        for loader in allloader:

            #pdb.set_trace()
            for it in range(0,len(loader)):
            #for it in range(0,10):
                loader.block_size = self.config.args.block_size

                npx,npy = loader.__getitem__(it)

                print('Predicting ' + npy + ' ' + npx[0])

                #print(str(it + 1) + ' ' + str(npx[0]) + str(npy))

                # place data on the correct device
                x_all = []
                for i in range(len(npx[1])):
                    xi = npx[1][i].to(self.device)
                    xi = xi.unsqueeze(0)
                    x_all.append(xi.to(self.device))

                with torch.set_grad_enabled(is_train):

                    try:
                        logits, loss = model(x_all)
                    except:
                        pdb.set_trace()

                    cpu_prob = logits.detach().cpu().numpy()[0]

                    pd_cpu_prob = pd.DataFrame(cpu_prob).T
                    pd_cpu_prob['samples'] = npx[0]
                    pd_cpu_prob['target'] = npy

                    pd_all_prob = pd_all_prob.append(pd_cpu_prob)

                    if predictvis:
                        features = model(x_all, y,predictvis)
                        features=features.cpu().numpy()[0]
                        list_feature.append(features)
        pd_all_prob = pd_all_prob.reset_index(drop=True)
        self.output_handler(logits=pd_all_prob)
        
    def output_handler(self,logits):
        #rename columns
        logits.columns = self.class_handler['class_name'].to_list() + logits.columns[-2:].to_list()
        #save prob
        logits.to_csv(self.config.args.output_pred_dir + self.config.args.input_filename[:-4]  + '_probability.tsv',sep='\t')