"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)
import pdb
import os
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report, precision_recall_fscore_support
from tensorboardX import SummaryWriter
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.nn import functional as F
from collections import Counter 
from torch.autograd import Variable
from tempfile import TemporaryFile

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.001 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False

    # checkpoint settings
    ckpt_path = None
    string_logs = None
    num_workers = 0 # for DataLoader
    ckpt_name = 'model'
    args = None

    if ckpt_path is not None:
        os.makedirs(ckpt_path, exist_ok=True) 

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.global_acc = 0
        self.pd_logits = []

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()      

    def vis_plot(self,epoch):
        y_true = np.asarray(self.all_target)
        y_pred = np.asarray(self.all_predicted)
        
        acc = accuracy_score(y_true,y_pred)

        try:
            if self.config.args.balance:
                label = sorted(set(self.train_dataset.pd_class_info.class_name))
            else:
                if self.config.args.dataset == 'finalpcawg' or self.config.args.dataset == 'wgspcawg':
                    label = sorted(set(self.train_dataset.pd_class_info.class_name))
                elif self.config.args.dataset == 'finaltcga' or self.config.args.dataset == 'westcga':
                    label = sorted(set(self.train_dataset.pd_class_info.class_name))
        except:
            if self.train_dataset is None:
                label = sorted(set(self.test_dataset[0].pd_class_info.class_name))
            else:
                label = sorted(set(self.train_dataset.pd_class_info.class_name))

        #pdb.set_trace()

        conf_mat = confusion_matrix(y_true, y_pred)
        conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

        df_cm = pd.DataFrame(conf_mat, index = label,columns = label)

        try:
            df_cm.to_csv(self.config.ckpt_path + 'conf_mat.csv')
        except:
            pdb.set_trace()
        df_cm.to_csv(self.config.ckpt_path + 'conf_mat.csv')

        #pdb.set_trace()

        plt.figure(figsize=(15,15))
        plot = sns.heatmap(df_cm, annot=True)
        fig = plot.get_figure()   
        plt.ylabel("actual")
        plt.tight_layout() 
        fig.savefig(self.config.ckpt_path + 'conf_mat.png')
        plt.clf()

        df_cm = pd.DataFrame(conf_mat_norm, index = label,columns = label)
        df_cm.to_csv(self.config.ckpt_path + 'conf_mat_norm.csv')
        
        plt.figure(figsize=(15,15))
        plot = sns.heatmap(df_cm, annot=True,fmt='.2f')
        fig = plot.get_figure()   
        plt.ylabel("actual")
        plt.tight_layout() 
        fig.savefig(self.config.ckpt_path + 'conf_mat_norm.png')
        plt.clf()

        prec_rec_f1 = precision_recall_fscore_support(y_true,y_pred)

        prec =list(prec_rec_f1[0])
        rec =list(prec_rec_f1[1])
        f1 =list(prec_rec_f1[2])

        prf = np.asarray([prec,rec,f1])
        prf = np.transpose(prf, (1, 0))

        df_cm = pd.DataFrame(prf, index = label,columns = ['prec','rec','f1'])

        plt.figure(figsize=(10,10))
        plot = sns.heatmap(df_cm, annot=True,fmt='.2f')
        fig = plot.get_figure()
        plt.ylabel("acc:{0:.2f}".format(acc))   
        fig.tight_layout() 
        fig.savefig(self.config.ckpt_path + str(epoch) + 'prf.png')
        plt.clf()
        
        df_cm.insert(3, "acc", acc, True) 

        df_cm.to_csv(self.config.ckpt_path  + str(epoch) + 'prf.csv')

    def save_checkpoint(self,epoch):

        if self.config.args.save_ckpt_dir != '':
            os.makedirs(self.config.args.save_ckpt_dir, exist_ok=True)
            #ckpt_model = self.model.module if hasattr(self.model, "module") else self.model
            logger.info("saving %s", (self.config.args.save_ckpt_dir + self.config.ckpt_name + '.pth') )

            ckpt_model = self.model.state_dict()

            ckpt_args = [ckpt_model,self.config.args]
            torch.save(ckpt_args, (self.config.args.save_ckpt_dir + self.config.ckpt_name  + '.pthx'))
            
            try:
                self.vis_plot(epoch)
            except:
                pass

    def save_checkpointall(self,epoch):

        if self.config.string_logs is not None:

            pwd = os.getcwd()

            if pwd == 'G:\\experiment\\litegpt':
                self.config.ckpt_path='./ckpt/' + self.config.string_logs
            elif pwd == '/csc/epitkane/projects/litegpt':
                self.config.ckpt_path='./ckptnew/' + self.config.string_logs
            else:
                self.config.ckpt_path='/scratch/project_2001668/primasan/ckpt/' + self.config.string_logs
            
            os.makedirs(self.config.ckpt_path, exist_ok=True)

            #ckpt_model = self.model.module if hasattr(self.model, "module") else self.model
            logger.info("saving %s", (self.config.ckpt_path + self.config.ckpt_name + '.pth') )

            ckpt_model = self.model.state_dict()
            #torch.save(ckpt_model, (self.config.ckpt_path + self.config.ckpt_name  + '.pth'))
            torch.save(self.model.state_dict(), (self.config.ckpt_path + self.config.ckpt_name  + str(epoch)+ '.pth'))


    def dynamic_stream(self):
        model, config = self.model, self.config
        model = model.to(self.device)

        model = torch.nn.DataParallel(model).to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9,weight_decay=config.weight_decay)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = data
            losses = []
            np.random.shuffle(selected_index)

            total = 0
            correct = 0

            self.all_predicted = []
            self.all_target = []

            for it in range(0,len(loader)):
            #for it in range(0,10):
                x,y = loader.__getitem__(selected_index[it])
                x_all = []
    
                for i in range(len(x[1])):
                    xi = x[1][i].to(self.device)
                    xi = xi.unsqueeze(0)
                    x_all.append(xi.to(self.device))
                
                y = y.to(self.device)
                y = y.unsqueeze(0)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    #try:
                    #pdb.set_trace()
                    logits, loss = model(x_all, y)
                    #except:
                    #pdb.set_trace()

                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    try:
                        losses.append(loss.item())
                    except:
                        pdb.set_trace()
                        
                    #compute accuracy
                    _, predicted = torch.max(logits.data, 1)

                    total += y.size(0)
                    correct += (predicted == y).sum().item()
                    acc = correct/total

                    self.all_predicted.append(predicted.detach().cpu().numpy())
                    self.all_target.append(y.detach().cpu().numpy())

                if is_train:
                    # backprop and update the parameters
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # report progress
                    if it % 1000 == 0:
                        print(f"epoch {epoch+1} iter {it}/{len(loader)}/{selected_index[it]}: train loss {loss.item():.5f}, acc {acc:.2f}, ({correct}/{total})")
                        #pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}, acc {acc:.2f}, ({correct}/{total})")
                        #self.save_checkpoint()

            if not is_train:
                logger.info("validation_loss : %f, validation acc: %f", np.mean(losses), acc)
                if acc > self.global_acc:
                    self.global_acc = acc
                    print(self.global_acc)
                    print('bestepoch-' + str(epoch))
                    self.save_checkpoint(epoch)
                    #self.save_checkpointall(epoch)

        self.tokens = 0 # counter used for learning rate decay

        self.global_acc = 0
        for epoch in range(config.max_epochs):
            selected_index = np.arange(len(self.train_dataset))
            run_epoch('train')
            if self.test_dataset is not None:
                selected_index = np.arange(len(self.test_dataset))
                run_epoch('test')

    def visualize_attention(self,visatt):
        model, config = self.model, self.config

        pwd = os.getcwd()

        if pwd == 'G:\\experiment\\litegpt':
            string_logs ='./ckptnew/' + self.config.string_logs
        elif pwd == '/csc/epitkane/projects/litegpt':
            string_logs ='./ckptnew/' + self.config.string_logs
        else:
            string_logs ='/scratch/project_2001668/primasan/ckpt/' + self.config.string_logs

        allloader = self.test_dataset

        if self.device == 'cpu':
            weight = torch.load(string_logs + self.config.ckpt_name + '.pthx',map_location=self.device)
        else:
            weight = torch.load(string_logs + self.config.ckpt_name + '.pthx')

        self.config.ckpt_path = string_logs

        model = model.to(self.device)
     
        model.load_state_dict(weight)

        split = 'test'
        is_train = split == 'train'

        model = model.to(self.device)

        model.train(is_train)

        #os.makedirs(self.config.ckpt_path + 'attentiondebug/', exist_ok=True) 
        #os.makedirs('E:/fullattention/', exist_ok=True) 

        for loader in allloader:
            for it in range(0,len(loader)):
            #for it in range(0,1):

                loader.block_size = self.config.args.block_size
                npx,npy = loader.__getitem__(it)

                print(str(it + 1) + ' ' + str(npx[0]) + str(npy))

                # place data on the correct device
                x_all = []
                for i in range(len(npx[1])):
                    xi = npx[1][i].to(self.device)
                    xi = xi.unsqueeze(0)
                    x_all.append(xi.to(self.device))
                    
                y = npy.to(self.device)
                y = y.unsqueeze(0)

                #pdb.set_trace()

                with torch.set_grad_enabled(is_train):
                    dot1 = model(x_all,visatt=visatt)
                
                dot1=dot1.cpu().numpy()

                dot1a = dot1[0]

                motifinput = npx[1][0].cpu().numpy()
                posinput = npx[1][1].cpu().numpy()

                pd_data = pd.DataFrame(motifinput)
                pd_data['position'] = posinput

                pd_data.columns = ['motif','position']

                pd_data.to_csv('/csc/epitkane/projects/MuAt/fullattention/inputdata/' + str(npx[0]))

                with open('/csc/epitkane/projects/MuAt/fullattention/attentionmatrix/'+ str(npx[0]) +'.npy', 'wb') as f:
                    np.save(f, dot1a)

    def predict(self,predictvis,adddata_dir):
        model, config = self.model, self.config

        pwd = os.getcwd()

        allloader = self.test_dataset

        split = 'test'
        is_train = split == 'train'

        model = model.to(self.device)

        model.train(is_train)

        #predict all
        list_feature = []
        for loader in allloader:

            #pdb.set_trace()
            for it in range(0,len(loader)):
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
                    pass
                else:
                    y = npy.to(self.device)
                    y = y.unsqueeze(0)

                with torch.set_grad_enabled(is_train):

                    try:
                        logits, loss = model(x_all)
                    except:
                        pdb.set_trace()
                    _, predicted = torch.max(logits.data, 1)

                    cpu_predicted = predicted.detach().cpu().numpy()[0]
                    class_name = loader.pd_class_info.loc[loader.pd_class_info['class_index']==cpu_predicted]['class_name'].values[0]

                    print('Predicting ' + npx[0][:-4] + ' as ' + str(class_name))

                    if predictvis:
                        features = model(x_all, y,predictvis)
                        features=features.cpu().numpy()[0]
                        list_feature.append(features)

        if predictvis:
            pd_feature = pd.DataFrame(list_feature)
            pd_feature.to_csv(self.config.args.output_data_dir + self.config.args.output_prefix + 'features.tsv',sep='\t')














