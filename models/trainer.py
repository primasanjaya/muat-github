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
import shutil

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

        string_log = f"{self.config.args.tag}_{self.config.args.arch}_fo{self.config.args.fold:.0f}_bs{self.config.args.block_size:.0f}_nl{self.config.args.n_layer:.0f}_nh{self.config.args.n_head:.0f}_ne{self.config.args.n_emb:.0f}_ba{self.config.args.batch_size:.0f}_ee{self.config.args.epi_emb:.0f}/"
        self.complete_save_dir = self.config.args.save_ckpt_dir + string_log

        os.makedirs(self.complete_save_dir, exist_ok=True)
      
    def vis_plot(self,epoch):

        pd_logits = pd.read_csv(self.complete_save_dir + 'best_vallogits.tsv',sep='\t')

        y_true = pd_logits['target_name']
        y_score = pd_logits.iloc[:,0:-3]

        top5all = []
        for i in range(len(y_score)):
            row = y_score.iloc[i]
            row = row.sort_values(ascending=False)
            best5 = row[0:5].index.tolist()
            best5 = tuple(best5)
            top5all.append(best5)

        pd_top5 = pd.DataFrame(top5all)
        pd_top5.columns = ['top1','top2','top3','top4','top5']
        pd_logits = pd.concat([pd_logits, pd_top5], axis=1)
        pd_logits.to_csv(self.complete_save_dir + 'best_vallogits.tsv',sep='\t')

        to_check = pd_top5['top1']
        correct1 = y_true==to_check
        top1acc = sum(correct1)/len(correct1)

        to_check = pd_top5['top2']
        correct2 = y_true==to_check
        correct2 = correct1 | correct2
        top2acc = sum(correct2)/len(correct2)

        to_check = pd_top5['top3']
        correct3 = y_true==to_check
        correct3 = correct2 | correct3
        top3acc = sum(correct3)/len(correct3)

        to_check = pd_top5['top4']
        correct4 = y_true==to_check
        correct4 = correct3 | correct4
        top4acc = sum(correct4)/len(correct4)

        to_check = pd_top5['top5']
        correct5 = y_true==to_check
        correct5 = correct4 | correct5
        top5acc = sum(correct5)/len(correct5)

        acc = top1acc

        label = y_score.columns
        y_pred = pd_top5['top1']

        conf_mat = confusion_matrix(y_true, y_pred)
        conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

        df_cm = pd.DataFrame(conf_mat, index = label,columns = label)

        df_cm.to_csv(self.complete_save_dir + 'conf_mat.csv')

        plt.figure(figsize=(15,15))
        plot = sns.heatmap(df_cm, annot=True)
        fig = plot.get_figure()   
        plt.ylabel("actual")
        plt.tight_layout() 
        fig.savefig(self.complete_save_dir + 'conf_mat.png')
        plt.clf()

        df_cm = pd.DataFrame(conf_mat_norm, index = label,columns = label)
        df_cm.to_csv(self.complete_save_dir + 'conf_mat_norm.csv')
        
        plt.figure(figsize=(15,15))
        plot = sns.heatmap(df_cm, annot=True,fmt='.2f')
        fig = plot.get_figure()   
        plt.ylabel("actual")
        plt.tight_layout() 
        fig.savefig(self.complete_save_dir + 'conf_mat_norm.png')
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
        fig.savefig(self.complete_save_dir + str(epoch) + 'prf.png')
        plt.clf()
        
        df_cm.insert(3, "top1", acc, True)
        df_cm.insert(4, "top2", top2acc, True)
        df_cm.insert(5, "top3", top3acc, True) 
        df_cm.insert(6, "top4", top4acc, True)
        df_cm.insert(7, "top5", top5acc, True)

        df_cm.to_csv(self.complete_save_dir  + str(epoch) + 'prf.csv')

    def save_checkpoint(self,epoch):
        if self.config.args.save_ckpt_dir != '':
            #ckpt_model = self.model.module if hasattr(self.model, "module") else self.model
            logger.info("saving %s", (self.complete_save_dir + self.config.ckpt_name + '.pth') )

            ckpt_model = self.model.state_dict()

            ckpt_args = [ckpt_model,self.config.args]
            torch.save(ckpt_args, (self.complete_save_dir + self.config.ckpt_name  + '.pthx'))
            torch.save(ckpt_args, (self.complete_save_dir + self.config.ckpt_name + str(epoch)  + '.pth'))
            
            self.vis_plot(epoch)
            
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
            #self.all_logits = []
            
            #create logits header file
            if is_train:
                self.logit_filename = 'train_logits.tsv'
            else:
                self.logit_filename = 'val_logits.tsv'
                f = open(self.complete_save_dir + self.logit_filename, 'w+')  # open file in write mode
                header_class = loader.pd_class_info['class_name'].tolist()
                header_class.append('target')
                header_class.append('target_name')
                header_class.append('sample')
                write_header = "\t".join(header_class)
                f.write(write_header)
                f.close()

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

                    if not is_train:
                        logits_cpu =logits.detach().cpu().numpy()
                        f = open(self.complete_save_dir + self.logit_filename, 'a+')
                        logits_cpu = logits_cpu.flatten()
                        logits_cpu = logits_cpu.tolist()
                        f.write('\n')
                        write_logits = ["%.8f" % i for i in logits_cpu]
                        write_logits.append(str(y.detach().cpu().numpy().tolist()[0]))
                        write_logits.append(x[0][1])
                        write_logits.append(x[0][0])
                        write_header = "\t".join(write_logits)
                        f.write(write_header)
                        f.close()

                if is_train:
                    # backprop and update the parameters
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # report progress
                    
                    if it % 1000 == 0:
                        print(f"epoch {epoch+1} iter {it}/{len(loader)}/{selected_index[it]}: train loss {loss.item():.5f}, acc {acc:.2f}, ({correct}/{total})")

            if not is_train:
                logger.info("validation_loss : %f, validation acc: %f", np.mean(losses), acc)
                if acc > self.global_acc:
                    self.global_acc = acc
                    print(self.global_acc)
                    print('bestepoch-' + str(epoch))

                    shutil.copyfile(self.complete_save_dir + self.logit_filename, self.complete_save_dir + 'best_vallogits.tsv')
                    os.remove(self.complete_save_dir + self.logit_filename)
                    self.save_checkpoint(epoch)

        self.tokens = 0 # counter used for learning rate decay

        self.global_acc = 0
        for epoch in range(config.max_epochs):
            selected_index = np.arange(len(self.train_dataset))
            run_epoch('train')
            if self.test_dataset is not None:
                selected_index = np.arange(len(self.test_dataset))
                run_epoch('test')
    

    def batch_train(self):
        model, config = self.model, self.config
        model = model.to(self.device)

        model = torch.nn.DataParallel(model).to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9,weight_decay=config.weight_decay)

        batch_size = self.config.args.batch_size

        trainloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        valloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        self.global_acc = 0

        for e in range(config.max_epochs):
            running_loss = 0
            model.train(True)

            train_corr = 0
            for batch_idx, (data, target) in enumerate(trainloader):
                string_data = data[0]
                numeric_data = data[1]

                for i in range(len(numeric_data)):
                    numeric_data[i] = numeric_data[i].to(self.device)

                target = target.to(self.device)

                # forward the model
                with torch.set_grad_enabled(True):

                    optimizer.zero_grad()

                    logits, loss = model(numeric_data, target)
                    pred = logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    train_corr += pred.eq(target.view_as(pred)).sum().item()
                    
                    loss.backward()
                    #And optimizes its weights here
                    optimizer.step()
                    running_loss += loss.item()

                    train_acc = train_corr / len(self.train_dataset)

                    if batch_idx % 100 == 0:
                        print("Epoch {} - Training loss: {:.4f} - Training Acc: {:.2f}".format(e, running_loss/len(self.train_dataset),  train_acc))

            #val
            test_loss = 0
            correct = 0

            self.logit_filename = 'val_logits.tsv'
            f = open(self.complete_save_dir + self.logit_filename, 'w+')  # open file in write mode
            header_class = self.test_dataset.pd_class_info['class_name'].tolist()
            header_class.append('target')
            header_class.append('target_name')
            header_class.append('sample')
            write_header = "\t".join(header_class)
            f.write(write_header)
            f.close()

            model.train(False)
            for (data, target) in valloader:
                string_data = data[0]
                numeric_data = data[1]
                for i in range(len(numeric_data)):
                    numeric_data[i] = numeric_data[i].to(self.device)
                target = target.to(self.device)
                # forward the model
                with torch.set_grad_enabled(False):
                    logits, loss = model(numeric_data, target)
                    _, predicted = torch.max(logits.data, 1)

                    predicted = logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += predicted.eq(target.view_as(predicted)).sum().item()

                    #write logits
                    logits_cpu =logits.detach().cpu().numpy()
                    f = open(self.complete_save_dir + self.logit_filename, 'a+')
                    for i in range(numeric_data[0].shape[0]):
                        f.write('\n')
                        logits_cpu_flat = logits_cpu[i].flatten()
                        logits_cpu_list = logits_cpu_flat.tolist()    
                        write_logits = ["%.8f" % i for i in logits_cpu_list]
                        write_logits.append(str(target.detach().cpu().numpy().tolist()[0]))
                        write_logits.append(string_data[1][i])
                        write_logits.append(string_data[0][i])
                        write_header = "\t".join(write_logits)
                        f.write(write_header)
                    f.close()

            test_loss /= len(self.test_dataset)
            

            local_acc = correct / len(self.test_dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(self.test_dataset), 100. * local_acc))

            if local_acc > self.global_acc:
                self.global_acc = local_acc
                print(self.global_acc)
                shutil.copyfile(self.complete_save_dir + self.logit_filename, self.complete_save_dir + 'best_vallogits.tsv')
                os.remove(self.complete_save_dir + self.logit_filename)
                self.save_checkpoint(e)


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
            #self.all_logits = []
            
            #create logits header file
            if is_train:
                self.logit_filename = 'train_logits.tsv'
            else:
                self.logit_filename = 'val_logits.tsv'
                f = open(self.complete_save_dir + self.logit_filename, 'w+')  # open file in write mode
                header_class = loader.pd_class_info['class_name'].tolist()
                header_class.append('target')
                header_class.append('target_name')
                header_class.append('sample')
                write_header = "\t".join(header_class)
                f.write(write_header)
                f.close()

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














