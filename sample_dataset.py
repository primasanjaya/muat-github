
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
import os
import pandas as pd
import pdb
import numpy as np
import math
import pickle
import random

class SampleDataset(Dataset):

    def __init__(self, dataset_name = None, data_dir=None, mode='training', block_size=128):
        
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.mode = mode

        self.dataset_root_folder = self.data_dir + '/'
        self.block_size = block_size

        os.makedirs(self.dataset_root_folder, exist_ok=True)
 
        self.read_dataset()

    def read_dataset(self):
        if self.mode=='training':   
            self.training_data = pd.read_csv(self.data_dir+'train.tsv',sep='\t')
        elif self.mode=='validation':
            self.validation_data = pd.read_csv(self.data_dir+'dev.tsv',sep='\t')
        elif self.mode=='testing':
            self.test_data = pd.read_csv(self.data_dir+'test.tsv',sep='\t')

        self.pd_vocab_counts = pd.read_csv(self.dataset_root_folder + 'vocab_counts.csv')

        self.vocab_size=len(self.pd_vocab_counts) + 1

    def tokenizing(self,raw_data):

        list_sentence = raw_data.split(" ")

        sentence_token = []

        for word in list_sentence:
            token = self.pd_vocab_counts.loc[self.pd_vocab_counts['vocab']==word]['index'].to_numpy()

            if token.size != 0:
                token = token[0]
                sentence_token.append(token)

        return np.array(sentence_token)
    
    def __len__(self):
        if self.mode=='training':
            return len(self.training_data)
        elif self.mode=='validation':
            return len(self.validation_data)
        elif self.mode=='testing':
            return len(self.test_data)

    def __getitem__(self, idx):       

        if self.mode=='training':
            instances=self.training_data.iloc[idx]
        elif self.mode=='validation':
            instances=self.validation_data.iloc[idx]
        elif self.mode=='testing':
            instances=self.test_data.iloc[idx]

        target = np.asarray(instances['label'])
        sentence_data = instances['sentence']
        
        sentence_token = self.tokenizing(sentence_data)

        mins = self.block_size - len(sentence_token)

        if mins < 0:
            sentence_token = sentence_token[0:args.block_size]
        else:
            sentence_token = np.copy(np.pad(sentence_token, ((0, mins)), mode='constant', constant_values=0))

        data = sentence_token.astype(np.int16)
        target = target.astype(np.int16)

        x = torch.tensor(data, dtype=torch.long)
        y = torch.tensor(target, dtype=torch.long)
        return x, y



if __name__ == '__main__':

    dataloader = SampleDataset(dataset_name = 'sample_data', data_dir='./data/', mode='training', block_size=128) 
    data,target = dataloader.__getitem__(0)
    pdb.set_trace()

    

    