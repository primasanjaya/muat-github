# make deterministic
from mingpt.utils import set_seed
set_seed(42)

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset

from mingpt.model import GPT, GPTConfig,GPTForClassification

from mingpt.trainer import Trainer, TrainerConfig
from mingpt.utils import sample

import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)
import pdb
from sample_dataset import SampleDataset

if __name__ == '__main__':

    block_size = 1000 # spatial extent of the model for its context
    train_dataset = SampleDataset(dataset_name = 'sample_data', data_dir='./data/', mode='training', block_size=block_size)    
    validation_dataset = SampleDataset(dataset_name = 'sample_data', data_dir='./data/', mode='validation', block_size=block_size)

    num_class = 2

    mconf = GPTConfig(train_dataset.vocab_size, block_size,num_class,
                    n_layer=8, n_head=8, n_embd=512)
    model = GPTForClassification(mconf)

    tconf = TrainerConfig(max_epochs=200, batch_size=12, learning_rate=6e-4,
                        lr_decay=True, warmup_tokens=12*20, final_tokens=200*len(train_dataset)*block_size,
                        num_workers=4,ckpt_path='./logs_/')
    trainer = Trainer(model, train_dataset, validation_dataset, tconf)
    trainer.train()

    print('Done')


