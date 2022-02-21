import numpy as np 
import pdb
import pandas as pd

data_dir = './data/'


def count_vocab(pd_data,word_dictionary):
    for i in range (0,len(pd_data)):
        sentence = pd_data['sentence'][i]
        list_sentence = sentence.split(" ")

        for word in list_sentence:
            if word.isalnum():
                if word in word_dictionary:
                    word_dictionary[word] = word_dictionary.get(word) + 1
                else:
                    word_dictionary[word] = 1

    return word_dictionary

training_data = pd.read_csv(data_dir+'train.tsv',sep='\t')
validation_data = pd.read_csv(data_dir+'dev.tsv',sep='\t')
test_data = pd.read_csv(data_dir+'test.tsv',sep='\t')

word_dictionary = {}
word_dictionary = count_vocab(training_data,word_dictionary)
word_dictionary = count_vocab(validation_data,word_dictionary)
word_dictionary = count_vocab(test_data,word_dictionary)

pd_vocab_counts = pd.DataFrame.from_dict(word_dictionary,orient='index',columns=['counts']).sort_values(by='counts',ascending=False).rename_axis('vocab').reset_index()
pd_vocab_counts.index = pd_vocab_counts.index + 1
pd_vocab_counts = pd_vocab_counts.rename_axis('index').reset_index()
pd_vocab_counts.to_csv(data_dir + 'vocab_counts.csv')