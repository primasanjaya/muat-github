import numpy as np
import pdb
import pandas as pd


comb1 = np.arange(22)+1
comb2 = [250,244,199,192,181,172,160,147,142,136,136,134,116,108,103,91,82,79,60,64,49,52]
comb3 = ['$','%','^',':',';','?']
comb4 = ['[C>A]','[C>G]','[C>T]','[T>A]','[T>C]','[T>G]']


sum_comb2 = np.arange(sum(comb2))

vocab_dict = []
clean_vocab_dict = []

for i in range(0,len(comb3)):
    for j in range(0,len(comb1)):
        for k in range(0,len(comb2)):
            for l in range(0,comb2[k]):
                firstchar = comb1[j]
                secondchar = l
                thirdchar = comb3[i]
                fourthchar = comb4[i]

                vocab = '_'.join([str(firstchar),str(secondchar),str(thirdchar)])
                #vocab2 = '_'.join([str(firstchar),str(secondchar),str(fourthchar)])

                vocab_dict.append(vocab)
                #clean_vocab_dict.append(vocab2)

vocab_dict=sorted(list(set(vocab_dict)))

for i in vocab_dict:
    words = i.split('_')

    if words[-1]=='$':
        tr = '[C>A]'
    elif words[-1]=='%':
        tr = '[C>G]'
    elif words[-1]=='^':
        tr = '[C>T]'
    elif words[-1]==':':
        tr = '[T>A]'
    elif words[-1]==';':
        tr = '[T>C]'
    elif words[-1]=='?':
        tr = '[T>G]'

    translate = '_'.join([str(words[0]),str(words[1]),str(tr)])
    clean_vocab_dict.append(translate)

data = list(zip(vocab_dict, clean_vocab_dict))  
pd_vocab = pd.DataFrame(data,columns=['vocab','translate'])

pd_vocab.to_csv('../extfile/vocab.csv')


#clean_vocab_dict=sorted(list(set(clean_vocab_dict)))





