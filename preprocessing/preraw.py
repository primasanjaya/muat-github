# %%
import pandas as pd
import pdb
import numpy as np
import os
import math

root_data = 'G:/experiment/data/new24classes/'
root_tcga = 'G:/experiment/data/new23classes/'

root_tcgacross = 'G:/experiment/data/crossdata/WES/'
root_pcross = 'G:/experiment/data/crossdata/WGS/'


nm_class = os.listdir(root_data)
nm_class1 = [i for i in nm_class if i[-4]!="." and i[0]!="."]

nm_class2 = os.listdir(root_tcga)
nm_class2 = [i for i in nm_class2 if len(i)==4]

nm_class3 = os.listdir(root_tcgacross)
nm_class3 = [i for i in nm_class3 if len(i)==4]

# %%
nm_class4 = os.listdir(root_pcross)
nm_class4 = [i for i in nm_class4 if i[-4]!="." and i[0]!="."]

# %%
nm_class4

# %%
'''
for count in range(0,len(nm_class2)):
    i = nm_class2[count]
    nm_samples = os.listdir(root_tcga + i)
    
    for nm in nm_samples:
        if nm[:8] == "tokennew" :
            os.remove(root_tcga + i + '/' + nm)

'''   

# %% [markdown]
# nm_class

# %%
triplet96 = pd.read_csv('../extfile/96triplet.csv')

# %%
tr96 = triplet96['0'].to_list()

# %%
mutcode = pd.read_csv('/csc/epitkane/projects/litegpt/extfile/mutation_codes_sv_mei_rearrange.csv')

# %%
tr96 = set(tr96)

# %%
mutcode = mutcode[['from','to','symbol','group','translate']]

# %%
mutcode.dropna(inplace=True)

# %%
mutcode

# %%
dictSeq = set()
dictGES = set()
dictChromPos = set()

# %%
dictSeq = set()
dictGES = set()
dictChromPos = set()





# %%
allSNV = pd.DataFrame()

# %%
allSNV

# %%
#recheck global Seq Dictionary
allSNV = pd.DataFrame()

for count in range(0,len(nm_class1)):
                   
    i = nm_class1[count]
    nm_samples = os.listdir(root_data + i)
    
    for j in nm_samples:
        if j[:3] == 'new':
            #pdb.set_trace()
            pd_data = pd.read_csv(root_data + i + '/' + j)
            print(root_data + i + '/' + j)
            pd_data = pd_data[['seq','typ']]
            allSNV = allSNV.append(pd_data[['seq','typ']])
            
for count in range(0,len(nm_class2)):
    i = nm_class2[count]
    nm_samples = os.listdir(root_tcga + i)
    
    for j in nm_samples:
        if j[:3] == 'new':
            pd_data = pd.read_csv(root_tcga + i + '/' + j)
            print(root_data + i + '/' + j)
            pd_data = pd_data[['seq','typ']]
            allSNV = allSNV.append(pd_data[['seq','typ']])
            
        




# %%


# %%
allSNV = allSNV.sort_values(by=['typ'],ascending=False)

# %%
allSNV.to_csv('G:/experiment/data/allsnv.csv')

# %%
NiSionly = allSNV.loc[allSNV['typ']=='NiSi']

# %%
NiSionly = NiSionly.drop_duplicates(subset="seq")

# %%
SNVonly = allSNV.loc[allSNV['typ']=='SNV']

# %%
SNVonly = SNVonly.drop_duplicates(subset="seq")

# %%
indelonly = allSNV.loc[allSNV['typ']=='indel']
indelonly = indelonly.drop_duplicates(subset="seq")

# %%
MEIonly = allSNV.loc[allSNV['typ']=='MEI']
MEIonly = MEIonly.drop_duplicates(subset="seq")

# %%
SVonly = allSNV.loc[allSNV['typ']=='SV']
SVonly = SVonly.drop_duplicates(subset="seq")

# %%
Normalonly = allSNV.loc[allSNV['typ']=='Normal']
Normalonly = Normalonly.drop_duplicates(subset="seq")

# %%
len(NiSionly) + len(SNVonly) + len(indelonly) + len(MEIonly) + len(SVonly) + len(Normalonly)

# %%
appendallDict = pd.read_csv(root_tcga + i + '/' + j)

# %%
SNVonly.to_csv('dictSNV.csv')

indelonly.to_csv('dicIndel.csv')

MEIonly.to_csv('dictMEI.csv')

SVonly.to_csv('dictSV.csv')

Normalonly.to_csv('dictNormal.csv')

# %%
appendallDict = pd.read_csv(root_tcga + i + '/' + j)

# %%
NiSiorder = pd.read_csv('dictNiSi.csv',index_col=0)

# %%
allDict = NiSiorder.append(SNVonly.append(indelonly.append(MEIonly.append(SVonly.append(Normalonly))))).reset_index(drop=True)

# %%


# %%
dictMutation = pd.read_csv('dictMutation.csv',index_col=0)

# %%
dictGES = pd.read_csv('dictGES.csv',index_col=0)

# %%
dictChpos = pd.read_csv('dictChpos.csv',index_col=0)

# %%


# %%


# %%


# %%


# %%
#ready to convert all

# %%


# %%


# %%
for count in range(0,len(nm_class1)):
                   
    i = nm_class1[count]
    nm_samples = os.listdir(root_data + i)
    print(root_data + i)
    
    for j in nm_samples:
        #pdb.set_trace()
        if j[0:3] == 'new':    
            pd_data = pd.read_csv(root_data + i + '/' + j,index_col=0)

            
            
            mergetriplet = pd_data.merge(dictMutation, left_on='seq', right_on='triplet', how='left')
            
            mergeges = mergetriplet.merge(dictGES, left_on='ges', right_on='ges', how='left')
            
            mergechrompos = mergeges.merge(dictChpos, left_on='chrompos', right_on='chrompos', how='left')
            
            mergeAlltoken =  mergechrompos[['triplettoken', 'token_y','token_x','rt','typ_y']]
            mergeAlltoken = mergeAlltoken.rename(columns={"token_y": "postoken", "token_x": "gestoken", "typ_y" : "type"})
            
            NiSionly = mergeAlltoken.loc[mergeAlltoken['type']=='NiSi']
            NiSionly = NiSionly.drop(columns=['type'])
            
            SNVonly = mergeAlltoken.loc[mergeAlltoken['type']=='SNV']
            SNVonly = SNVonly.drop(columns=['type'])
            
            indelonly = mergeAlltoken.loc[mergeAlltoken['type']=='indel']
            indelonly = indelonly.drop(columns=['type'])
            
            MEISVonly = mergeAlltoken.loc[mergeAlltoken['type'].isin(['MEI','SV'])]
            
            Normalonly = mergeAlltoken.loc[mergeAlltoken['type']=='Normal']
            Normalonly = Normalonly.drop(columns=['type'])
            
            try:
                os.remove(root_data + i + '/' + 'SV_' + j)
                os.remove(root_data + i + '/' + 'MEI_' + j)
            except:
                pass
            
            
            NiSionly.to_csv(root_data + i + '/' + 'NiSi_' + j)
            SNVonly.to_csv(root_data + i + '/' + 'SNV_' + j)
            indelonly.to_csv(root_data + i + '/' + 'indel_' + j)
            MEISVonly.to_csv(root_data + i + '/' + 'MEISV_' + j)
            Normalonly.to_csv(root_data + i + '/' + 'Normal_' + j)
            
            pd_count = pd.DataFrame([len(NiSionly),len(SNVonly),len(indelonly),len(MEISVonly),len(Normalonly)])
            
            pd_count.to_csv(root_data + i + '/' + 'count_' + j)
            
            

# %%
for count in range(0,len(nm_class2)):
                   
    i = nm_class2[count]
    nm_samples = os.listdir(root_tcga + i)
    
    print(root_data + i + '/')
    
    for j in nm_samples:
        #pdb.set_trace()
        if j[0:3] == 'new':    
            pd_data = pd.read_csv(root_tcga + i + '/' + j,index_col=0)
            
            mergetriplet = pd_data.merge(dictMutation, left_on='seq', right_on='triplet', how='left')
            
            mergeges = mergetriplet.merge(dictGES, left_on='ges', right_on='ges', how='left')
            
            mergechrompos = mergeges.merge(dictChpos, left_on='chrompos', right_on='chrompos', how='left')
            
            mergeAlltoken =  mergechrompos[['triplettoken', 'token_y','token_x','rt','typ_y']]
            mergeAlltoken = mergeAlltoken.rename(columns={"token_y": "postoken", "token_x": "gestoken", "typ_y" : "type"})
            
            NiSionly = mergeAlltoken.loc[mergeAlltoken['type']=='NiSi']
            NiSionly = NiSionly.drop(columns=['type'])
            
            SNVonly = mergeAlltoken.loc[mergeAlltoken['type']=='SNV']
            SNVonly = SNVonly.drop(columns=['type'])
            
            indelonly = mergeAlltoken.loc[mergeAlltoken['type']=='indel']
            indelonly = indelonly.drop(columns=['type'])
            
            MEISVonly = mergeAlltoken.loc[mergeAlltoken['type'].isin(['MEI','SV'])]
            
            Normalonly = mergeAlltoken.loc[mergeAlltoken['type']=='Normal']
            Normalonly = Normalonly.drop(columns=['type'])
            
            try:
                os.remove(root_data + i + '/' + 'SV_' + j)
                os.remove(root_data + i + '/' + 'MEI_' + j)
            except:
                pass
            
            NiSionly.to_csv(root_tcga + i + '/' + 'NiSi_' + j)
            SNVonly.to_csv(root_tcga + i + '/' + 'SNV_' + j)
            indelonly.to_csv(root_tcga + i + '/' + 'indel_' + j)
            MEISVonly.to_csv(root_tcga + i + '/' + 'MEISV_' + j)
            Normalonly.to_csv(root_tcga + i + '/' + 'Normal_' + j)
            
            
            pd_count = pd.DataFrame([len(NiSionly),len(SNVonly),len(indelonly),len(MEISVonly),len(Normalonly)])
            
            pd_count.to_csv(root_tcga + i + '/' + 'count_' + j)

# %%
pd_combine = pd.read_csv('dictseq.csv',index_col=0)

pd_dictges = pd.read_csv('dictGES.csv',index_col=0)

pd_dictChpos = pd.read_csv('dictChpos',index_col=0)


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%
pd_comb = pd.read_csv('pd_comb_ok.csv',index_col=0)

# %%
pd_comb

# %%
pd_combine = pd_comb

# %%
pd_dictges = pd.DataFrame(dictGES)

# %%
pd_dictges['token'] = pd_dictges.index.to_list()
pd_dictges['token'] = pd_dictges['token'] + 1

# %%
pd_dictges = pd_dictges.rename(columns={0:'ges'})

# %%
pd_dictges

# %%
pd_dictges.to_csv('dictGES.csv')

# %%


# %%


# %%


# %%


# %%


# %%


# %%
pd_dictChpos = pd.DataFrame(dictChromPos)

# %%
pd_dictges

# %%
pd_dictges

# %%


# %%
from natsort import natsorted, index_natsorted, order_by_index

# %%
pd_dictChpos = pd_dictChpos.reindex(index=natsorted(pd_dictChpos[0]))

# %%
pd_dictChpos = pd_dictChpos.reset_index()

# %%


# %%
pd_dictChpos = pd_dictChpos.rename(columns={0:'nothing','index':'chrompos'})

# %%
pd_dictChpos = pd_dictChpos.drop(['nothing'],axis=1)

# %%
pd_dictChpos['token'] = pd_dictChpos.index.to_list()

# %%
pd_dictChpos['token'] = pd_dictChpos['token'] + 1

# %%
pd_dictChpos.to_csv('dictChpos.csv')

# %%
#collectdict

allTriplet = []
for count in range(0,len(nm_class1)):
                   
    i = nm_class1[count]
    nm_samples = os.listdir(root_data + i)
    
    for j in nm_samples:
        #pdb.set_trace()
        if j[0:3] == 'new':           
            pd_data = pd.read_csv(root_data + i + '/' + j,index_col=0)
            #pd_data = pd.read_csv('/csc/epitkane/projects/PCAWG20191001/data/modified_data/train/new24classes/Bone-Osteosarc/new_f856fa85-fdb8-c0b0-e040-11ac0d480b4e.csv')

            print(root_data + i + '/' + j)
            
            trip = pd_data['seq'].tolist()
            allTriplet = allTriplet + trip

# %%
len(allTriplet)

# %%
dictTrip = set(allTriplet)

# %%
len(dictTrip)

# %%
pd_2 = pd.DataFrame(tr96)

# %%


# %%
len(set(dictTrip))

# %%
non96 = dictTrip - tr96

# %%
len(non96)

# %%
pd_seq = 

# %%
for count in range(0,len(nm_class2)):
                   
    i = nm_class2[count]
    nm_samples = os.listdir(root_tcga + i)
    
    for j in nm_samples:
        #pdb.set_trace()
        if j[0:3] == 'new':           
            pd_data = pd.read_csv(root_tcga + i + '/' + j,index_col=0)
            
            print(root_tcga + i + '/' + j)
            
            trip = pd_data['seq'].tolist()
            
            allTriplet = allTriplet + trip

# %%

            

# %%


# %%
pd_combine.to_csv('pd_comb_ok.csv')

# %%
samples=[]
class_name=[]
num_mutation=[]


for count in range(0,len(nm_class1)):
                   
    i = nm_class1[count]
    nm_samples = os.listdir(root_data + i)
    
    for j in nm_samples:
        
        if j[0:3] == 'new':
            pd_data = pd.read_csv(root_data + i + '/' + j,index_col=0)
            
            numut = len(pd_data)
            
            num_mutation.append(numut)
            class_name.append(i)
            samples.append(j[4:])

# %%
pd_combine

# %%
pd_data = pd.DataFrame()

# %%
pd_data['nm_class'] = class_name

# %%
pd_data['samples'] = samples

# %%
pd_data['numut'] = num_mutation

# %%
pd_data.to_csv('numberofmutation_pcawg.csv')

# %%
medianall = []
mutationspersamples = []
num_samples = []

for i in range(len(pd_data.nm_class.unique())):
    class_name = pd_data.nm_class.unique()[i]
    selected = pd_data.loc[pd_data['nm_class']==class_name]
    
    num_samples.append(len(selected))
    medianall.append(np.median(selected.numut.values))
    mutationspersamples.append(sum(selected.numut.values))

# %%
pd_classinfo = pd.DataFrame()
pd_classinfo['class_name'] = nm_class1
pd_classinfo['n_samples'] = num_samples
pd_classinfo['median'] = medianall




# %%
pd_classinfo.to_csv('classinfo_pcawg.csv')

# %%


# %%
samples=[]
class_name=[]
num_mutation=[]

for count in range(0,len(nm_class2)):
                   
    i = nm_class2[count]
    nm_samples = os.listdir(root_tcga + i)
    
    for j in nm_samples:
        
        if j[0:3] == 'new':
            pd_data = pd.read_csv(root_tcga + i + '/' + j,index_col=0)
            
            numut = len(pd_data)
            
            num_mutation.append(numut)
            class_name.append(i)
            samples.append(j[4:])

# %%
pd_data = pd.DataFrame()
pd_data['nm_class'] = class_name
pd_data['samples'] = samples
pd_data['numut'] = num_mutation

# %%
pd_data.to_csv('numberofmutation_tcga.csv')

# %%
pd_data

# %%
medianall = []
mutationspersamples = []
num_samples = []

for i in range(len(pd_data.nm_class.unique())):
    class_name = pd_data.nm_class.unique()[i]
    selected = pd_data.loc[pd_data['nm_class']==class_name]
    
    num_samples.append(len(selected))
    medianall.append(np.median(selected.numut.values))
    mutationspersamples.append(sum(selected.numut.values))
    
pd_classinfo = pd.DataFrame()
pd_classinfo['class_name'] = nm_class2
pd_classinfo['n_samples'] = num_samples
pd_classinfo['median'] = medianall

# %%
pd_classinfo

# %%
pd_classinfo.to_csv('classinfo_tcga.csv')

# %%


# %%


# %%
#create 10folds

pd_data = pd.DataFrame()
listfold = []

for count in range(0,len(nm_class1)):
                   
    i = nm_class1[count]
    nm_samples = os.listdir(root_data + i)
    nm_samples = [x for x in nm_samples if x[:10]=='count_new_']
    
    fold = 0
    for j in nm_samples: 
        
        if fold >= 10:
            fold = 1
        else:
            fold = fold + 1 
        nm_class = i
        samples = j[10:]
        count = pd.read_csv(root_data + i + '/' + 'count_new_' + samples,index_col=0)

        count = count['0'].tolist()

        allcombine = [nm_class] + [samples] + count

        pd_allcombine = pd.DataFrame(allcombine).T

        pd_data = pd_data.append(pd_allcombine)
        listfold.append(fold)
        
pd_data = pd_data.reset_index(drop=True)
pd_data = pd_data.rename(columns={0:'nm_class',1:'samples',2:'NiSi',3:'SNV',4:'indel',5:'SVMEI',6:'Normal'})





# %%
pd_data['fold'] = listfold

# %%
pd_data.to_csv('allfold_pcawg.csv')

# %%
#remove SiNi zero
pd_data = pd_data.drop(pd_data.loc[pd_data['NiSi']==0].index)

# %%
listfold = []
fold = 0
for i in range(len(pd_data)):
    if fold >= 10:
        fold = 1
    else:
        fold = fold + 1 
        
    listfold.append(fold)

# %%


# %%


# %%

#create 10folds tcga

pd_datatcga = pd.DataFrame()
listfold = []

for count in range(0,len(nm_class2)):
                   
    i = nm_class2[count]
    nm_samples = os.listdir(root_tcga + i)
    nm_samples = [x for x in nm_samples if x[:10]=='count_new_']
    
    fold = 0
    for j in nm_samples:
        
        if fold >= 10:
            fold = 1
        else:
            fold = fold + 1 
            
        nm_class = i
        samples = j[10:]
        count = pd.read_csv(root_tcga + i + '/' + 'count_new_' + samples,index_col=0)

        count = count['0'].tolist()

        allcombine = [nm_class] + [samples] + count

        pd_allcombine = pd.DataFrame(allcombine).T

        pd_datatcga = pd_datatcga.append(pd_allcombine)
        listfold.append(fold)
        
pd_datatcga = pd_datatcga.reset_index(drop=True)
pd_datatcga = pd_datatcga.rename(columns={0:'nm_class',1:'samples',2:'NiSi',3:'SNV',4:'indel',5:'SVMEI',6:'Normal'})



# %%
pd_datatcga['fold'] = listfold

# %%
pd_datatcga.to_csv('allfold_tcga.csv')

# %%
for valfold in range(1,11):
    val = pd_data.loc[pd_data['fold']==valfold]
    train = pd_data.loc[pd_data['fold']!=valfold]
    
    val.to_csv('pcawg_valfold' + str(valfold) + '.csv')
    train.to_csv('pcawg_trainfold' + str(valfold) + '.csv')

# %%
for valfold in range(1,11):
    val = pd_datatcga.loc[pd_datatcga['fold']==valfold]
    train = pd_datatcga.loc[pd_datatcga['fold']!=valfold]
    
    val.to_csv('tcga_valfold' + str(valfold) + '.csv')
    train.to_csv('tcga_trainfold' + str(valfold) + '.csv')

# %%
pd_datatcga.groupby(['nm_class']).size()

# %%


# %%
nm_class3

# %%
#create 10folds tcga

pd_datatcga = pd.DataFrame()
listfold = []

for count in range(0,len(nm_class3)):
                   
    i = nm_class3[count]
    nm_samples = os.listdir(root_tcga + i)
    nm_samples = [x for x in nm_samples if x[:10]=='count_new_']
    
    fold = 0
    for j in nm_samples:
        
        if fold >= 10:
            fold = 1
        else:
            fold = fold + 1 
            
        nm_class = i
        samples = j[10:]
        count = pd.read_csv(root_tcga + i + '/' + 'count_new_' + samples,index_col=0)

        count = count['0'].tolist()

        allcombine = [nm_class] + [samples] + count

        pd_allcombine = pd.DataFrame(allcombine).T

        pd_datatcga = pd_datatcga.append(pd_allcombine)
        listfold.append(fold)
        
pd_datatcga = pd_datatcga.reset_index(drop=True)
pd_datatcga = pd_datatcga.rename(columns={0:'nm_class',1:'samples',2:'NiSi',3:'SNV',4:'indel',5:'SVMEI',6:'Normal'})

# %%
pd_datatcga.to_csv('allwes.csv')

# %%
pd_datatcga['fold'] = listfold

# %%
pd_datatcga

# %%
for valfold in range(1,11):
    val = pd_datatcga.loc[pd_datatcga['fold']==valfold]
    train = pd_datatcga.loc[pd_datatcga['fold']!=valfold]
    
    val.to_csv('tcgawes_valfold' + str(valfold) + '.csv')
    train.to_csv('tcgawes_trainfold' + str(valfold) + '.csv')

# %%


# %%
#create 10folds

pd_data = pd.DataFrame()
listfold = []

for count in range(0,len(nm_class4)):
                   
    i = nm_class4[count]
    nm_samples = os.listdir(root_pcross + i)
    nm_samples = [x for x in nm_samples if x[:10]=='count_new_']
    
    fold = 0
    print(i)
    for j in nm_samples: 
        
        if fold >= 10:
            fold = 1
        else:
            fold = fold + 1 
        nm_class = i
        samples = j[10:]
        count = pd.read_csv(root_pcross + i + '/' + 'count_new_' + samples,index_col=0)

        count = count['0'].tolist()

        allcombine = [nm_class] + [samples] + count
        #pdb.set_trace()

        pd_allcombine = pd.DataFrame(allcombine).T

        pd_data = pd_data.append(pd_allcombine)
        listfold.append(fold)
        
pd_data = pd_data.reset_index(drop=True)
pd_data = pd_data.rename(columns={0:'nm_class',1:'samples',2:'NiSi',3:'SNV',4:'indel',5:'SVMEI',6:'Normal'})

# %%
set(pd_data.nm_class.to_list())

# %%


# %%
root_pcross

# %%
pd_data['fold'] = listfold

# %%
pd_data.to_csv('allgws.csv')

# %%
for valfold in range(1,11):
    val = pd_data.loc[pd_data['fold']==valfold]
    train = pd_data.loc[pd_data['fold']!=valfold]
    
    val.to_csv('pcawgwgs_valfold' + str(valfold) + '.csv')
    train.to_csv('pcawgwgs_trainfold' + str(valfold) + '.csv')
    
   # tcgawes_trainfold

# %%
pd_data

# %%


# %%



