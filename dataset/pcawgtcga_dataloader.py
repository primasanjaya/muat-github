
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
from sklearn.utils import shuffle

class TCGAPCAWG_Dataloader(Dataset):

    def __init__(self, dataset_name = None, 
                data_dir=None, 
                mode='training', 
                curr_fold=1, 
                block_size=5000, 
                load=False,
                addtriplettoken=False,
                addpostoken=False,
                addgestoken=False,
                addrt=False,
                nummut = 0,
                frac = 0,
                crossdata=False,
                crossdatadir=None,
                pcawg2tgca_class=False,
                tcga2pcawg_class=False,
                mutratio = '1-0-0-0-0',
                adddatadir = None,
                input_filename=None,
                args = None,
                gx_dir=None,
                addepigen=False):

        self.dataset_name = dataset_name
        self.data_dir=data_dir
        self.mode=mode
        self.curr_fold=int(curr_fold)
        self.block_size=block_size
        self.load=load
        self.addtriplettoken=addtriplettoken
        self.addpostoken=addpostoken
        self.addrt=addrt
        self.nummut = nummut
        self.frac = frac
        self.addgestoken = addgestoken
        self.crossdata= crossdata
        self.crossdatadir = crossdatadir
        self.adddatadir = adddatadir
        self.args = args
        self.gx_dir = gx_dir
        self.mutratio = mutratio

        self.pcawg2tgca_class=pcawg2tgca_class
        self.tcga2pcawg_class=tcga2pcawg_class

        self.newformat = True
        self.newformat = False

        self.NiSi = False
        self.SNV = False
        self.indel = False
        self.SVMEI = False
        self.Normal = False

        self.dnn_input = 1

        if self.args == None:
            self.single_pred_vcf = False
            self.cwd = str(os.path.abspath('..')) + '/'
        else:
            self.single_pred_vcf = self.args.single_pred_vcf
            self.cwd = self.args.cwd


        self.input_filename = input_filename

        if self.nummut > 0 :
            self.block_size = self.nummut

        if self.dataset_name == 'pcawg':
            if self.args.multi_pred_vcf:
                fulltuple = []
                for idx in range(len(input_filename)):
                    va = input_filename[idx]
                    onetup = (va[:-4],'',1,1)
                    #print(onetup)
                    fulltuple.append(onetup)
                self.validation_fold = pd.DataFrame(fulltuple,columns =['samples', 'nm_class', 'slices','fold'])
                self.test_fold = self.validation_fold
                self.newformat = True
            
            if self.single_pred_vcf:
                self.onlyfilename = self.args.input_filename[:-4]
                onetup = [(self.onlyfilename,'',1,1)]
                self.validation_fold = pd.DataFrame(onetup,columns =['samples', 'nm_class', 'slices','fold'])
                self.test_fold = self.validation_fold
                
                self.newformat = True


            '''
            else:
                if self.newformat:
                    self.training_fold = pd.read_csv('./dataset_utils/pcawg_train.csv',index_col=0)
                    self.training_fold = self.training_fold.loc[self.training_fold['fold']==self.curr_fold]
                    self.validation_fold = pd.read_csv('./dataset_utils/pcawg_val.csv',index_col=0)
                    self.validation_fold = self.validation_fold.loc[self.validation_fold['fold']==self.curr_fold]
                else:
                    self.training_fold = pd.read_csv('./oldformat/pcawg_trainfold' + str(self.curr_fold) + '.csv',index_col=0)
                    self.validation_fold = pd.read_csv('./oldformat/pcawg_valfold' + str(self.curr_fold) + '.csv',index_col=0)
            '''
        elif self.dataset_name == 'tcga':
            self.training_fold = pd.read_csv('./dataset_utils/tcga_trainfold' + str(self.curr_fold) + '.csv',index_col=0)
            self.validation_fold = pd.read_csv('./dataset_utils/tcga_valfold' + str(self.curr_fold) + '.csv',index_col=0)
        elif self.dataset_name == 'westcga':
            self.training_fold = pd.read_csv('./dataset_utils/tcgawes_trainfold' + str(self.curr_fold) + '.csv',index_col=0)
            self.validation_fold = pd.read_csv('./dataset_utils/tcgawes_valfold' + str(self.curr_fold) + '.csv',index_col=0)
        elif self.dataset_name == 'wgspcawg':
            self.training_fold = pd.read_csv('./dataset_utils/pcawgwgs_trainfold' + str(self.curr_fold) + '.csv',index_col=0)
            self.validation_fold = pd.read_csv('./dataset_utils/pcawgwgs_valfold' + str(self.curr_fold) + '.csv',index_col=0)

        if self.dataset_name == 'wgsgx':
            self.gx = pd.read_csv(self.gx_dir + 'PCAWG_gene_expression.tsv',sep='\t',index_col=0)
            #self.gx = self.gx.iloc[:,-100:]

            self.training_fold = pd.read_csv(self.cwd + 'dataset_utils/wgsgx_train.csv',index_col=0)
            self.training_fold = self.training_fold.loc[self.training_fold['fold'] == self.curr_fold]

            self.validation_fold = pd.read_csv(self.cwd + 'dataset_utils/wgsgx_val.csv',index_col=0)
            self.validation_fold = self.validation_fold.loc[self.validation_fold['fold'] == self.curr_fold]  
            self.newformat = True

            self.dnn_input = len(self.gx.iloc[0,:-2])
            #pdb.set_trace()

        if self.adddatadir is not None:
            adddata = pd.DataFrame(columns=self.validation_fold.columns)
            adddata.columns = self.validation_fold.columns

            folder = os.listdir(self.adddatadir)

            for i in folder:

                samples = os.listdir(self.adddatadir + i )
                for j in samples:
                    if j[0:3] == 'new':
                        counter = pd.read_csv(self.adddatadir + i + '/count_new_' + j[4:],index_col=0)

                        listall = [i,j[4:]] + counter['0'].values.tolist() + [1]

                        pds = pd.DataFrame(listall)
                        pds = pds.T
                        pds.columns=self.validation_fold.columns

                        adddata = adddata.append(pds)

            adddata = adddata.reset_index(drop=True)

            self.adddata = adddata

            #self.validation_fold = self.validation_fold.append(self.adddata)
            self.validation_fold = self.adddata
            self.data_dir = self.adddatadir

        if self.single_pred_vcf:
            samples_names = input_filename[:-4]
            pd_count = pd.read_csv(args.tmp_dir + 'count_' + input_filename[:-4] + '.csv', index_col=0)['0'].to_list()
            onerow = ['',samples_names] + pd_count + [1]
            pd_data = pd.DataFrame(onerow).T
            pd_data.columns = ['nm_class','samples','NiSi','SNV','indel','SVMEI','Normal','fold']
            self.validation_fold = pd_data
            self.test_fold = self.validation_fold

        self.load_classinfo()

        self.vocab_mutation = pd.read_csv(self.cwd + 'extfile/dictMutation.csv',index_col=0)
        self.allSNV_index = 0

        if self.mutratio is not None:
            self.mutratio = mutratio.split('-')
            self.mutratio = [float(i) for i in self.mutratio]
        
            if self.mutratio[0]>0:
                self.NiSi = True 
            if self.mutratio[1]>0:
                self.SNV = True
            if self.mutratio[2]>0:
                self.indel = True
            if self.mutratio[3]>0:
                self.SVMEI = True
            if self.mutratio[4]>0:
                self.Normal = True

        vocabsize = 0
        if self.NiSi:
            vocabsize = len(self.vocab_mutation.loc[self.vocab_mutation['typ']=='NiSi'])
        if self.SNV:
            vocabsize = vocabsize + len(self.vocab_mutation.loc[self.vocab_mutation['typ']=='SNV'])
        if self.indel:
            vocabsize = vocabsize + len(self.vocab_mutation.loc[self.vocab_mutation['typ']=='indel'])                   
        if self.SVMEI:
            vocabsize = vocabsize + len(self.vocab_mutation.loc[self.vocab_mutation['typ'].isin(['MEI','SV'])])
        if self.Normal:
            vocabsize = vocabsize + len(self.vocab_mutation.loc[self.vocab_mutation['typ']=='Normal'])

        self.vocab_size = vocabsize + 1
        #print(self.vocab_size)

        #pdb.set_trace()

        self.pd_position_vocab = pd.read_csv(self.cwd + 'extfile/dictChpos.csv',index_col=0)
        self.pd_ges_vocab = pd.read_csv(self.cwd + 'extfile/dictGES.csv',index_col=0)

        self.position_size = len(self.pd_position_vocab) + 1
        self.ges_size = len(self.pd_ges_vocab) + 1
        
        self.rt_size =  1

        self.midstring = '.' + self.dataset_name + str(mutratio) + str(int(self.addtriplettoken)) + str(int(self.addpostoken)) +  str(int(self.addgestoken)) +  str(int(self.addrt)) + '/' 
        
        if self.mode == 'validation' or self.mode == 'testing':
            if self.crossdata:
                os.makedirs(self.crossdatadir + self.midstring, exist_ok=True)
                self.data_dir = self.crossdatadir
                #pdb.set_trace()
                
            else:
                os.makedirs(self.data_dir + self.midstring, exist_ok=True)

    def load_classinfo(self):
        if self.dataset_name == 'pcawg':
            pd_data = pd.read_csv(self.cwd + 'dataset_utils/classinfo_pcawg.csv',index_col = 0)
            self.pd_class_info = pd.DataFrame(pd_data)
        elif self.dataset_name == 'wgsgx':
            pd_data = pd.read_csv(self.cwd + 'dataset_utils/classinfo_wgsgx.csv',index_col = 0)
            self.pd_class_info = pd.DataFrame(pd_data)
        else:
            num_class = os.listdir(self.data_dir)
            name_class = [i for i in num_class if len(i.split('.'))==1]
            name_class = sorted(name_class)
            n_samples = []
            for idx,nm_class in enumerate(name_class):
                samples = os.listdir(self.data_dir+nm_class)
                samples = [x for x in samples if x[:10]=='count_new_']
                n_samples.append(len(samples))
            data = list(zip(name_class, np.arange(len(name_class)),n_samples))  
            self.pd_class_info = pd.DataFrame(data,columns=['class_name','class_index','n_samples'])

    def get_data(self,idx):
            
        if self.mode=='training':
            instances=self.training_fold.iloc[idx]     
        elif self.mode=='validation':
            instances=self.validation_fold.iloc[idx]
        elif self.mode == 'testing':
            instances=self.test_fold.iloc[idx]

        if self.newformat:
            samples = instances['samples'] + '.csv'
            target_name = instances['nm_class']

            if self.single_pred_vcf:
                pd_row = pd.read_csv(self.data_dir +'/count_' + samples,index_col=0).T
                row_count = pd_row.values[0]
            else:
                pd_row = pd.read_csv(self.data_dir + target_name +'/count_' + samples,index_col=0).T
                row_count = pd_row.values[0]
            
        else:
            target_name = instances['nm_class']
            samples = instances[1]
            row_count = instances[['NiSi','SNV','indel','SVMEI','Normal']].to_numpy()

        if self.mutratio is not None:
            avail_count = np.asarray(self.mutratio) * self.block_size      
            
            diff = avail_count - row_count
            pos = diff>0
            avail_count1 = row_count * pos
            diff = row_count > avail_count

            avail_count2 = avail_count * diff
            avail_count3 = avail_count1 + avail_count2
            shadowavail_count3 = avail_count3
            shadowavail_count3[0] = row_count[0]

            if sum(shadowavail_count3) > self.block_size:
                diff = self.block_size - sum(avail_count3) 
                shadowavail_count3[0] = diff + avail_count3[0]
                
            avail_count2 = shadowavail_count3.astype(int)

            if avail_count2[0]<0:
    
                secondmax = avail_count2[np.argmax(avail_count2)]
                avail_count2 = avail_count2 * 0.7

                avail_count = avail_count2

                diff = avail_count - row_count
                pos = diff>0
                avail_count1 = row_count * pos
                diff = row_count > avail_count

                avail_count2 = avail_count * diff
                avail_count3 = avail_count1 + avail_count2
                shadowavail_count3 = avail_count3
                shadowavail_count3[0] = row_count[0]

                if sum(shadowavail_count3) > self.block_size:
                    diff = self.block_size - sum(avail_count3) 
                    shadowavail_count3[0] = diff + avail_count3[0]
                    
                avail_count2 = shadowavail_count3.astype(int)

            avail_count = avail_count2

    
        def grab(pd_input,grabcol):
            return pd_input[grabcol]

        def allgrab(grabcol):
            

            if self.NiSi:
                #pdb.set_trace()
                if self.newformat:
                    pd_nisi = pd.read_csv(self.data_dir + target_name + '/' + 'SNV_' + samples,index_col=0)
                else:
                    pd_nisi = pd.read_csv(self.data_dir + target_name + '/' + 'NiSi_new_' + samples,index_col=0)
                pd_nisi = pd_nisi.sample(n = avail_count[0], replace = False)
                pd_nisi = grab(pd_nisi,grabcol)

                if self.SNV:
                    if self.newformat:
                        pd_SNV = pd.read_csv(self.data_dir + target_name + '/' + 'MNV_' + samples,index_col=0)
                    else:
                        pd_SNV = pd.read_csv(self.data_dir + target_name + '/' + 'SNV_new_' + samples,index_col=0)
                    pd_SNV = pd_SNV.sample(n = avail_count[1], replace = False)
                    pd_SNV = grab(pd_SNV,grabcol)
                    pd_nisi = pd_nisi.append(pd_SNV)

                    if self.indel:
                        pd_indel = pd.read_csv(self.data_dir + target_name + '/' + 'indel_' + samples,index_col=0)
                        pd_indel = pd_indel.sample(n = avail_count[2], replace = False)
                        pd_indel = grab(pd_indel,grabcol)
                        pd_nisi = pd_nisi.append(pd_indel)
                        
                        if self.SVMEI:
                            if self.newformat:
                                pd_meisv = pd.read_csv(self.data_dir + target_name + '/' + 'MEISV_' + samples,index_col=0)
                            else:
                                pd_meisv = pd.read_csv(self.data_dir + target_name + '/' + 'MEISV_new_' + samples,index_col=0)
                            pd_meisv = pd_meisv.sample(n = avail_count[3], replace = False)
                            pd_meisv = grab(pd_meisv,grabcol)
                            pd_nisi = pd_nisi.append(pd_meisv)

                            if self.Normal:
                                if self.newformat:
                                    pd_normal = pd.read_csv(self.data_dir + target_name + '/' + 'Neg_' + samples,index_col=0)
                                else:
                                    pd_normal = pd.read_csv(self.data_dir + target_name + '/' + 'Normal_new_' + samples,index_col=0)
                                pd_normal = pd_normal.sample(n = avail_count[4], replace = False)
                                pd_normal = grab(pd_normal,grabcol)
                                pd_nisi = pd_nisi.append(pd_normal) 

                pd_nisi = pd_nisi.fillna(0)
            return pd_nisi

        pd_nisi = pd.DataFrame()
        if self.addtriplettoken:
            if self.mode=='training' :
                pd_nisi = allgrab(['triplettoken'])
            else:
                filename = self.data_dir + self.midstring + 'val_' + samples
                if os.path.isfile(filename):
                    try:
                        pd_nisi = pd.read_csv(filename,index_col=0)
                    except:
                        pd_nisi = allgrab(['triplettoken'])
                        pd_nisi = pd_nisi.dropna()
                        pd_nisi.to_csv(filename)                        
                    
                else:
                    pd_nisi = allgrab(['triplettoken'])
                    pd_nisi.to_csv(filename)

        if self.addpostoken:
            if self.mode=='training' :
                pd_nisi = allgrab(['triplettoken','postoken'])
            else:
                #pdb.set_trace()
                filename = self.data_dir + self.midstring + 'val_' + samples
                if os.path.isfile(filename):
                    try:
                        pd_nisi = pd.read_csv(filename,index_col=0)
                    except:
                        pd_nisi = allgrab(['triplettoken','postoken'])
                        pdb.set_trace()
                        pd_nisi.to_csv(filename)
                else:
                    pd_nisi = allgrab(['triplettoken','postoken'])
                    pd_nisi.to_csv(filename)
      
        if self.addgestoken:
            if self.mode=='training' :
                pd_nisi = allgrab(['triplettoken','postoken','gestoken'])
            else:
                filename = self.data_dir + self.midstring + 'val_' + samples
                if os.path.isfile(filename):
                    try:
                        pd_nisi = pd.read_csv(filename,index_col=0)
                    except:
                        pd_nisi = allgrab(['triplettoken','postoken','gestoken'])
                        pd_nisi.to_csv(filename)

                else:
                    pd_nisi = allgrab(['triplettoken','postoken','gestoken'])
                    pd_nisi.to_csv(filename)

        if self.addrt:
            if self.mode=='training' :
                 pd_nisi = allgrab(['triplettoken','postoken','gestoken','rt'])
            else:
                filename = self.data_dir + self.midstring + 'val_' + samples
                if os.path.isfile(filename):
                    try:
                        pd_nisi = pd.read_csv(filename,index_col=0)
                    except:
                        pd_nisi = allgrab(['triplettoken','postoken','gestoken','rt'])
                        pd_nisi.to_csv(filename)

                else:
                    pd_nisi = allgrab(['triplettoken','postoken','gestoken','rt'])
                    pd_nisi.to_csv(filename)

        #pdb.set_trace()
        pd_nisi = pd_nisi.dropna()
        
        if self.nummut > 0:
            if self.nummut < len(pd_nisi):
                pd_nisi = pd_nisi.sample(n = self.nummut, replace = False)
            else:
                pd_nisi = pd_nisi.sample(n = len(pd_nisi), replace = False)
               
        #pdb.set_trace()

        if self.frac > 0:
            pd_nisi = pd_nisi.sample(frac = self.frac)

        if self.mode =='training':
            pd_nisi = pd_nisi.sample(frac = 1)

        #pdb.set_trace()

        np_triplettoken = pd_nisi.to_numpy()    

        is_padding = False
        if len(pd_nisi) < self.block_size:
            mins = self.block_size - len(np_triplettoken)
            is_padding = True
            
        datanumeric = []
        #pdb.set_trace()
        for i in pd_nisi.columns:
            np_data = pd_nisi[i].to_numpy() 
            if is_padding:
                np_data = np.copy(np.pad(np_data, ((0, mins)), mode='constant', constant_values=0))
            
            if i == 'rt':
                tensordata = torch.tensor(np.round(np_data, 1), dtype=torch.half)
                #tensordata = np.round(np_data, 3)

            if len(np_data) > self.block_size:
                np_data = np.asarray(np_data[:self.block_size],dtype=int)
                tensordata = torch.tensor(np_data, dtype=torch.long)
            else:
                np_data = np.asarray(np_data,dtype=int)
                tensordata = torch.tensor(np_data, dtype=torch.long)
            datanumeric.append(tensordata)
        
        datastring = samples

        if self.dataset_name=='wgsgx':
            #pdb.set_trace()
            gx_data = self.gx.loc[self.gx['samples']==samples[:-4]]
            gx_data = gx_data.iloc[:,:-2].values
            tensorgx_data = torch.tensor(gx_data, dtype=torch.float)

            datanumeric.append(tensorgx_data)

        #print(datanumeric)
        data=[datastring,datanumeric]
        #pdb.set_trace()

        if target_name != '':
            if self.crossdata:
                #pdb.set_trace()
                target = self.pd_class_infoto.loc[self.pd_class_infoto['class_name']==target_name].class_index.values[0]
            else:
                target = self.pd_class_info.loc[self.pd_class_info['class_name']==target_name].class_index.values[0]
            target = target.astype(np.int16)
            target = torch.tensor(target, dtype=torch.long)
        else:
            target = ''


        if self.adddatadir is not None:
            return data,[target,target_name]
        else:    
            return data,target

    def __len__(self):

        if self.mode=='training':
            return len(self.training_fold)
        elif self.mode=='validation':
            return len(self.validation_fold)
        elif self.mode=='testing':
            return len(self.test_fold)

    def __getitem__(self, idx):       

        data,target = self.get_data(idx)

        return data, target


if __name__ == '__main__':

    #dataloader = PCAWG(dataset_name = 'PCAWG', data_dir='/csc/epitkane/projects/PCAWG/shuffled_samples/', mode='training',portion = [8,1,1], folds=10, curr_fold=1,load=True,load_token=True)

    #dataloader = PCAWG(dataset_name = 'pcawg_mut3_comb0', data_dir='/csc/epitkane/projects/PCAWG20191001/data/modified_data/train/all24classes/', mode='training',portion = [8,1,1], folds=10, curr_fold=1,load=True,load_token=True,ncontext=3,addposition=False,filter=False,topk=5000)
    #dataloaderVal = PCAWG(dataset_name = 'pcawg_mut3_comb0', data_dir='/csc/epitkane/projects/PCAWG20191001/data/modified_data/train/all24classes/', mode='validation',portion = [8,1,1], folds=10, curr_fold=1,load=True,load_token=True,ncontext=3,addposition=False,filter=False,topk=5000)
    #/csc/epitkane/projects/tcga/new23classes/
    #/csc/epitkane/projects/PCAWG20191001/data/modified_data/train/new24classes/

    #G:/experiment/data/new24classes/
    '''
    dataloaderVal = FinalTCGAPCAWG(dataset_name = 'finalpcawg', 
                                data_dir='G:/experiment/data/new24classes/', 
                                mode='validation', 
                                curr_fold=1, 
                                block_size=5000, 
                                load=False,
                                mutratio = '0.3-0.3-0.3-0-0',
                                addtriplettoken=False,
                                addpostoken=False,
                                addgestoken=True,
                                addrt=False,
                                nummut = 0,
                                frac = 0,
                                adddatadir='G:/experiment/data/icgc/')

    #pdb.set_trace()
    data,target = dataloaderVal.__getitem__(0)
    pdb.set_trace()

    for k in range(0,len(dataloaderVal)):
        print(k)
        data,target = dataloaderVal.__getitem__(k)
    '''



    '''
    WGS GX
    '''

    #/scratch/project_2001668/data/pcawg

    dataloaderVal = TCGAPCAWG_Dataloader(dataset_name = 'wgsgx', 
                                        data_dir='/scratch/project_2001668/data/pcawg/allclasses/newformat/', 
                                        mode='training', 
                                        curr_fold=1, 
                                        block_size=5000, 
                                        load=False,
                                        addtriplettoken=True,
                                        addpostoken=False,
                                        addgestoken=False,
                                        addrt=False,
                                        nummut = 0,
                                        frac = 0,
                                        mutratio = '1-0-0-0-0',
                                        adddatadir = None,
                                        input_filename=None,
                                        args = None,
                                        gx_dir = '/scratch/project_2001668/data/pcawg/PCAWG_geneexp/')
    
    data,target = dataloaderVal.__getitem__(0)
    pdb.set_trace()

    '''
    fold = [1,2,3,4,5,6,7,8,9,10]
    mutratios = ['1-0-0-0-0','0.5-0.5-0-0-0','0.4-0.3-0.3-0-0','0.3-0.3-0.20-0.20-0','0.25-0.25-0.25-0.15-0.1']

    retrieve = ['addtriplettoken','addpostoken','addgestoken','addrt']

    for fo in fold:
        for i in retrieve:
            if i == 'addtriplettoken':
                addtriplettoken = True
            else:
                addtriplettoken = False
            
            if i == 'addpostoken':
                addpostoken = True
            else:
                addpostoken = False

            if i == 'addgestoken':
                addgestoken = True
            else:
                addgestoken = False

            if i == 'addrt':
                addrt = True
            else:
                addrt = False

            for j in mutratios:
                dataloaderVal = FinalTCGAPCAWG(dataset_name = 'finalpcawg', 
                                    data_dir='G:/experiment/data/new24classes/', 
                                    mode='validation', 
                                    curr_fold=1, 
                                    block_size=5000, 
                                    load=False,
                                    mutratio = j,
                                    addtriplettoken=addtriplettoken,
                                    addpostoken=addpostoken,
                                    addgestoken=addgestoken,
                                    addrt=addrt,
                                    nummut = 0,
                                    frac = 0)
                for k in range(0,len(dataloaderVal)):
                    print(str(fo) + ' ' + str(k) + ' ' + i + ' ' + j + ' ' + str(addtriplettoken) + str(addpostoken) + str(addgestoken) + str(addrt))
                    data,target = dataloaderVal.__getitem__(k)
    pdb.set_trace()

    dataloaderVal = TCGA(dataset_name = 'tcga_emb', data_dir='/csc/epitkane/projects/tcga/all23classes/', mode='validation',portion = [8,1,1], folds=10, curr_fold=1,load=True,load_token=True,ncontext=64,addposition=True,filter=True,block_size=300,withclass=True,twostream=False)

    for i in range(len(dataloaderVal)):
        data,target = dataloaderVal.__getitem__(i)

    dataloaderVal = TCGA(dataset_name = 'tcga_emb', data_dir='/csc/epitkane/projects/tcga/all23classes/', mode='testing',portion = [8,1,1], folds=10, curr_fold=1,load=True,load_token=True,ncontext=64,addposition=True,filter=True,block_size=300,loaddist=False,withclass=True,twostream=False)

    for i in range(len(dataloaderVal)):
        data,target = dataloaderVal.__getitem__(i)
    
    pdb.set_trace()
    '''

    