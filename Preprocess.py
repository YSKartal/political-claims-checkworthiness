#%%
import logging as LOG
import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
import os
import glob
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
from nltk.tokenize import RegexpTokenizer
import csv

#%%
######################## PARAMETRELER ########################
year = "2018"           #2019, 2018
mode = "train"          #test, train
##############################################################


#%%
######################## HAZIRLIKLAR #########################
LOG.basicConfig(level=LOG.INFO)

ext = ".tsv"
if(year == "2018"):
    ext = ".txt"

tokenizer = RegexpTokenizer(r'\w+')         #tokenizer
ps = PorterStemmer()            #stemmer
stop = set(stopwords.words('english'))          #stop words

columns = ['linenumber', 'speaker', 'text', 'label']
inputFiles = []

for file in glob.glob( "../" + mode + "/" + year  + "/*" + ext): inputFiles.append(file)
##############################################################

################### DOSYALARI ÖN İŞLEME ######################
for f in inputFiles:
    df = pd.DataFrame()
    tmp = pd.read_csv(f, delimiter = '\t', encoding = 'utf-8', names = columns)
    df = df.append(tmp, ignore_index = True)
    newTx=[]
    for sentence in df['text']:
        newSen=[]
        sL = tokenizer.tokenize(str(sentence).lower())            #decapitilize
        for i in sL:
            if i not in stop:           #stop words
                newSen+=[ps.stem(i)]            #stemmer
        newTx+=[" ".join(newSen)]
    df['text']=newTx
    outFile = "../" + mode + "/" + year + '/pre/pre_' + mode + "{:02d}".format(inputFiles.index(f)+1) + ext
    df.to_csv(outFile,header=False, index_label=None,index=False,sep='\t', quoting=csv.QUOTE_NONE)
    LOG.info("{} dosyası preprocess edilerek {} olarak kaydedildi.".format(f,outFile))

##############################################################

#%%



#%%
