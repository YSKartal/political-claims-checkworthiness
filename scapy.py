#%%
import spacy
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

nlp = spacy.load('en_core_web_sm')

#%%
######################## PARAMETRELER ########################
year = "2019"           #2019, 2018
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

for file in glob.glob( "../train/" + year  + "/*" + ext): inputFiles.append(file)
print(inputFiles)
##############################################################

#%%
vocab=[]
for file in inputFiles:
    df = pd.DataFrame()
    tmp = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8', names = columns)
    df = df.append(tmp, ignore_index = True)
    newTx=[]
    print(str(inputFiles.index(file)))
    for idx,sentence in enumerate(df['text']):
        #print(str(inputFiles.index(file)) + ' ** '  +  str(idx))
        doc = nlp(sentence)
        for token in doc:
            if(token.dep_=='ROOT' and token.pos_=='VERB'):
                vocab+=[token.lemma_]
                #print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.is_stop)
        
    #df['text']=newTx
    #outFile = "../" + mode + "/" + year + '/pre/pre_' + mode + "{:02d}".format(inputFiles.index(file)+1) + ext
    #df.to_csv(outFile,header=False, index_label=None,index=False,sep='\t', quoting=csv.QUOTE_NONE)
    #LOG.info("{} dosyası preprocess edilerek {} olarak kaydedildi.".format(f,outFile))
print(len(vocab))
#%%
dvocab = list(set(vocab))
for i in range(0,100):

    print(dvocab[i])
print(len(dvocab))

#%%
trainFiles = []
testFiles = []
for file in glob.glob( "../train/" + year  + "/*" + ext): trainFiles.append(file)
for file in glob.glob( "../test/" + year  + "/*" + ext): testFiles.append(file)

for file in trainFiles:
    idx=trainFiles.index(file)
    outfile = "../train/" + year + "/vectors/verb/verb_train{:02d}".format(idx+1)  + ext            
    
    dfN = pd.DataFrame()
    tmp = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8', names = columns)
    dfN = dfN.append(tmp, ignore_index = True)

    a = np.zeros(shape=(len(dfN),len(dvocab)))
    for lineIdx in range(0,len(dfN.text)):
        if(len(str(dfN.text[lineIdx]).split(" "))>=2):
            doc = nlp(str(dfN.text[lineIdx]))
            for token in doc:
                if(token.lemma_ in dvocab and token.dep_=='ROOT'):
                    a[lineIdx][dvocab.index(token.lemma_)]=1
    np.savetxt(outfile, a, fmt = '%f')
    LOG.info("{} dosyası için {} verb vektör dosyası oluşturuldu.".format(trainFiles[idx],outfile))

for file in testFiles:
    idx=testFiles.index(file)
    outfile = "../test/" + year + "/vectors/verb/verb_test{:02d}".format(idx+1)  + ext            
    
    dfN = pd.DataFrame()
    tmp = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8', names = columns)
    dfN = dfN.append(tmp, ignore_index = True)

    a = np.zeros(shape=(len(dfN),len(dvocab)))
    for lineIdx in range(0,len(dfN.text)):
        if(len(str(dfN.text[lineIdx]).split(" "))>=2):
            doc = nlp(str(dfN.text[lineIdx]))
            for token in doc:
                if(token.lemma_ in dvocab and token.dep_=='ROOT'):
                    a[lineIdx][dvocab.index(token.lemma_)]=1
    np.savetxt(outfile, a, fmt = '%f')
    LOG.info("{} dosyası için {} verb vektör dosyası oluşturuldu.".format(testFiles[idx],outfile))


#%%
trainFiles = []
testFiles = []
for file in glob.glob( "../train/" + year  + "/*" + ext): trainFiles.append(file)
for file in glob.glob( "../test/" + year  + "/*" + ext): testFiles.append(file)

for file in trainFiles:
    idx=trainFiles.index(file)
    outfile = "../train/" + year + "/vectors/tense/tense_train{:02d}".format(idx+1)  + ext            
    
    dfN = pd.DataFrame()
    tmp = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8', names = columns)
    dfN = dfN.append(tmp, ignore_index = True)

    a = np.zeros(shape=(len(dfN),9))
    for lineIdx in range(0,len(dfN.text)):
        if(len(str(dfN.text[lineIdx]).split(" "))>=2):
            doc = nlp(str(dfN.text[lineIdx]))
            for token in doc:
                if(token.dep_=='ROOT'):
                    #print(token.tag_)
                    if(token.tag_=='VB'):
                        a[lineIdx][0]=1
                    elif(token.tag_=='VBD'):
                        a[lineIdx][1]=1
                    elif(token.tag_=='VBG'):
                        a[lineIdx][2]=1
                    elif(token.tag_=='VBN'):
                        a[lineIdx][3]=1
                    elif(token.tag_=='VBP'):
                        a[lineIdx][4]=1
                    elif(token.tag_=='VBZ'):
                        a[lineIdx][5]=1
                    elif(token.tag_=='MD'):
                        a[lineIdx][6]=1
                    elif(token.tag_=='HVS'):
                        a[lineIdx][7]=1
                    elif(token.tag_=='BES'):
                        a[lineIdx][8]=1
    np.savetxt(outfile, a, fmt = '%f')
    LOG.info("{} dosyası için {} tense vektör dosyası oluşturuldu.".format(trainFiles[idx],outfile))

for file in testFiles:
    idx=testFiles.index(file)
    outfile = "../test/" + year + "/vectors/tense/tense_test{:02d}".format(idx+1)  + ext            
    
    dfN = pd.DataFrame()
    tmp = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8', names = columns)
    dfN = dfN.append(tmp, ignore_index = True)

    a = np.zeros(shape=(len(dfN),9))
    for lineIdx in range(0,len(dfN.text)):
        if(len(str(dfN.text[lineIdx]).split(" "))>=2):
            doc = nlp(str(dfN.text[lineIdx]))
            for token in doc:
                if(token.dep_=='ROOT'):
                    if(token.tag_=='VB'):
                        a[lineIdx][0]=1
                    elif(token.tag_=='VBD'):
                        a[lineIdx][1]=1
                    elif(token.tag_=='VBG'):
                        a[lineIdx][2]=1
                    elif(token.tag_=='VBN'):
                        a[lineIdx][3]=1
                    elif(token.tag_=='VBP'):
                        a[lineIdx][4]=1
                    elif(token.tag_=='VBZ'):
                        a[lineIdx][5]=1
                    elif(token.tag_=='MD'):
                        a[lineIdx][6]=1
                    elif(token.tag_=='HVS'):
                        a[lineIdx][7]=1
                    elif(token.tag_=='BES'):
                        a[lineIdx][8]=1
    np.savetxt(outfile, a, fmt = '%f')
    LOG.info("{} dosyası için {} bigram tense dosyası oluşturuldu.".format(testFiles[idx],outfile))


#%%
doc = nlp(u"I think it's the worst deal I've ever seen negotiated.")

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)

#%%