import json
from stanfordcorenlp import StanfordCoreNLP
from textblob import TextBlob
import logging as LOG
import pandas as pd
import numpy as np
import os
import glob
import csv
import nltk
from nltk import word_tokenize
from nltk.collocations import BigramCollocationFinder
from sklearn.feature_extraction.text import CountVectorizer

#%%
######################## PARAMETRELER ########################
year = "2019"           #2019, 2018
mode = "train"          #test, train
windowSize = 5          
##############################################################

#%%
######################## HAZIRLIKLAR #########################

LOG.basicConfig(level=LOG.INFO)

ext = ".tsv"            #2019 dosyalar tsv uzantılı
if(year == "2018"):         #2018 dosyalar tsv uzantılı
    ext = ".txt"

columns = ['pol', 'sub']
inputFiles = []

for file in glob.glob( "../" + mode + "/" + year  + '/vectors/sentiment/*1' + ext ): inputFiles.append(file)
##############################################################

##################### SENTIMENT ÇIKARMA ######################
for f in inputFiles:
    df = pd.DataFrame()
    tmp = pd.read_csv(f, delimiter = ' ', encoding = 'utf-8', names = columns)
    df = df.append(tmp, ignore_index = True)
    pol=[]
    sub=[]
    lineIdx = 0
    for item in df['pol']:
        #beginInd = 0 if (lineIdx+1-windowSize < 0) else lineIdx+1-windowSize
        beginInd = 0 if (lineIdx-windowSize < 0) else lineIdx-windowSize
        if(lineIdx==0):
            pol+=[0]
        else:
            #pol+=[df['pol'][beginInd:lineIdx+1].mean()]
            pol+=[df['pol'][beginInd:lineIdx].mean()]
        lineIdx += 1
    df['pol']=pol

    lineIdx = 0
    for item in df['sub']:
        #beginInd = 0 if (lineIdx+1-windowSize < 0) else lineIdx+1-windowSize
        beginInd = 0 if (lineIdx-windowSize < 0) else lineIdx-windowSize
        '''if(lineIdx<50):
            print(str(df['sub'][beginInd:lineIdx]) + "**" + str(df['sub'][beginInd:lineIdx].mean()) + "**" + str(df['sub'][lineIdx].mean()))'''
        if(lineIdx==0):
            sub+=[0]
        else:
            #sub+=[df['sub'][beginInd:lineIdx+1].mean()]
            sub+=[df['sub'][beginInd:lineIdx].mean()]
        lineIdx += 1
    df['sub']=sub

    outFile = "../" + mode + "/" + year + '/vectors/sentiment/sentiment_' + mode + "{:02d}_x{}".format(inputFiles.index(f)+1, windowSize) + ext
    df.to_csv(outFile,header=False, columns=['pol','sub'], index_label=None,index=False,sep=' ', quoting=csv.QUOTE_NONE)
    LOG.info("{} dosyası polarity ve subjectivity analizi edilerek {} olarak kaydedildi.".format(f,outFile))

##############################################################
