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
##############################################################

#%%
######################## HAZIRLIKLAR #########################

LOG.basicConfig(level=LOG.INFO)

ext = ".tsv"            #2019 dosyalar tsv uzantılı
if(year == "2018"):         #2018 dosyalar tsv uzantılı
    ext = ".txt"

columns = ['linenumber', 'speaker', 'text', 'label']
inputFiles = []

for file in glob.glob( "../" + mode + "/" + year  + "/*" + ext): inputFiles.append(file)
##############################################################

##################### SENTIMENT ÇIKARMA ######################
for f in inputFiles:
    df = pd.DataFrame()
    tmp = pd.read_csv(f, delimiter = '\t', encoding = 'utf-8', names = columns)
    df = df.append(tmp, ignore_index = True)
    pol=[]
    sub=[]
    for sentence in df['text']:
        testimonial = TextBlob(sentence)
        pol+=[testimonial.sentiment[0]]
        sub+=[testimonial.sentiment[1]]
    df['pol']=pol
    df['sub']=sub
    outFile = "../" + mode + "/" + year + '/vectors/sentiment/sentiment_' + mode + "{:02d}_1".format(inputFiles.index(f)+1) + ext
    df.to_csv(outFile,header=False, columns=['pol','sub'], index_label=None,index=False,sep=' ', quoting=csv.QUOTE_NONE)
    LOG.info("{} dosyası polarity ve subjectivity analizi edilerek {} olarak kaydedildi.".format(f,outFile))

##############################################################
