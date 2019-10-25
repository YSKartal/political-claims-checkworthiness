
#%%
import logging as LOG
import pandas as pd
import numpy as np
import os
import glob
import csv
import nltk
from nltk import word_tokenize
from nltk.collocations import BigramCollocationFinder

#%%
year = '2019'
bigramMinDF = 10
limit = 20
#%%
LOG.basicConfig(level=LOG.INFO)

ext = ".tsv"            #2019 dosyalar tsv uzantılı
if(year == "2018"):         #2018 dosyalar tsv uzantılı
    ext = ".txt"

scoreColumns = ['idx', 'score']
goldColumns = ['linenumber', 'speaker', 'text', 'label']
#%%
scoreFiles = []
goldFiles = []
for file in glob.glob( "../test/" + year + "/vectors" + "/score" +  "/score_*_" +  str(bigramMinDF) + ext): scoreFiles.append(open(file,'r',encoding="utf8"))
for file in glob.glob( "../gold/" + year + "/*" + ext): goldFiles.append(file)

print(scoreFiles)

for file in scoreFiles:
    df = pd.read_csv(file,  delimiter = '\t', encoding = 'utf-8', names = scoreColumns)
    df.sort_values('score', ascending=False, inplace=True)
    #print(df['idx'])

    dfG = pd.read_csv(goldFiles[scoreFiles.index(file)],  delimiter = '\t', encoding = 'utf-8', names = goldColumns)
    LOG.info('{} dosyası için en yüksek {} puanlı claim:'.format((goldFiles[scoreFiles.index(file)]), limit))
    for i, idx in enumerate(df['idx']):
        if(i<limit):
            LOG.info(' {}: {} ~ {}'.format(idx,dfG['label'][idx-1],dfG['text'][idx-1]))


#%%
goldFiles = []
for file in glob.glob( "../gold/" + year + "/*" + ext): goldFiles.append(file)

for file in goldFiles:
    dfG = pd.read_csv(file,  delimiter = '\t', encoding = 'utf-8', names = goldColumns)
    LOG.info('{} dosyasındaki {} check worthy claims:'.format(file,limit))
    cntr = 0
    for idx, text in enumerate(dfG['text']):
        if(cntr<limit and dfG['label'][idx]==1):
            LOG.info('{}: {}'.format(idx,text))
            cntr+=1


#%%