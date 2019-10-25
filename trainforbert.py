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
######################## PARAMETRELER ########################
year = "2019"           #2019, 2018
bigramMinDF = 10         #2, 3
##############################################################

#%%
######################## HAZIRLIKLAR #########################

LOG.basicConfig(level=LOG.INFO)

ext = ".tsv"            #2019 dosyalar tsv uzantılı
if(year == "2018"):         #2018 dosyalar tsv uzantılı
    ext = ".txt"

columns = ['linenumber', 'speaker', 'text', 'label']
trainFiles = []


for file in glob.glob( "../train/" + year + "/*" + ext): trainFiles.append(file)

fileStartIdx=[0]
print(trainFiles)

outFile = "../train/" + year + "/vectors/merged/" + "bertmerged_train" + str(bigramMinDF) + ".csv"
fout = open(outFile, 'w')

outw = "../train/" + year + "/vectors/worthy/"
outn = "../train/" + year + "/vectors/notworthy/"

idx = 0

df = pd.DataFrame()
for file in trainFiles:
    tmp = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8', names = columns)
    df = df.append(tmp, ignore_index = True)

for index, row in df.iterrows():
    if(row['label']=='1'):
        f= open(outw+str(index)+".txt","w+")
        f.write(row['text'])
    elif(row['label']=='0'):
        f= open(outn+str(index)+".txt","w+")
        f.write(row['text'])

df['alfa']='a'

df.to_csv(outFile, columns=['text', 'label'], index_label=None, sep=',')
LOG.info("{} etiketli train dosyası oluşturuldu.".format(outFile))


#%%
