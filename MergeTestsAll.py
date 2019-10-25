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
bigramMinDF = 6         #2, 3
##############################################################

#%%
######################## HAZIRLIKLAR #########################
LOG.basicConfig(level=LOG.INFO)
mode = "test"
ext = ".tsv"
if(year == "2018"):
    ext = ".txt"


columns = ['linenumber', 'speaker', 'text', 'label']
bigramFiles = []
posFiles = []
neFiles = []
catFiles = []
senFiles = []
goldFiles = []
cbFiles = []

globIdx = 0
newsFiles = []

for file in glob.glob( "../test/" + year + "/vectors" + "/news" +  "/*_1" + ext): newsFiles.append(open(file,'r',encoding="utf8"))

for file in glob.glob( "../test/" + year + "/vectors" + "/bigram_real" +  "/*" + str(bigramMinDF) + ".tsv"): bigramFiles.append(open(file,'r',encoding="utf8"))
for file in glob.glob( "../test/" + year + "/vectors" + "/pos" +  "/*" +  ".tsv"): posFiles.append(open(file,'r',encoding="utf8"))
for file in glob.glob( "../test/" + year + "/vectors" + "/ne" +  "/*" +  ".tsv"): neFiles.append(open(file,'r',encoding="utf8"))
for file in glob.glob( "../test/" + year + "/vectors" + "/cat" +  "/*" +  ".tsv"): catFiles.append(open(file,'r',encoding="utf8"))
for file in glob.glob( "../test/" + year + "/vectors" + "/sentiment" +  "/*" +  "x5.tsv"): senFiles.append(open(file,'r',encoding="utf8"))
for file in glob.glob( "../cb_pred/" + year + "/test/*" + ext): cbFiles.append(open(file,'r',encoding="utf8"))
for file in glob.glob( "../gold/" + year + "/*" + ext): goldFiles.append(file)


lineIdx=0
for idx in range(0,len(posFiles)):
    outFile = "../test/" + year + "/vectors/merged/" + "allmerged_test{}_".format(str(idx+1)) + str(bigramMinDF) + ext
    fout = open(outFile, 'w')
    newLine = ' '.join(bigramFiles[idx].readline().rsplit('\n'))+ ' '.join(newsFiles[idx].readline().rsplit('\n'))  + cbFiles[idx].readline().split()[1] + '\n'
    fout.write(newLine)
    while newLine:
        cbLine = cbFiles[idx].readline().split()
        cbScore =  cbLine[1]+'\n' if (len(cbLine)!=0) else ''

        newLine =  ' '.join(bigramFiles[idx].readline().rsplit('\n'))+ ' '.join(newsFiles[idx].readline().rsplit('\n'))  + cbScore
        fout.write(newLine)


'''

lineIdx=0
for idx in range(0,len(posFiles)):
    outFile = "../test/" + year + "/vectors/merged/" + "allmerged_test{}_".format(str(idx+1)) + str(bigramMinDF) + ext
    fout = open(outFile, 'w')
    newLine = ' '.join(bigramFiles[idx].readline().rsplit('\n'))  + ' '.join(posFiles[idx].readline().rsplit('\n')) + ' '.join(neFiles[idx].readline().rsplit('\n')) + ' '.join(catFiles[idx].readline().rsplit('\n')) + cbFiles[idx].readline().split()[1] + ' ' + senFiles[idx].readline()
    fout.write(newLine)
    while newLine:
        cbLine = cbFiles[idx].readline().split()
        cbScore =  cbLine[1]+' ' if (len(cbLine)!=0) else ''

        newLine =  ' '.join(bigramFiles[idx].readline().rsplit('\n'))  + ' '.join(posFiles[idx].readline().rsplit('\n')) + ' '.join(neFiles[idx].readline().rsplit('\n')) + ' '.join(catFiles[idx].readline().rsplit('\n')) + cbScore + senFiles[idx].readline()
        fout.write(newLine)

for vf in bigramFiles:
    idx = bigramFiles.index(vf)
    outFile = "../" + mode + "/" + year + "/vectors" + "/merged/" + "mlmerged_" + mode + "{:02d}".format(idx+1)+"_{:d}".format(bigramMinDF) + ext
    fout = open(outFile, "w")
    df = pd.read_csv(goldFiles[idx], delimiter = '\t', encoding = 'utf-8', names = columns)
    f = open(vf, "r", encoding="utf8")
    lineIdx=0
    
    for line in f:
        newL=[]
        #print(newL)
        features = line.split(" ")
        for i in range(0,len(features)):
            newL+=[features[i]]
        fout.write(' '.join(newL))
        lineIdx += 1
        globIdx = len(features)
    f.close()
    fout.close()
    LOG.info("{} bigram vektör ve {} gold test label eklendi ve {} dosyasına yazıldı.".format(vf,goldFiles[idx],outFile))

print(globIdx)
print(posFiles)
for vf in posFiles:
    idx = posFiles.index(vf)
    outFile = "../" + mode + "/" + year + "/vectors" + "/merged/" + "mlmerged_" + mode + "{:02d}".format(idx+1)+"_{:d}".format(bigramMinDF) + ext
    print(outFile)
    fout = open(outFile, "r")
    
    dan = fout.readlines()
    foutw = open(outFile, "w")

    f = open(vf, "r", encoding="utf8")
    lineIdx=0
    for line in f:
        newL=[]
        features = line.split(" ")
        for i in range(0,len(features)):
            newL+=[features[i]]
        dan[lineIdx] = ' '.join(dan[lineIdx].rsplit()) + ' ' + (' '.join(newL))
        lineIdx += 1

    foutw.writelines(dan)
        
    f.close()
    foutw.close()
    LOG.info("{} bigram vektör ve {} gold test label eklendi ve {} dosyasına yazıldı.".format(vf,goldFiles[idx],outFile))


for vf in neFiles:
    idx = neFiles.index(vf)
    outFile = "../" + mode + "/" + year + "/vectors" + "/merged/" + "mlmerged_" + mode + "{:02d}".format(idx+1)+"_{:d}".format(bigramMinDF) + ext
    print(outFile)
    fout = open(outFile, "r")
    
    dan = fout.readlines()
    foutw = open(outFile, "w")
    globIdx = len(dan[0].split()) - 2
    f = open(vf, "r", encoding="utf8")
    lineIdx=0
    for line in f:
        newL=[]
        features = line.split(" ")
        for i in range(0,len(features)):
            newL+=[features[i]]
        dan[lineIdx] = ' '.join(dan[lineIdx].rsplit()) + ' ' + (' '.join(newL))
        lineIdx += 1

    foutw.writelines(dan)
        
    f.close()
    foutw.close()
    LOG.info("{} bigram vektör ve {} gold test label eklendi ve {} dosyasına yazıldı.".format(vf,goldFiles[idx],outFile))

for vf in catFiles:
    idx = catFiles.index(vf)
    outFile = "../" + mode + "/" + year + "/vectors" + "/merged/" + "mlmerged_" + mode + "{:02d}".format(idx+1)+"_{:d}".format(bigramMinDF) + ext
    print(outFile)
    fout = open(outFile, "r")
    
    dan = fout.readlines()
    foutw = open(outFile, "w")
    globIdx = len(dan[0].split()) - 2
    f = open(vf, "r", encoding="utf8")
    lineIdx=0
    for line in f:
        newL=[]
        features = line.split(" ")
        for i in range(0,len(features)):
            newL+=[features[i]]
        dan[lineIdx] = ' '.join(dan[lineIdx].rsplit()) + ' ' + (' '.join(newL))
        lineIdx += 1

    foutw.writelines(dan)
    f.close()
    foutw.close()
    LOG.info("{} bigram vektör ve {} gold test label eklendi ve {} dosyasına yazıldı.".format(vf,goldFiles[idx],outFile))


for vf in senFiles:
    idx = senFiles.index(vf)
    outFile = "../" + mode + "/" + year + "/vectors" + "/merged/" + "mlmerged_" + mode + "{:02d}".format(idx+1)+"_{:d}".format(bigramMinDF) + ext
    print(outFile)
    fout = open(outFile, "r")
    
    dan = fout.readlines()
    foutw = open(outFile, "w")
    globIdx = len(dan[0].split()) - 2
    f = open(vf, "r", encoding="utf8")
    lineIdx=0
    for line in f:
        newL=[]
        features = line.split(" ")
        newL+=[features[0]]
        newL+=[features[1]]
        dan[lineIdx] = ' '.join(dan[lineIdx].rsplit()) + ' ' + (' '.join(newL)) 
        lineIdx += 1
        
    foutw.writelines(dan)
        
    f.close()
    foutw.close()
    LOG.info("{} bigram vektör ve {} gold test label eklendi ve {} dosyasına yazıldı.".format(vf,goldFiles[idx],outFile))

'''