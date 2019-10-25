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

globIdx = 0

for file in glob.glob( "../test/" + year + "/vectors" + "/bigram_real" +  "/*" + str(bigramMinDF) + ".tsv"): bigramFiles.append(file)
for file in glob.glob( "../test/" + year + "/vectors" + "/pos" +  "/*" +  ".tsv"): posFiles.append(file)
for file in glob.glob( "../test/" + year + "/vectors" + "/ne" +  "/*" +  ".tsv"): neFiles.append(file)
for file in glob.glob( "../test/" + year + "/vectors" + "/cat" +  "/*" +  ".tsv"): catFiles.append(file)
for file in glob.glob( "../test/" + year + "/vectors" + "/sentiment" +  "/*" +  "x5.tsv"): senFiles.append(file)
for file in glob.glob( "../gold/" + year + "/*" + ext): goldFiles.append(file)

mergedFiles = []
for file in glob.glob( "../test/" + year + "/vectors" + "/merged" +  "/all*" +  "_{}.tsv".format(str(bigramMinDF))): mergedFiles.append(file)

gid = 0
for file in mergedFiles:
    readFile = open(file,'r',encoding="utf8")
    outFile = "../test/" + year + "/vectors/merged/" + "RankLibmerged_test{}_".format(str(mergedFiles.index(file)+1)) + str(bigramMinDF) + ext
    fout = open(outFile, 'w')

    for line in readFile:
        #print(gid)
        features = line.split()
        newL=['0']
        newL+=["qid=1"]
        #print(len(features))
        for idx, feature in enumerate(features, start=1):
            newL+=[str(idx) + ":" + feature]
        gid+=1
        fout.write(' '.join(newL) + '\n')

'''for vf in bigramFiles:
    idx = bigramFiles.index(vf)
    outFile = "../" + mode + "/" + year + "/vectors" + "/merged/" + "xmerged_" + mode + "{:02d}".format(idx+1)+"_{:d}".format(bigramMinDF) + ext
    fout = open(outFile, "w")
    df = pd.read_csv(goldFiles[idx], delimiter = '\t', encoding = 'utf-8', names = columns)
    f = open(vf, "r", encoding="utf8")
    lineIdx=0
    
    for line in f:
        newL=[]
        newL += str(df['label'][lineIdx])
        newL += ["qid:1"]
        #print(newL)
        features = line.split(" ")
        for i in range(0,len(features)):
            newL+=[str(i+1) + ":" + features[i]]
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
    outFile = "../" + mode + "/" + year + "/vectors" + "/merged/" + "xmerged_" + mode + "{:02d}".format(idx+1)+"_{:d}".format(bigramMinDF) + ext
    print(outFile)
    fout = open(outFile, "r")
    
    dan = fout.readlines()
    foutw = open(outFile, "w")

    df = pd.read_csv(goldFiles[idx], delimiter = '\t', encoding = 'utf-8', names = columns)
    f = open(vf, "r", encoding="utf8")
    lineIdx=0
    for line in f:
        newL=[]
        features = line.split(" ")
        for i in range(0,len(features)):
            newL+=[str(globIdx + i+1) + ":" + features[i]]
        dan[lineIdx] = ' '.join(dan[lineIdx].rsplit()) + ' ' + (' '.join(newL))
        lineIdx += 1

    foutw.writelines(dan)
        
    f.close()
    foutw.close()
    LOG.info("{} bigram vektör ve {} gold test label eklendi ve {} dosyasına yazıldı.".format(vf,goldFiles[idx],outFile))


for vf in neFiles:
    idx = neFiles.index(vf)
    outFile = "../" + mode + "/" + year + "/vectors" + "/merged/" + "xmerged_" + mode + "{:02d}".format(idx+1)+"_{:d}".format(bigramMinDF) + ext
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
            newL+=[str(globIdx + i+1) + ":" + features[i]]
        dan[lineIdx] = ' '.join(dan[lineIdx].rsplit()) + ' ' + (' '.join(newL))
        lineIdx += 1

    foutw.writelines(dan)
        
    f.close()
    foutw.close()
    LOG.info("{} bigram vektör ve {} gold test label eklendi ve {} dosyasına yazıldı.".format(vf,goldFiles[idx],outFile))

for vf in catFiles:
    idx = catFiles.index(vf)
    outFile = "../" + mode + "/" + year + "/vectors" + "/merged/" + "xmerged_" + mode + "{:02d}".format(idx+1)+"_{:d}".format(bigramMinDF) + ext
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
            newL+=[str(globIdx + i+1) + ":" + features[i]]
        dan[lineIdx] = ' '.join(dan[lineIdx].rsplit()) + ' ' + (' '.join(newL))
        lineIdx += 1

    foutw.writelines(dan)
    f.close()
    foutw.close()
    LOG.info("{} bigram vektör ve {} gold test label eklendi ve {} dosyasına yazıldı.".format(vf,goldFiles[idx],outFile))


for vf in senFiles:
    idx = senFiles.index(vf)
    outFile = "../" + mode + "/" + year + "/vectors" + "/merged/" + "xmerged_" + mode + "{:02d}".format(idx+1)+"_{:d}".format(bigramMinDF) + ext
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
        newL+=[str(globIdx +1) + ":" + features[1]]
        dan[lineIdx] = ' '.join(dan[lineIdx].rsplit()) + ' ' + (' '.join(newL)) 
        lineIdx += 1
        
    foutw.writelines(dan)
        
    f.close()
    foutw.close()
    LOG.info("{} bigram vektör ve {} gold test label eklendi ve {} dosyasına yazıldı.".format(vf,goldFiles[idx],outFile))

'''