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
bigramMinDF = 5         #2, 3
##############################################################

#%%
######################## HAZIRLIKLAR #########################

LOG.basicConfig(level=LOG.INFO)

ext = ".tsv"            #2019 dosyalar tsv uzantılı
if(year == "2018"):         #2018 dosyalar tsv uzantılı
    ext = ".txt"

columns = ['linenumber', 'speaker', 'text', 'label']
bigramFiles = []
trainFiles = []
neFiles = []
catFiles = []
senFiles = []
posFiles = []

for file in glob.glob( "../train/" + year + "/vectors" + "/bigram_real" +  "/*" + str(bigramMinDF) + ext): bigramFiles.append(file)
for file in glob.glob( "../train/" + year + "/vectors" + "/pos" +  "/*" +  ".tsv"): posFiles.append(file)
for file in glob.glob( "../train/" + year + "/vectors" + "/ne" +  "/*" +  ".tsv"): neFiles.append(file)
for file in glob.glob( "../train/" + year + "/vectors" + "/sentiment" +  "/*" +  "x5.tsv"): senFiles.append(file)
for file in glob.glob( "../train/" + year + "/vectors" + "/cat" +  "/*" +  ".tsv"): catFiles.append(file)
for file in glob.glob( "../train/" + year + "/*" + ext): trainFiles.append(file)



readFile = open("../train/" + year + "/vectors/merged/allmerged_train" + str(bigramMinDF) + ext,'r',encoding="utf8")
outFile = "../train/" + year + "/vectors/merged/" + "RankLibmerged_train" + str(bigramMinDF) + ext
fout = open(outFile, 'w')

for line in readFile:
    features = line.split()
    newL=[features[0]]
    newL+=["qid=1"]
    for idx, feature in enumerate(features, start=0):
        if(idx>0):
            newL+=[str(idx) + ":" + feature]
    
    fout.write(' '.join(newL) + '\n')








'''
for line in readFile:
    features = line.split()
    newL=[features[0]]
    newL+=["qid=1"]
    for idx, feature in enumerate(features, start=0):
        if(idx>0):
            newL+=[str(idx) + ":" + feature]
    
    fout.write(' '.join(newL) + '\n')

globIdx = 0
fileStartIdx=[0]

outFile = "../train/" + year + "/vectors/merged/" + "xmerged_train" + str(bigramMinDF) + ext
fout = open(outFile, 'w')
fout = open(outFile, 'a')

for vf in bigramFiles:
    idx = bigramFiles.index(vf)
    df = pd.read_csv(trainFiles[idx], delimiter = '\t', encoding = 'utf-8', names = columns)
    #print(trainFiles[idx] + "-" + vf)
    
    f = open(vf, "r", encoding="utf8")
    
    lineIdx=0
    for line in f:
        newL=[]
        #print(df['label'][lineIdx])
        newL += str(df['label'][lineIdx])
        newL += ["qid:1"]
        features = line.split(" ")
        for i in range(0,len(features)):
            newL+=[str(i+1) + ":" + features[i]]
        fout.write(' '.join(newL))
        lineIdx += 1
        globIdx = len(features)
    f.close()
    fileStartIdx += [fileStartIdx[-1] + lineIdx]
    print(fileStartIdx)
    LOG.info("{} bigram vektör ve {} train label eklendi.".format(vf,trainFiles[idx]))
fout.close()

def matchFile(idx, idxList):
    for i in range(0, len(idxList)-1):
        if(idx >= idxList[i] and idx < idxList[i+1]):
            return i


lineIdx=0
fout = open(outFile, "r")
additionFiles = [open(posFiles[0], "r", encoding="utf8"),open(posFiles[1], "r", encoding="utf8"),open(posFiles[2], "r", encoding="utf8"),open(posFiles[3], "r", encoding="utf8"),open(posFiles[4], "r", encoding="utf8"),open(posFiles[5], "r", encoding="utf8"),open(posFiles[6], "r", encoding="utf8"),open(posFiles[7], "r", encoding="utf8"),open(posFiles[8], "r", encoding="utf8"),open(posFiles[9], "r", encoding="utf8"),open(posFiles[10], "r", encoding="utf8"),open(posFiles[11], "r", encoding="utf8"),open(posFiles[12], "r", encoding="utf8"),open(posFiles[13], "r", encoding="utf8"),open(posFiles[14], "r", encoding="utf8"),open(posFiles[15], "r", encoding="utf8"),open(posFiles[16], "r", encoding="utf8"),open(posFiles[17], "r", encoding="utf8"),open(posFiles[18], "r", encoding="utf8")]
foutw= open("../train/" + year + "/vectors/merged/" + "zmerged_train" + str(bigramMinDF) + ext, "w") 
for line in fout:
    fileIdx = matchFile(lineIdx,fileStartIdx)
    #print(str(lineIdx) + ' - ' + posFiles[fileIdx])
    #print(str(lineIdx) + "- " + str(fileIdx))
    for fi, afLine in enumerate(additionFiles[fileIdx]):
        #print(str(fi+fileStartIdx[fileIdx]) + " - " + str(lineIdx))
        if (fi+fileStartIdx[fileIdx]) == lineIdx:
            #print(fi+fileStartIdx[fileIdx])
            newL=[]
            features = afLine.split(" ")
            for i in range(0,len(features)):
                newL+=[str(globIdx + i+1) + ":" + features[i]]
            
    newLine= ' '.join(line.rsplit()) + ' ' + (' '.join(newL))
    foutw.write(newLine)
    additionFiles[fileIdx].seek(0)
    lineIdx+=1


lineIdx=0
fout = open("../train/" + year + "/vectors/merged/" + "zmerged_train" + str(bigramMinDF) + ext, "r")
additionFiles = [open(neFiles[0], "r", encoding="utf8"),open(neFiles[1], "r", encoding="utf8"),open(neFiles[2], "r", encoding="utf8"),open(neFiles[3], "r", encoding="utf8"),open(neFiles[4], "r", encoding="utf8"),open(neFiles[5], "r", encoding="utf8"),open(neFiles[6], "r", encoding="utf8"),open(neFiles[7], "r", encoding="utf8"),open(neFiles[8], "r", encoding="utf8"),open(neFiles[9], "r", encoding="utf8"),open(neFiles[10], "r", encoding="utf8"),open(neFiles[11], "r", encoding="utf8"),open(neFiles[12], "r", encoding="utf8"),open(neFiles[13], "r", encoding="utf8"),open(neFiles[14], "r", encoding="utf8"),open(neFiles[15], "r", encoding="utf8"),open(neFiles[16], "r", encoding="utf8"),open(neFiles[17], "r", encoding="utf8"),open(neFiles[18], "r", encoding="utf8")]
foutw= open("../train/" + year + "/vectors/merged/" + "tmerged_train" + str(bigramMinDF) + ext, "w") 
for line in fout:
    globIdx = len(line.split()) - 2
    fileIdx = matchFile(lineIdx,fileStartIdx)
    print(str(lineIdx) + ' - ' + str(globIdx))
    #print(str(lineIdx) + ' - ' + posFiles[fileIdx])
    #print(str(lineIdx) + "- " + str(fileIdx))
    for fi, afLine in enumerate(additionFiles[fileIdx]):
        #print(str(fi+fileStartIdx[fileIdx]) + " - " + str(lineIdx))
        if (fi+fileStartIdx[fileIdx]) == lineIdx:
            #print(fi+fileStartIdx[fileIdx])
            newL=[]
            features = afLine.split(" ")
            for i in range(0,len(features)):
                newL+=[str(globIdx + i+1) + ":" + features[i]]
            
    newLine= ' '.join(line.rsplit()) + ' ' + (' '.join(newL))
    foutw.write(newLine)
    additionFiles[fileIdx].seek(0)
    lineIdx+=1
    
lineIdx=0
fout = open("../train/" + year + "/vectors/merged/" + "tmerged_train" + str(bigramMinDF) + ext, "r")
additionFiles = [open(catFiles[0], "r", encoding="utf8"),open(catFiles[1], "r", encoding="utf8"),open(catFiles[2], "r", encoding="utf8"),open(catFiles[3], "r", encoding="utf8"),open(catFiles[4], "r", encoding="utf8"),open(catFiles[5], "r", encoding="utf8"),open(catFiles[6], "r", encoding="utf8"),open(catFiles[7], "r", encoding="utf8"),open(catFiles[8], "r", encoding="utf8"),open(catFiles[9], "r", encoding="utf8"),open(catFiles[10], "r", encoding="utf8"),open(catFiles[11], "r", encoding="utf8"),open(catFiles[12], "r", encoding="utf8"),open(catFiles[13], "r", encoding="utf8"),open(catFiles[14], "r", encoding="utf8"),open(catFiles[15], "r", encoding="utf8"),open(catFiles[16], "r", encoding="utf8"),open(catFiles[17], "r", encoding="utf8"),open(catFiles[18], "r", encoding="utf8")]
foutw= open("../train/" + year + "/vectors/merged/" + "ymerged_train" + str(bigramMinDF) + ext, "w") 
for line in fout:
    globIdx = len(line.split()) - 2
    fileIdx = matchFile(lineIdx,fileStartIdx)
    print(str(lineIdx) + ' - ' + str(globIdx))
    #print(str(lineIdx) + "- " + str(fileIdx))
    for fi, afLine in enumerate(additionFiles[fileIdx]):
        #print(str(fi+fileStartIdx[fileIdx]) + " - " + str(lineIdx))
        if (fi+fileStartIdx[fileIdx]) == lineIdx:
            #print(fi+fileStartIdx[fileIdx])
            newL=[]
            features = afLine.split(" ")
            for i in range(0,len(features)):
                newL+=[str(globIdx + i+1) + ":" + features[i]]
            
    newLine= ' '.join(line.rsplit()) + ' ' + (' '.join(newL))
    foutw.write(newLine)
    additionFiles[fileIdx].seek(0)
    lineIdx+=1


lineIdx=0
fout = open("../train/" + year + "/vectors/merged/" + "ymerged_train" + str(bigramMinDF) + ext, "r")
additionFiles = [open(senFiles[0], "r", encoding="utf8"),open(senFiles[1], "r", encoding="utf8"),open(senFiles[2], "r", encoding="utf8"),open(senFiles[3], "r", encoding="utf8"),open(senFiles[4], "r", encoding="utf8"),open(senFiles[5], "r", encoding="utf8"),open(senFiles[6], "r", encoding="utf8"),open(senFiles[7], "r", encoding="utf8"),open(senFiles[8], "r", encoding="utf8"),open(senFiles[9], "r", encoding="utf8"),open(senFiles[10], "r", encoding="utf8"),open(senFiles[11], "r", encoding="utf8"),open(senFiles[12], "r", encoding="utf8"),open(senFiles[13], "r", encoding="utf8"),open(senFiles[14], "r", encoding="utf8"),open(senFiles[15], "r", encoding="utf8"),open(senFiles[16], "r", encoding="utf8"),open(senFiles[17], "r", encoding="utf8"),open(senFiles[18], "r", encoding="utf8")]
foutw= open("../train/" + year + "/vectors/merged/" + "amerged_train" + str(bigramMinDF) + ext, "w") 
for line in fout:
    globIdx = len(line.split()) - 2
    fileIdx = matchFile(lineIdx,fileStartIdx)
    print(str(lineIdx) + ' - ' + str(globIdx))
    #print(str(lineIdx) + "- " + str(fileIdx))
    for fi, afLine in enumerate(additionFiles[fileIdx]):
        #print(str(fi+fileStartIdx[fileIdx]) + " - " + str(lineIdx))
        if (fi+fileStartIdx[fileIdx]) == lineIdx:
            #print(fi+fileStartIdx[fileIdx])
            newL=[]
            features = afLine.split(" ")
            newL+=[str(globIdx +1) + ":" + features[1]]
            
    newLine= ' '.join(line.rsplit()) + ' ' + (' '.join(newL))
    foutw.write(newLine)
    additionFiles[fileIdx].seek(0)
    lineIdx+=1
'''