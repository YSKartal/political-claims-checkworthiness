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
bigramFiles = []
trainFiles = []
neFiles = []
catFiles = []
senFiles = []
posFiles = []
cbFiles = []
newsFiles = []
verbFiles = []
tenseFiles = []

for file in glob.glob( "../train/" + year + "/vectors" + "/tense" +  "/*" + ext): tenseFiles.append(open(file,'r',encoding="utf8"))
for file in glob.glob( "../train/" + year + "/vectors" + "/verb" +  "/*" + ext): verbFiles.append(open(file,'r',encoding="utf8"))
for file in glob.glob( "../train/" + year + "/vectors" + "/news" +  "/*_immg" + ext): newsFiles.append(open(file,'r',encoding="utf8"))
for file in glob.glob( "../train/" + year + "/vectors" + "/bigram_real" +  "/*" + str(bigramMinDF) + ext): bigramFiles.append(open(file,'r',encoding="utf8"))
for file in glob.glob( "../train/" + year + "/vectors" + "/pos" +  "/*" +  ext): posFiles.append(open(file,'r',encoding="utf8"))
for file in glob.glob( "../train/" + year + "/vectors" + "/ne" +  "/*" + ext): neFiles.append(open(file,'r',encoding="utf8"))
for file in glob.glob( "../train/" + year + "/vectors" + "/sentiment" +  "/*" +  "x5" + ext): senFiles.append(open(file,'r',encoding="utf8"))
for file in glob.glob( "../train/" + year + "/vectors" + "/cat" +  "/*" + ext): catFiles.append(open(file,'r',encoding="utf8"))
for file in glob.glob( "../train/" + year + "/*" + ext): trainFiles.append(open(file,'r',encoding="utf8"))
for file in glob.glob( "../cb_pred/" + year + "/train/*" + ext): cbFiles.append(open(file,'r',encoding="utf8"))

fileStartIdx=[0]
print(newsFiles)
outFile = "../train/" + year + "/vectors/merged/" + "allmerged_train" + str(bigramMinDF) + ext
fout = open(outFile, 'w')

lineIdx=0
for idx in range(0,len(bigramFiles)):
    newLine = ' '.join(trainFiles[idx].readline().rsplit())[-1] + ' ' + ' '.join(bigramFiles[idx].readline().rsplit('\n')) + ' '.join(verbFiles[idx].readline().rsplit('\n'))  + ' '.join(tenseFiles[idx].readline().rsplit('\n'))+ ' '.join(newsFiles[idx].readline().rsplit('\n')) + cbFiles[idx].readline().split()[1] + '\n'
    fout.write(newLine)
    while newLine:
        #print(lineIdx)
        cbLine = cbFiles[idx].readline().split()
        cbScore =  cbLine[1] + '\n' if (len(cbLine)!=0) else ''
        
        trainLine=trainFiles[idx].readline()
        
        newLine=' '.join(trainLine.rsplit())[-1]+' ' if (len(trainLine)!=0) else ''
        newLine +=  ' '.join(bigramFiles[idx].readline().rsplit('\n')) + ' '.join(verbFiles[idx].readline().rsplit('\n')) + ' '.join(tenseFiles[idx].readline().rsplit('\n'))+ ' '.join(newsFiles[idx].readline().rsplit('\n')) + cbScore
        fout.write(newLine)
        lineIdx+=1
LOG.info("{} etiketli train dosyası oluşturuldu.".format(outFile))

### TEST DOSYALARI

bigramFiles = []
posFiles = []
neFiles = []
catFiles = []
senFiles = []
goldFiles = []
cbFiles = []

globIdx = 0
newsFiles = []
verbFiles = []
tenseFiles = []

for file in glob.glob( "../test/" + year + "/vectors" + "/tense" +  "/*" + ext): tenseFiles.append(open(file,'r',encoding="utf8"))
for file in glob.glob( "../test/" + year + "/vectors" + "/verb" +  "/*" + ext): verbFiles.append(open(file,'r',encoding="utf8"))
for file in glob.glob( "../test/" + year + "/vectors" + "/news" +  "/*_immg" + ext): newsFiles.append(open(file,'r',encoding="utf8"))
for file in glob.glob( "../test/" + year + "/vectors" + "/bigram_real" +  "/*" + str(bigramMinDF) + ext): bigramFiles.append(open(file,'r',encoding="utf8"))
for file in glob.glob( "../test/" + year + "/vectors" + "/pos" +  "/*" + ext): posFiles.append(open(file,'r',encoding="utf8"))
for file in glob.glob( "../test/" + year + "/vectors" + "/ne" +  "/*" + ext): neFiles.append(open(file,'r',encoding="utf8"))
for file in glob.glob( "../test/" + year + "/vectors" + "/cat" +  "/*" + ext): catFiles.append(open(file,'r',encoding="utf8"))
for file in glob.glob( "../test/" + year + "/vectors" + "/sentiment" +  "/*" +  "x5" + ext): senFiles.append(open(file,'r',encoding="utf8"))
for file in glob.glob( "../cb_pred/" + year + "/test/*" + ".tsv"): cbFiles.append(open(file,'r',encoding="utf8"))
for file in glob.glob( "../gold/" + year + "/*" + ext): goldFiles.append(file)


lineIdx=0
for idx in range(0,len(bigramFiles)):
    outFile = "../test/" + year + "/vectors/merged/" + "allmerged_test{}_".format(str(idx+1)) + str(bigramMinDF) + ext
    fout = open(outFile, 'w')
    newLine = ' '.join(bigramFiles[idx].readline().rsplit('\n'))+ ' '.join(verbFiles[idx].readline().rsplit('\n')) + ' '.join(tenseFiles[idx].readline().rsplit('\n')) + ' '.join(newsFiles[idx].readline().rsplit('\n')) + cbFiles[idx].readline().split()[1] + '\n'
    fout.write(newLine)
    while newLine:
        cbLine = cbFiles[idx].readline().split()
        cbScore =  cbLine[1]+'\n' if (len(cbLine)!=0) else ''

        newLine =  ' '.join(bigramFiles[idx].readline().rsplit('\n')) + ' '.join(verbFiles[idx].readline().rsplit('\n')) + ' '.join(tenseFiles[idx].readline().rsplit('\n')) + ' '.join(newsFiles[idx].readline().rsplit('\n')) + cbScore
        fout.write(newLine)
    LOG.info("{} test dosyası oluşturuldu.".format(outFile))

### TRAIN RANKLIB

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

LOG.info("{} RankLib train dosyası oluşturuldu.".format(outFile))

### TEST RANKLIB


mergedFiles = []
for file in glob.glob( "../test/" + year + "/vectors" + "/merged" +  "/all*" +  "_{}".format(str(bigramMinDF)) + ext): mergedFiles.append(file)


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
        
        fout.write(' '.join(newL) + '\n')

    LOG.info("{} RankLib test dosyası oluşturuldu.".format(outFile))


'''
for idx in range(0,len(posFiles)):
    newLine = ' '.join(trainFiles[idx].readline().rsplit())[-1] + ' ' + ' '.join(bigramFiles[idx].readline().rsplit('\n'))  + ' '.join(posFiles[idx].readline().rsplit('\n')) + ' '.join(neFiles[idx].readline().rsplit('\n')) + ' '.join(catFiles[idx].readline().rsplit('\n')) + cbFiles[idx].readline().split()[1] + ' ' + senFiles[idx].readline()
    fout.write(newLine)
    while newLine:
        #print(lineIdx)
        cbLine = cbFiles[idx].readline().split()
        cbScore =  cbLine[1]+' ' if (len(cbLine)!=0) else ''
        
        trainLine=trainFiles[idx].readline()
        
        newLine=' '.join(trainLine.rsplit())[-1]+' ' if (len(trainLine)!=0) else ''
        newLine +=  ' '.join(bigramFiles[idx].readline().rsplit('\n'))  + ' '.join(posFiles[idx].readline().rsplit('\n')) + ' '.join(neFiles[idx].readline().rsplit('\n')) + ' '.join(catFiles[idx].readline().rsplit('\n')) + cbScore + senFiles[idx].readline()
        fout.write(newLine)
        lineIdx+=1


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
        features = line.split(" ")
        for i in range(0,len(features)):
            newL+=[features[i]]
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
foutw= open("../train/" + year + "/vectors/merged/" + "ml2merged_train" + str(bigramMinDF) + ext, "w") 
for line in fout:
    fileIdx = matchFile(lineIdx,fileStartIdx)
    for fi, afLine in enumerate(additionFiles[fileIdx]):
        if (fi+fileStartIdx[fileIdx]) == lineIdx:
            newL=[]
            features = afLine.split(" ")
            for i in range(0,len(features)):
                newL+=[features[i]]
            
    newLine= ' '.join(line.rsplit('\n')) + ' ' + (' '.join(newL))
    foutw.write(newLine)
    additionFiles[fileIdx].seek(0)
    lineIdx+=1


lineIdx=0
fout = open("../train/" + year + "/vectors/merged/" + "ml2merged_train" + str(bigramMinDF) + ext, "r")
additionFiles = [open(neFiles[0], "r", encoding="utf8"),open(neFiles[1], "r", encoding="utf8"),open(neFiles[2], "r", encoding="utf8"),open(neFiles[3], "r", encoding="utf8"),open(neFiles[4], "r", encoding="utf8"),open(neFiles[5], "r", encoding="utf8"),open(neFiles[6], "r", encoding="utf8"),open(neFiles[7], "r", encoding="utf8"),open(neFiles[8], "r", encoding="utf8"),open(neFiles[9], "r", encoding="utf8"),open(neFiles[10], "r", encoding="utf8"),open(neFiles[11], "r", encoding="utf8"),open(neFiles[12], "r", encoding="utf8"),open(neFiles[13], "r", encoding="utf8"),open(neFiles[14], "r", encoding="utf8"),open(neFiles[15], "r", encoding="utf8"),open(neFiles[16], "r", encoding="utf8"),open(neFiles[17], "r", encoding="utf8"),open(neFiles[18], "r", encoding="utf8")]
foutw= open("../train/" + year + "/vectors/merged/" + "ml3merged_train" + str(bigramMinDF) + ext, "w") 
for line in fout:
    globIdx = len(line.split()) - 2
    fileIdx = matchFile(lineIdx,fileStartIdx)
    #print(str(lineIdx) + ' - ' + str(globIdx))
    for fi, afLine in enumerate(additionFiles[fileIdx]):
        if (fi+fileStartIdx[fileIdx]) == lineIdx:
            newL=[]
            features = afLine.split(" ")
            for i in range(0,len(features)):
                newL+=[features[i]]
            
    newLine= ' '.join(line.rsplit('\n')) + ' ' + (' '.join(newL))
    foutw.write(newLine)
    additionFiles[fileIdx].seek(0)
    lineIdx+=1
    
print(catFiles)
lineIdx=0
fout = open("../train/" + year + "/vectors/merged/" + "ml3merged_train" + str(bigramMinDF) + ext, "r")
additionFiles = [open(catFiles[0], "r", encoding="utf8"),open(catFiles[1], "r", encoding="utf8"),open(catFiles[2], "r", encoding="utf8"),open(catFiles[3], "r", encoding="utf8"),open(catFiles[4], "r", encoding="utf8"),open(catFiles[5], "r", encoding="utf8"),open(catFiles[6], "r", encoding="utf8"),open(catFiles[7], "r", encoding="utf8"),open(catFiles[8], "r", encoding="utf8"),open(catFiles[9], "r", encoding="utf8"),open(catFiles[10], "r", encoding="utf8"),open(catFiles[11], "r", encoding="utf8"),open(catFiles[12], "r", encoding="utf8"),open(catFiles[13], "r", encoding="utf8"),open(catFiles[14], "r", encoding="utf8"),open(catFiles[15], "r", encoding="utf8"),open(catFiles[16], "r", encoding="utf8"),open(catFiles[17], "r", encoding="utf8"),open(catFiles[18], "r", encoding="utf8")]
foutw= open("../train/" + year + "/vectors/merged/" + "ml4merged_train" + str(bigramMinDF) + ext, "w") 
for line in fout:
    globIdx = len(line.split()) - 2
    fileIdx = matchFile(lineIdx,fileStartIdx)
    #print(str(lineIdx) + ' - ' + str(globIdx))
    for fi, afLine in enumerate(additionFiles[fileIdx]):
        if (fi+fileStartIdx[fileIdx]) == lineIdx:
            newL=[]
            features = afLine.split(" ")
            for i in range(0,len(features)):
                newL+=[features[i]]
            
    newLine= ' '.join(line.rsplit('\n')) + ' ' + (' '.join(newL))
    foutw.write(newLine)
    additionFiles[fileIdx].seek(0)
    lineIdx+=1

print(senFiles)
lineIdx=0
fout = open("../train/" + year + "/vectors/merged/" + "ml4merged_train" + str(bigramMinDF) + ext, "r")
additionFiles = [open(senFiles[0], "r", encoding="utf8"),open(senFiles[1], "r", encoding="utf8"),open(senFiles[2], "r", encoding="utf8"),open(senFiles[3], "r", encoding="utf8"),open(senFiles[4], "r", encoding="utf8"),open(senFiles[5], "r", encoding="utf8"),open(senFiles[6], "r", encoding="utf8"),open(senFiles[7], "r", encoding="utf8"),open(senFiles[8], "r", encoding="utf8"),open(senFiles[9], "r", encoding="utf8"),open(senFiles[10], "r", encoding="utf8"),open(senFiles[11], "r", encoding="utf8"),open(senFiles[12], "r", encoding="utf8"),open(senFiles[13], "r", encoding="utf8"),open(senFiles[14], "r", encoding="utf8"),open(senFiles[15], "r", encoding="utf8"),open(senFiles[16], "r", encoding="utf8"),open(senFiles[17], "r", encoding="utf8"),open(senFiles[18], "r", encoding="utf8")]
foutw= open("../train/" + year + "/vectors/merged/" + "ml5merged_train" + str(bigramMinDF) + ext, "w") 
for line in fout:
    globIdx = len(line.split()) - 2
    fileIdx = matchFile(lineIdx,fileStartIdx)
    #print(str(lineIdx) + ' - ' + str(globIdx))
    #print(str(lineIdx) + "- " + str(fileIdx))
    for fi, afLine in enumerate(additionFiles[fileIdx]):
        #print(str(fi+fileStartIdx[fileIdx]) + " - " + str(lineIdx))
        if (fi+fileStartIdx[fileIdx]) == lineIdx:
            #print(fi+fileStartIdx[fileIdx])
            newL=[]
            features = afLine.split(" ")
            newL+=[features[0]]
            newL+=[features[1]]
            
    newLine= ' '.join(line.rsplit('\n')) + ' ' + (' '.join(newL))
    foutw.write(newLine)
    additionFiles[fileIdx].seek(0)
    lineIdx+=1
print("ok")'''

#%%
