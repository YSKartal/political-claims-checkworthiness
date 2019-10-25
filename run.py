#%%
import logging as LOG
import collections
import glob
import sys
import os
import pandas as pd
import csv

#%%
######################## PARAMETRELER ########################
year = "2019"           #2019, 2018
bigramMinDF = 10         #2, 3
##############################################################
LOG.basicConfig(level=LOG.INFO)
mode = "test"
ext = ".tsv"
if(year == "2018"):
    ext = ".txt"

columns = ['linenumber', 'speaker', 'text', 'label']

testFiles = []
for file in glob.glob( "../test/" + year + "/vectors/merged" + "/RankLib*" + str(bigramMinDF) + ext): testFiles.append(file)

trainFile = year + '/vectors/merged/' + 'RankLibmerged_train' + str(bigramMinDF) + ext

# leaf sayısını değiştir
os.system("java -Xmx1g -jar RankLib.jar \
        -train {} -gmax 1 -ranker 0 -tree 50 -leaf 2 -metric2t MAP \
        -save model.txt".format(trainFile))
for file in testFiles:
    print(file)
    idx = testFiles.index(file)
    scoreFile = "../test/" + year + "/vectors/score/" + "score_" + mode + "{:02d}".format(idx+1)+"_{:d}".format(bigramMinDF) + ext
    os.system("java -Xmx1g -jar RankLib.jar -load model.txt -rank {} -metric2T MAP -score {}".format(file, scoreFile))

#%%
scoreColumns = ['first', 'idx', 'score']
ofColumns = ['idx', 'score']
scoreFiles = []

for file in glob.glob( "../test/" + year + "/vectors/score" + "/s*" + str(bigramMinDF) + ext): scoreFiles.append(file)
print(scoreFiles)

newoffiles = []
for file in glob.glob( "../test/" + year + "/vectors/newof" + "/newof*" + ext): newoffiles.append(file)
print(newoffiles)


for file in scoreFiles:
    df = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8', names = scoreColumns)
    dfof = pd.read_csv(newoffiles[scoreFiles.index(file)], delimiter = '\t', encoding = 'utf-8', names = ofColumns)
    df['idx']+=1
    for index, row in df.iterrows():
        if(dfof['score'][index]>0.6):
            df['score'][index]=0.0
    print("../test/" + year + "/vectors/score" + "/wscore_test{:02d}_10")    
    df.to_csv( "../test/" + year + "/vectors/score" + "/wscore_test{:02d}_10".format(scoreFiles.index(file)+1)+ext,header=False, columns=[ 'idx', 'score'], index_label=None,index=False,sep='\t', quoting=csv.QUOTE_NONE)

#%%
