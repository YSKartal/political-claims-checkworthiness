#%%
from sklearn import linear_model
import pickle
import numpy as np
import pandas as pd
import logging as LOG
import numpy as np
import os
import gensim
import spacy as sp
from spacy.lang.en import English
from sklearn.metrics.pairwise import cosine_similarity
from statistics import mean
import heapq
import sklearn 
import glob
import csv
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# %%
def merge(year,mode):
    ext = ".tsv"            #2019 dosyalar tsv uzantılı
    if(year == "2018"):         #2018 dosyalar tsv uzantılı
        ext = ".txt"
    
    trainFiles = []
    bFiles= []
    for file in glob.glob( "../train/prop/"  + year +  "_"+mode+"*" + ext): bFiles.append(open(file,'r',encoding="utf8"))

    wFiles= []
    for file in glob.glob( "../train/prop/f/"  + year +  "_train*" + ext): wFiles.append(open(file,'r',encoding="utf8"))

    nFiles= []
    for file in glob.glob( "../train/prop/w/"  + year +  "_train*" + ext): nFiles.append(open(file,'r',encoding="utf8"))

    for file in glob.glob( "../train/" + year + "/*" + ext): trainFiles.append(open(file,'r',encoding="utf8"))

    e1Files= []
    c1Files= []
    f1Files= []
    i1Files= []
    p1Files= []
    i2Files= []
    t1Files= []
    b1Files= []
    for file in glob.glob( "../train/prop/bert"  + year +  "_train*" + ext): b1Files.append(open(file,'r',encoding="utf8"))
    for file in glob.glob( "../train/prop/em"  + year +  "_train*" + ext): e1Files.append(open(file,'r',encoding="utf8"))
    for file in glob.glob( "../train/prop/cs"  + year +  "_train*" + ext): c1Files.append(open(file,'r',encoding="utf8"))
    for file in glob.glob( "../train/prop/ft"  + year +  "_train*" + ext): f1Files.append(open(file,'r',encoding="utf8"))
    for file in glob.glob( "../train/prop/impV"  + year +  "_train*" + ext): i1Files.append(open(file,'r',encoding="utf8"))
    for file in glob.glob( "../train/prop/prevSEMV"  + year +  "_train*" + ext): p1Files.append(open(file,'r',encoding="utf8"))
    for file in glob.glob( "../train/prop/isfz"  + year +  "_train*" + ext): i2Files.append(open(file,'r',encoding="utf8"))
    for file in glob.glob( "../train/" + year+"/vectors/tense/*" + ext): t1Files.append(open(file,'r',encoding="utf8"))


    print(trainFiles)
    outFile = "./data/"  + year + "_mergedtrain" + ext
    fout = open(outFile, 'w')

    lineIdx=0
    for idx in range(0,len(e1Files)):
        """print(trainFiles[idx])
        print(e1Files[idx])
        print(c1Files[idx])
        print(i1Files[idx])
        print(i2Files[idx])
        print(f1Files[idx])
        print(p1Files[idx])"""
        newLine = ' '.join(trainFiles[idx].readline().rsplit())[-1] + ' '  +' '.join(b1Files[idx].readline().rsplit('\n'))+' '.join(c1Files[idx].readline().rsplit('\n'))+' '.join(e1Files[idx].readline().rsplit('\n'))+' '.join(f1Files[idx].readline().rsplit('\n'))+' '.join(i1Files[idx].readline().rsplit('\n'))+' '.join(i2Files[idx].readline().rsplit('\n'))+ t1Files[idx].readline()
        fout.write(newLine)
        while newLine:
            #print(lineIdx)
            
            trainLine=trainFiles[idx].readline()
            newLine=' '.join(trainLine.rsplit())[-1]+' ' if (len(trainLine)!=0) else ''

            bLine=' '.join(b1Files[idx].readline().rsplit('\n'))+' '.join(c1Files[idx].readline().rsplit('\n'))+' '.join(e1Files[idx].readline().rsplit('\n'))+' '.join(f1Files[idx].readline().rsplit('\n'))+' '.join(i1Files[idx].readline().rsplit('\n'))+' '.join(i2Files[idx].readline().rsplit('\n'))+ t1Files[idx].readline()
            #newLine +=  ' '.join(bigramFiles[idx].readline().rsplit('\n')) + bLine
            newLine +=   bLine
            fout.write(newLine)
            lineIdx+=1
    print("{} etiketli train dosyası oluşturuldu.".format(outFile))

    e1Files= []
    c1Files= []
    f1Files= []
    i1Files= []
    p1Files= []
    i2Files= []
    t1Files= []
    b1Files= []
    for file in glob.glob( "../train/prop/bert"  + year +  "_test*" + ext): b1Files.append(open(file,'r',encoding="utf8"))
    for file in glob.glob( "../train/prop/em"  + year +  "_test*" + ext): e1Files.append(open(file,'r',encoding="utf8"))
    for file in glob.glob( "../train/prop/cs"  + year +  "_test*" + ext): c1Files.append(open(file,'r',encoding="utf8"))
    for file in glob.glob( "../train/prop/ft"  + year +  "_test*" + ext): f1Files.append(open(file,'r',encoding="utf8"))
    for file in glob.glob( "../train/prop/impV"  + year +  "_test*" + ext): i1Files.append(open(file,'r',encoding="utf8"))
    for file in glob.glob( "../train/prop/prevSEMV"  + year +  "_test*" + ext): p1Files.append(open(file,'r',encoding="utf8"))
    for file in glob.glob( "../train/prop/isfz"  + year +  "_test*" + ext): i2Files.append(open(file,'r',encoding="utf8"))
    for file in glob.glob( "../test/" + year+"/vectors/tense/*" + ext): t1Files.append(open(file,'r',encoding="utf8"))


    print(e1Files)
    print(t1Files)

    for idx in range(0,len(e1Files)):
        
        outFile = "./data/"  + year + "_mergedtest{}".format(str(idx+1)) + ext
        fout = open(outFile, 'w')
        newLine = ' '.join(b1Files[idx].readline().rsplit('\n'))+' '.join(c1Files[idx].readline().rsplit('\n'))+' '.join(e1Files[idx].readline().rsplit('\n'))+' '.join(f1Files[idx].readline().rsplit('\n'))+' '.join(i1Files[idx].readline().rsplit('\n'))+' '.join(i2Files[idx].readline().rsplit('\n'))+ t1Files[idx].readline()
        fout.write(newLine)
        while newLine:
            newLine =  ' '.join(b1Files[idx].readline().rsplit('\n'))+' '.join(c1Files[idx].readline().rsplit('\n'))+' '.join(e1Files[idx].readline().rsplit('\n'))+' '.join(f1Files[idx].readline().rsplit('\n'))+' '.join(i1Files[idx].readline().rsplit('\n'))+' '.join(i2Files[idx].readline().rsplit('\n'))+ t1Files[idx].readline()
            fout.write(newLine)
        print("{} test dosyası oluşturuldu.".format(outFile))




#%%
def clfModel(year, clf):
    ext = ".tsv"            #2019 dosyalar tsv uzantılı
    if(year == "2018"):         #2018 dosyalar tsv uzantılı
        ext = ".txt"
    x = pd.read_csv("./data/"+year+"_mergedtrain" + ext,sep=' ',header=None) 
    a = np.array(x)
    xTrain = a[:,1:]
    yTrain = a[:,0]
    clf.fit(xTrain, yTrain)  
    testFiles = []
    for file in glob.glob( "./data/"+year+"_mergedtest*"+ext): testFiles.append(file)
    print(testFiles)
    for file in testFiles:
        df = pd.DataFrame()
        xTest = pd.read_csv(file,sep=' ',header=None) 
        xTestArray = np.array(xTest)
        print(xTestArray.shape)
        predList = clf.predict_proba(xTestArray)
        dfList=[]
        for pred in predList:
            dfList+=[pred[1]]
        df['score']= dfList
        df['idx'] = df.index + 1 
        df.to_csv( "./data/score/" + year + "clf_score_test{:02d}".format(testFiles.index(file)+1) + ext,header=False, columns=[ 'idx', 'score'], index_label=None,index=False,sep='\t', quoting=csv.QUOTE_NONE)

#%%
def prepForRank(year,lines):
    ext = ".tsv"            #2019 dosyalar tsv uzantılı
    if(year == "2018"):         #2018 dosyalar tsv uzantılı
        ext = ".txt"
    x = pd.read_csv("./data/"+year+"_mergedtrain" + ext,sep=' ',header=None) 
    a = np.array(x)
    xTrain = a[:,1:]
    yTrain = a[:,0]

    arr = []
    numL = np.size(xTrain,0)
    ginx=0
    for row in xTrain:
        arri=[str(yTrain[(ginx)]),"qid:1"]
        idx=1
        for el in row:
            arri+=[str(idx)+":"+str(el)]
            idx+=1
        arr+=[arri]
        ginx+=1

    outFile = "./data/rank"  + year + "_mergedtrain" + ext
    fout = open(outFile, 'w')
    for line in arr:
        
        newLine = ' '.join(line)
        fout.write(newLine + '\n')
        
    print(fout)

    testFiles = []
    for file in glob.glob( "./data/"+year+"_mergedtest*"+ext): testFiles.append(file)
    print(testFiles)
    for file in testFiles:
        x = pd.read_csv(file,sep=' ',header=None) 
        a = np.array(x)
        xTrain = a[:,0:]

        arr = []
        ginx=0
        for row in xTrain:
            arri=["0","qid:1"]
            idx=1
            for el in row:
                arri+=[str(idx)+":"+str(el)]
                idx+=1
            arr+=[arri]
            ginx+=1

        outFile = "./data/rank" + year + "_mergedtest{:02d}".format(testFiles.index(file)+1) + ext
        fout = open(outFile, 'w')
        for line in arr:
            
            newLine = ' '.join(line)
            fout.write(newLine + '\n')
            
        print(fout)

#%%


def run_algo(year):
    ext = ".tsv"            #2019 dosyalar tsv uzantılı
    if(year == "2018"):         #2018 dosyalar tsv uzantılı
        ext = ".txt"
    testFiles = []
    for file in glob.glob( "./data/rank"+year+"_mergedtest*"+ext): testFiles.append(file)
    print(testFiles)
    train_file = "./data/rank"+year+"_mergedtrain"+ ext
    os.system("java -jar RankLib.jar \
            -train {} -gmax 1 -ranker 3 -tree 50 -leaf 2 -metric2t MAP \
            -save model.txt".format(train_file))
    for file in testFiles:
        print("***")
        score_file = "./data/score/AUNrank" + year + "_score_test{:02d}".format(testFiles.index(file)+1) + ext
        os.system("java -jar RankLib.jar -load model.txt -rank {} -metric2T MAP -score {}".format(file, score_file))
    return None

#%%
def tidy(year):
    ext = ".tsv"            #2019 dosyalar tsv uzantılı
    if(year == "2018"):         #2018 dosyalar tsv uzantılı
        ext = ".txt"
    testFiles = []
    for file in glob.glob( "./data/score/RUNrank"+year+"*"+ext): testFiles.append(file)
    print(testFiles)
    col=['1','2','score']
    for file in testFiles:
        df = pd.read_csv(file,sep='\t', names=col ,header=None) 
        df.index += 1
        score_file = "./data/score/Rrank" + year + "_score_test{:02d}".format(testFiles.index(file)+1) + ext
        df.to_csv(score_file,columns=['score'], header=False,sep='\t')


# %%
merge('2019','train')
merge('2018','train')

merge('2019','train')
merge('2018','train')


clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
clnf = LogisticRegression()

clfw = linear_model.SGDClassifier(loss='log')
clfw = svm.SVC(probability=True)
clf2 = linear_model.SGDClassifier(max_iter=1000, tol=1e-3,loss='modified_huber')
clfw = RandomForestClassifier(n_estimators=50, max_depth=5,random_state=0)

neigh = KNeighborsClassifier(n_neighbors=10)
#clf = RandomForestClassifier()
clfModel('2019',clf)
clfModel('2018',clf)

prepForRank('2018',"dede")
# %%
run_algo('2019')

# %%
tidy('2018')

# %%
