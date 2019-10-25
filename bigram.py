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
from sklearn.feature_extraction.text import CountVectorizer

#%%
######################## PARAMETRELER ########################
year = "2018"           #2019, 2018
mode = "train"          #test, train
bigramMinDF = 10         #2, 3
##############################################################

#%%
######################## HAZIRLIKLAR #########################

LOG.basicConfig(level=LOG.INFO)

ext = ".tsv"            #2019 dosyalar tsv uzantılı
if(year == "2018"):         #2018 dosyalar tsv uzantılı
    ext = ".txt"

LOG.info("{} {} için en çok {} satırda bulunan bigram kullanılarak vektör oluşturma başlıyor.".format(year,mode,bigramMinDF))

trainFiles = []         #train dosyaları bigram listesi oluşturmak için kullanılacak
inputFiles = []
for file in glob.glob("../train/" + year + "/pre" + "/*" + ext): trainFiles.append(file)
for file in glob.glob("../" + mode + "/" + year + "/pre" + "/*" + ext): inputFiles.append(file)

columns = ['linenumber', 'speaker', 'text', 'label']
dfF = pd.DataFrame()
dfN = pd.DataFrame()

for file in trainFiles:
    tmp = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8', names = columns)
    dfF = dfF.append(tmp, ignore_index = True)

LOG.info("Train dosyalarındaki toplam satır sayısı: {}".format(len(dfF)))

if(bigramMinDF == 2):           # en az 2 satırda(dokuman) var olan için min_df=0.000121795262164
    vectorizer = CountVectorizer(ngram_range=(2, 2),min_df=0.000121795262164,token_pattern=u"(?u)\\b\\w+\\b")   
elif(bigramMinDF == 3):         # en az 3 satırda(dokuman) var olan için min_df=0.000182692893246
    vectorizer = CountVectorizer(ngram_range=(2, 2),min_df=0.000182692893246,token_pattern=u"(?u)\\b\\w+\\b")   
elif(bigramMinDF == 4):         # en az 3 satırda(dokuman) var olan için min_df=0.000182692893246
    vectorizer = CountVectorizer(ngram_range=(2, 2),min_df=0.0002435905,token_pattern=u"(?u)\\b\\w+\\b")   
elif(bigramMinDF == 5):         # en az 3 satırda(dokuman) var olan için min_df=0.000182692893246
    vectorizer = CountVectorizer(ngram_range=(2, 2),min_df=0.0003044881,token_pattern=u"(?u)\\b\\w+\\b")   
elif(bigramMinDF == 6):         # en az 3 satırda(dokuman) var olan için min_df=0.000182692893246
    vectorizer = CountVectorizer(ngram_range=(2, 2),min_df=0.0003653857,token_pattern=u"(?u)\\b\\w+\\b")   
elif(bigramMinDF == 7):         # en az 3 satırda(dokuman) var olan için min_df=0.000182692893246
    vectorizer = CountVectorizer(ngram_range=(2, 2),min_df=0.0004262834,token_pattern=u"(?u)\\b\\w+\\b")   
elif(bigramMinDF == 8):         # en az 3 satırda(dokuman) var olan için min_df=0.000182692893246
    vectorizer = CountVectorizer(ngram_range=(2, 2),min_df=0.0004871810,token_pattern=u"(?u)\\b\\w+\\b")   
elif(bigramMinDF == 9):         # en az 3 satırda(dokuman) var olan için min_df=0.000182692893246
    vectorizer = CountVectorizer(ngram_range=(2, 2),min_df=0.0005480786,token_pattern=u"(?u)\\b\\w+\\b")   
elif(bigramMinDF == 10):         # en az 3 satırda(dokuman) var olan için min_df=0.000182692893246
    vectorizer = CountVectorizer(ngram_range=(2, 2),min_df=0.0006089763,token_pattern=u"(?u)\\b\\w+\\b")   
    

vectorizer.fit_transform(dfF['text'].values.astype('U'))
LOG.info("Kullanılacak bigram sayısı: {}".format(len(vectorizer.get_feature_names())))

bigramF = vectorizer.get_feature_names()            #kullanılacak bigram listesi

##############################################################

#%%
######### DOSYALARDAN BIGRAM VEKTORLERI OLUŞTUR ##############
def makeBigram(vectorizer,dfN,bigramF,idx):
    outfile = "../" + mode + "/" + year + "/vectors/bigram_real/bigram_" + mode +"{:02d}".format(idx+1) +"_{:d}".format(bigramMinDF) + ext            #bigram_train01_2
    
    a = np.zeros(shape=(len(dfN),len(bigramF)))
    for lineIdx in range(0,len(dfN.text)):
        #print((str(dfN.text[lineIdx]).split(" ")))
        #bigrm = list(nltk.bigrams(str(dfN.text[i]).split()))
        if(len(str(dfN.text[lineIdx]).split(" "))>=2):
            vectorizer.fit_transform([str(dfN.text[lineIdx])])
            bigrm = vectorizer.get_feature_names()
            for gram in bigrm:
                if(gram in bigramF):
                    a[lineIdx][bigramF.index(gram)]=1
    np.savetxt(outfile, a, fmt = '%f')
    LOG.info("{} dosyası için {} bigram vektör dosyası oluşturuldu.".format(inputFiles[idx],outfile))
    #print(a)
##############################################################


#%%
######################## MAIN ################################
if __name__ == "__main__":
    vectorizer = CountVectorizer(ngram_range=(2, 2),token_pattern=u"(?u)\\b\\w+\\b")
    for file in inputFiles:
        LOG.info("{} dosyası için çalışma başlıyor.".format(file))
        dfN = pd.DataFrame()
        tmp = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8', names = columns)
        dfN = dfN.append(tmp, ignore_index = True)
        makeBigram(vectorizer,dfN,bigramF,inputFiles.index(file))


##############################################################




#%%
######################## TEST KODLARI #########################
def test():
    corpus = [
        'This is the first a document.',
        'This document is the second document.',
        'And this is the third and this one.',
        'Is this the first document?',]
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    X = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names())
    print(X.toarray())  
    print(len(X.toarray()))  

#91004
#61062

##############################################################

#%%
