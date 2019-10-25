#%%
import io
import numpy as np
import logging as LOG
import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
import os
import glob
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
from nltk.tokenize import RegexpTokenizer
import csv
import re
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.wrappers import FastText
import gensim.models

#%%
######################## PARAMETRELER ########################
year = "2019"           #2019, 2018
mode = "train"
##############################################################


#%%
ext = ".tsv"            #2019 dosyalar tsv uzantılı
if(year == "2018"):         #2018 dosyalar tsv uzantılı
    ext = ".txt"


tokenizer = RegexpTokenizer(r'\w+')         #tokenizer
stop = set(stopwords.words('english'))          #stop words
#%%
model = gensim.models.KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec/wiki-news-300d-1M.vec')


#%%
def get_mean_vector(word2vec_model, words):
    # remove out-of-vocabulary words
    words = [word for word in words if word in word2vec_model.vocab]
    if len(words) >= 1:
        #print(len(words))
        return np.mean(word2vec_model[words], axis=0)
    else:
        return np.zeros(shape=(1,300))
#%%
print(model.most_similar(['dioxide']))
print(max(1,5,7,2,3,9.9))


#%%
immgVec = get_mean_vector(model,['immigrants','borders','mexico','wall','immigration','illegal','refugees','workers','jobs','employees','workers','wealth','business','investments','taxes','economy','companies','industry','deal','trade','cash','unemployment','$','debts','finance','expenses','bank','prices','conditions','money','financial','works','black','lives','matter','racial','movement','white','taxes','african','folks'])
newsVecs = [get_mean_vector(model,['immigrants','borders','mexican','illegal', 'latino', 'hispanic']), \
    get_mean_vector(model, ['black', 'slavery', 'supremacist', 'racist', 'african']), \
        get_mean_vector(model, ['education', 'college', 'student', 'tuition', 'university']), \
            get_mean_vector(model, ['climate', 'warming', 'pollution', 'fossil', 'fuel', 'carbon', 'dioxide']), \
                get_mean_vector(model, ['gun', 'shooting', 'weapon']), \
                    get_mean_vector(model, ['hospital', 'insurance', 'health', 'prescription', 'pharmaceutical', 'clinic', 'Medicare', 'Affordable']), \
                        get_mean_vector(model, ['abortion', 'pregnancy', 'fetus',' pro-choice', 'pro-life']), \
                            get_mean_vector(model, ['gay', 'same-sex', 'lgbt', 'lesbian', 'homosexuality', 'transgender', 'marriage']), \
                                get_mean_vector(model, ['islam', 'muslim']), \
                                    get_mean_vector(model, ['terror', 'isis', 'radical', 'suicide', 'Al-Qaeda']), \
                                        get_mean_vector(model, ['iraq', 'baghdad', 'afghanistan', 'war', 'battle', 'military', 'soldier', 'kabul'])]
print(newsVecs)

'''
'immigrants','borders','mexico','wall','immigration','illegal','refugees','workers'])
#ecoVec = get_mean_vector(model,['jobs','employees','workers','wealth','business','investments','taxes','economy','companies','industry','deal','trade','cash','unemployment','$','debts','finance','expenses','bank','prices','conditions','money','financial','works'])
ecoVec = get_mean_vector(model,['black','lives','matter','racial','movement','white','taxes','african','folks'
mexican: 'immigrants','borders','mexican','illegal', 'latino', 'hispanic'
economy: 'workers', 'jobs','employees','wealth','business','investments','taxes','economy','companies','industry', 'unemployment', 'debts','finance','expenses', 'prices','money'
black: 'black', 'slavery', 'supremacist', 'racist', 'african'
education: 'education', 'college', 'student', 'tuition', 'university'
environment: 'climate', 'warming', 'pollution', 'fossil', 'fuel', 'carbon', 'dioxide'
Gun policy: 'gun', 'shooting', 'weapon'
health care: 'hospital', 'insurance', 'health', 'prescription', 'pharmaceutical', 'clinic', 'Medicare', 'Affordable' 
abortion: abortion, pregnancy, fetus, pro-choice, pro-life, 
lgbt: gay, same-sex, lgbt, lesbian, homosexuality, transgender, marriage
muslim: islam, muslim
terror: terror, isis, radical, suicide, Al-Qaeda  
war: iraq, baghdad, afghanistan, war, battle, military, soldier, kabul, 
'''

#%%
inputFiles = []
for file in glob.glob( "../" +mode + "/" + year  + "/*" + ext): inputFiles.append(file)
print(inputFiles)

columns = ['linenumber', 'speaker', 'text', 'label']
ilk=0
for file in inputFiles:
    df = pd.DataFrame()
    tmp = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8', names = columns)
    df = df.append(tmp, ignore_index = True)
    vecList=[]
    ty = 0
    for line in df['text']:
        ilk+=1
        newSen=[]
        line = re.sub("\'", "", str(line)) # remove single quotes
        line = re.sub("\.", "", line) # remove single quotes
        sL = tokenizer.tokenize(line.lower())            #decapitilize

        for i in sL:
            if i not in stop:           #stop words
                newSen+=[i]       

        #print(newSen)
        simList = []
        vec = get_mean_vector(model,newSen)
        for news in newsVecs:
            senList=[]
            for sen in newSen:
                if(sen in model.vocab):
                    senList += [cosine_similarity(news.reshape(-1, 300),model[sen].reshape(-1, 300))[0][0]]
                #print( cosine_similarity(news.reshape(-1, 300),model['sen'].reshape(-1, 300))[0][0])
            if(ilk<20):
                print(newSen)
                print(senList)
            if(len(senList)!=0):
                simList+=max(senList)
            else:
                simList+=0

        #print(sim)
        vecList+=[' '.join(simList)]
        #print(vec)
    df['cos']=vecList
    outFile = "../" + mode + "/" + year + '/vectors/news/news_' + mode + "{:02d}".format(inputFiles.index(file)+1) + '_{}'.format('immg') + ext
    df.to_csv(outFile,header=False, columns=['cos'],index_label=None,index=False,sep='\t', quoting=csv.QUOTE_NONE)

    print(len(vecList))

#%%
print(cosine_similarity(newsVecs[0].reshape(-1, 300),model['good'].reshape(-1, 300)))
print(cosine_similarity(newsVecs[0].reshape(-1, 300),model['evening'].reshape(-1, 300)))

#%%
inputFiles = []
for file in glob.glob( "../" +mode + "/" + year  + "/*" + ext): inputFiles.append(file)
print(inputFiles)

columns = ['linenumber', 'speaker', 'text', 'label']
for file in inputFiles:
    df = pd.DataFrame()
    tmp = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8', names = columns)
    df = df.append(tmp, ignore_index = True)
    vecList=[]
    ty = 0
    for line in df['text']:
        newSen=[]
        line = re.sub("\'", "", str(line)) # remove single quotes
        line = re.sub("\.", "", line) # remove single quotes
        sL = tokenizer.tokenize(line.lower())            #decapitilize
        for i in sL:
            if i not in stop:           #stop words
                newSen+=[i]        
        #print(newSen)
        simList = []
        vec = get_mean_vector(model,newSen)
        for news in newsVecs:
            sim = cosine_similarity(news.reshape(-1, 300),vec.reshape(-1, 300))
            simList += [str(sim[0][0])]
        #print(sim)
        vecList+=[' '.join(simList)]
        #print(vec)
    df['cos']=vecList
    outFile = "../" + mode + "/" + year + '/vectors/news/news_' + mode + "{:02d}".format(inputFiles.index(file)+1) + '_{}'.format('immg') + ext
    df.to_csv(outFile,header=False, columns=['cos'],index_label=None,index=False,sep='\t', quoting=csv.QUOTE_NONE)

    print(len(vecList))

#%%
'''
immgFiles = []
for file in glob.glob( "2016_election_news/Immigration/*.txt"): immgFiles.append(file)
terFiles=[]
for file in glob.glob( "2016_election_news/Terrorism/*.txt"): terFiles.append(file)

print(immgFiles)

#%%
def get_mean_vector(word2vec_model, words):
    # remove out-of-vocabulary words
    words = [word for word in words if word in word2vec_model.vocab]
    if len(words) >= 1:
        #print(len(words))
        return np.mean(word2vec_model[words], axis=0)
    else:
        return np.zeros(shape=(1,300))
#%%
immgVec = get_mean_vector(model,['immigrants','borders','mexico','wall','immigration','illegal','refugees','workers','jobs','employees','workers','wealth','business','investments','taxes','economy','companies','industry','deal','trade','cash','unemployment','$','debts','finance','expenses','bank','prices','conditions','money','financial','works','Obama','health','care','Obamacare','medical','drugs','insurance','diseases','cancer','clinics'])
ecoVec = get_mean_vector(model,['jobs','employees','workers','wealth','business','investments','taxes','economy','companies','industry','deal','trade','cash','unemployment','$','debts','finance','expenses','bank','prices','conditions','money','financial','works'])
blackVec = get_mean_vector(model,['black','lives','matter','racial','movement','white','taxes','african','folks'])
gunVec=['guns','assault','weapon','policy']
helVec = get_mean_vector(model,['Obama','health','care','Obamacare','medical','drugs','insurance','diseases','cancer','clinics'])

#%%
tokenizer = RegexpTokenizer(r'\w+')         #tokenizer
stop = set(stopwords.words('english'))          #stop words
#%%
inputFiles = []
for file in glob.glob( "../" +mode + "/" + year  + "/*" + ".tsv"): inputFiles.append(file)
print(inputFiles)

#%%
columns = ['linenumber', 'speaker', 'text', 'label']
for file in inputFiles:
    df = pd.DataFrame()
    tmp = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8', names = columns)
    df = df.append(tmp, ignore_index = True)
    vecList=[]
    ty = 0
    for line in df['text']:
        newSen=[]
        line = re.sub("\'", "", line) # remove single quotes
        line = re.sub("\.", "", line) # remove single quotes
        sL = tokenizer.tokenize(line.lower())            #decapitilize
        for i in sL:
            if i not in stop:           #stop words
                newSen+=[i]        
        #print(newSen)
        vec = get_mean_vector(model,newSen)
        sim = cosine_similarity(immgVec.reshape(-1, 300),vec.reshape(-1, 300))
        #print(sim)
        vecList+=[sim[0][0]]
        #print(vec)
    df['cos']=vecList
    outFile = "../" + mode + "/" + year + '/vectors/news/news_' + mode + "{:02d}".format(inputFiles.index(file)+1) + '_{}.tsv'.format('immg')
    df.to_csv(outFile,header=False, columns=['cos'],index_label=None,index=False,sep='\t', quoting=csv.QUOTE_NONE)

    print(len(vecList))
   
    



#%%
print(model['forest'])
print(immgVec.reshape(-1, 300))

#%%
print(model.most_similar(['job','employee','worker','wealth','business','investments','taxes','economy','companies','industry']))
#%%
model.most_similar(positive=['Erdogan', 'Egypt'], negative=['Turkey'], topn=5)

#%%
model = gensim.models.KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec')

#%%
analyser = SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))
#%%
sentiment_analyzer_scores("The phone is super cool.")

#%%
print(analyser.polarity_scores("The phone is super cool.")['compound'])

'''
#%%