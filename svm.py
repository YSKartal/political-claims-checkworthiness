#%%
import numpy as np 
import pandas as pd 
import sklearn 
import glob
from sklearn import svm

#%%
clf = svm.SVC(gamma='scale',probability=True)

#%%
x = pd.read_csv("../train/2019/vectors/merged/ml5merged_train3.tsv",sep=' ') 
a = np.array(x)

#%%
print(a.shape)

#%%
xTrain = a[:,1:]
yTrain = a[:,0]

#%%
clf.fit(xTrain, yTrain)  

#%%
testFiles = []
for file in glob.glob( "../test/2019/vectors" + "/merged" +  "/ml*.tsv"): testFiles.append(file)
print(testFiles)

#%%
for file in testFiles:
    xTest = pd.read_csv(file,sep=' ') 
    xTestArray = np.array(xTest)
    print(xTestArray.shape)
    predList = clf.predict_proba(xTestArray)
    print(predList)


#%%
