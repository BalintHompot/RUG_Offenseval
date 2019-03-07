
### this script is for training a naive model based on an offensive word list (see README). The script trains a model and outputs the model in the Models directory.
### The model can be SVM or LR, comment out the not-needed one. For generating output, see the prediction script.

import json
import pickle
import fileinput

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold, cross_validate
from joblib import dump, load
import helperFunctions
import transformers

offenseval_train = './Data/train/offenseval-training-v1.tsv'
offenseval_test = './Data/train/testset-taska.tsv'
offense_words = './Resources/offensive_words_eng.txt'
path_to_embs = './Resources/glove.twitter.27B.200d.txt'

TASK = 'binary'
#TASK = 'multi'

IDsTrain, Xtrain,Ytrain = helperFunctions.read_corpus(offenseval_train)
print('test data read')

# Minimal preprocessing / cleaning

Xtrain = helperFunctions.clean_samples(Xtrain)
offensiveWords = helperFunctions.load_offense_words(offense_words)

print("offensive words loaded")
embeddings, vocab = helperFunctions.load_embeddings(path_to_embs)
print("embeddings loaded")

transformer = transformers.naiveOffensiveWordSimilarity(embeddings, offensiveWords)

#lr = LogisticRegressionCV(max_iter=10000)
svc = LinearSVC(max_iter=10000)

vectors = transformer.fit_transform(Xtrain)
#print("score is")
#print(cross_validate(lr, vectors, Ytrain,cv=10))
#print(cross_validate(svc, vectors, Ytrain,cv=10))

#lr.fit(vectors, Ytrain)
svc.fit(vectors, Ytrain)
print("storing")
dump(svc, "./Models/NaiveSVC.joblib")
#dump(lr, "NaiveLR.joblib")



