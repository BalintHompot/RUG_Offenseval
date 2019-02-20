
import json
import pickle
import gensim.models as gm
import re
import fileinput
from sklearn.base import TransformerMixin
from nltk.tokenize import TweetTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from pandas import get_dummies
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_validate
from joblib import dump, load


def read_corpus_otherSet(corpus_file, binary=True):
    '''Reading in data from corpus file'''
    with open(corpus_file, 'r', encoding='utf-8') as fi:
        fi = fi.readlines()
        tweets = []
        labels = []

        line = 0
        while line < (len(fi)):
            d = fi[line].strip().split(',')
            while d[-1] not in ['CAG', 'NAG', 'OAG']:
                line += 1
                dataPart = fi[line].strip().split(',')
                d += dataPart
            
            data = [d[0], "".join(d[1:len(d)-1]) ,d[len(d)-1]]

            # making sure no missing labels
            if len(data) != 3:
                raise IndexError('Missing data for tweet "%s"' % data[0])
            #print(data)
            tweets.append(data[1])
            if binary:
                if data[2] == 'NAG':
                    labels.append('NON')
                else:
                    labels.append('OFF')
            else:
                labels.append(data[2])
            line += 1
    #print(labels)
    print("read " + str(len(tweets)) + " tweets.")
    return tweets, labels

def read_corpus(corpus_file, binary=True):
    '''Reading in data from corpus file'''
    ids = []
    tweets = []
    labels = []
    with open(corpus_file, 'r', encoding='utf-8') as fi:
        for line in fi:
            data = line.strip().split('\t')
            # making sure no missing labels
            if len(data) != 5:
                raise IndexError('Missing data for tweet "%s"' % data[0])

            ids.append(data[0])


            tweets.append(data[1])
            labels.append(data[2])
    #print(ids[1:20])
    #print(tweets[1:20])
    #print(labels[1:20])
    return ids[1:], tweets[1:], labels[1:]

def load_embeddings(embedding_file):
    '''
    loading embeddings from file
    input: embeddings stored as txt
    output: embeddings in a dict-like structure available for look-up, vocab covered by the embeddings as a set
    '''

    print ("Loading Glove Model")
    f = open(embedding_file,'r')
    model = {}
    vocab = []
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
        vocab.append(word)
    print ("Done.",len(model)," words loaded!")
    return model, vocab


def clean_samples(samples):
    '''
    Simple cleaning: removing URLs, line breaks, abstracting away from user names etc.
    '''

    new_samples = []
    for tw in samples:

        tw = re.sub(r'@\S+','User', tw)
        tw = re.sub(r'\|LBR\|', '', tw)
        tw = re.sub(r'http\S+\s?', '', tw)
        tw = re.sub(r'\#', '', tw)
        new_samples.append(tw)

    return new_samples


def load_offense_words(path):
    ow = []
    f = open(path, "r") 
    for line in f:
        ow.append(line[:-1])
    return ow

class naiveOffensiveWordSimilarity(TransformerMixin):
    def __init__(self, embeddings, offenseWordList):
        self.embeddings = embeddings
        self.offenseWordList = offenseWordList
        self.offenseWordEmbeddings = []
        self.oneHotList = get_dummies(offenseWordList)
        #print(self.oneHotList)


    def transform(self, docs):
        fullList = []
        for tweet in docs:
            tweetSimilarity = []
            tweetVectorList = []
            for token in TweetTokenizer().tokenize(tweet):
                try:
                    tweetVectorList.append(self.embeddings[token])
                except KeyError:
                    pass

            #print("-----------")
            #print(len(tweetVectorList))
            #print(len(self.offenseWordEmbeddings))
            if len(tweetVectorList) == 0:
                tweetVectorList.append([0] * len(self.embeddings['cat']))
            similarity = cosine_similarity((tweetVectorList), (self.offenseWordEmbeddings))
            #print((similarity[0][0]))
            #tweetSimilarity += list(np.average(similarity, axis=0))
            tweetSimilarity += list(np.amax(similarity, axis=0))
            tweetSimilarity += list(np.amin(similarity, axis=0))
            #tweetSimilarity += list(np.median(similarity, axis=0))
            fullList.append(tweetSimilarity)
        return fullList

    def transform2(self, docs, *_):
        totalFreqList = []
        twt = TweetTokenizer()
        for text in docs:
            categoryTokenList = np.array([0]*len(self.offenseWordList))
            #oneHotTerms = get_dummies(self.getKMostImportantToken(docs))
            oneHotTerms = self.oneHotList
            for token in twt.tokenize(text):

                try:
                    currToken = np.array(list(oneHotTerms[token]))
                except KeyError:
                    currToken = np.array([0]*len(self.offenseWordList))
                categoryTokenList += currToken
            categoryTokenList = [int(e>0) for e in categoryTokenList]
            
            totalFreqList.append(list(categoryTokenList))
        return list(totalFreqList)

    def fit(self):
        for word in self.offenseWordList:
            try:
                self.offenseWordEmbeddings.append(self.embeddings[word])
            except KeyError:
                pass

    def fit_transform(self, docs):
        self.fit()
        ret = self.transform(docs)
        return ret

offenseval_train = '/Users/balinthompot/RUG/Honours/HateSpeech/offenseval-rug-master/Data/train/offenseval-training-v1.tsv'
offenseval_test = '/Users/balinthompot/RUG/Honours/HateSpeech/offenseval-rug-master/Data/train/agr_en_dev.csv'
offense_words = '/Users/balinthompot/RUG/Honours/HateSpeech/offenseval-rug-master/Resources/offensive_words_eng.txt'
path_to_embs = '/Users/balinthompot/RUG/Honours/HateSpeech/offenseval-rug-master/Resources/glove.twitter.27B.200d.txt'

TASK = 'binary'
#TASK = 'multi'
#print('Reading in offenseval training data...')

IDsTrain, Xtrain,Ytrain = read_corpus(offenseval_train)


#print('Reading in Test data...')
Xtest_raw = []
X_test_dev_labels = []
#Xtest_raw, Y_test_dev = read_corpus(offenseval_test)

#As we use cross validation, we can also use the provided test for training
#Xtrain = Xtrain + Xtest_raw
#Ytrain = Ytrain + Y_test_dev


# Minimal preprocessing / cleaning

Xtrain = clean_samples(Xtrain)
#Xtest = clean_samples(Xtest_raw)
offensiveWords = load_offense_words(offense_words)

print("offensive words loaded")
embeddings, vocab = load_embeddings(path_to_embs)

transformer = naiveOffensiveWordSimilarity(embeddings, offensiveWords)

#lr = LogisticRegressionCV(max_iter=10000)
svc = LinearSVC(max_iter=10000)

vectors = transformer.fit_transform(Xtrain)
print(Xtrain[0])
print(vectors[0])
print(len(vectors[0]))



#print("score is")
#print(cross_validate(lr, vectors, Ytrain,cv=10))
#print(cross_validate(svc, vectors, Ytrain,cv=10))

#lr.fit(vectors, Ytrain)
svc.fit(vectors, Ytrain)
print("storing")
dump(svc, "NaiveSVC.joblib")
#dump(lr, "NaiveLR.joblib")



