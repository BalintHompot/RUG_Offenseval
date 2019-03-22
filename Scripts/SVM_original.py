'''
SVM systems for germeval, modified for English data from twitter and reddit, with crossvalidation and no test phase
original by Caselli et al: https://github.com/malvinanissim/germeval-rug
'''
#### PARAMS #############################################
source = 'Twitter'      ## options: Twitter, Reddit
dataSet = 'WaseemHovy'
offensiveRatio = 1/3
nonOffensiveRatio = 2/3

trainPath = './Data/train/WaseemHovy_Tweets_June2016_Dataset.csv'
path_to_embs = './Resources/glove.twitter.27B.200d.txt'
#########################################################

import helperFunctions
import transformers
import argparse
import re
import statistics as stats
import stop_words
import json
import pickle
import gensim.models as gm

import features
from sklearn.base import TransformerMixin
from nltk.tokenize import TweetTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_validate
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils import shuffle

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


if __name__ == '__main__':

    TASK = 'binary'
    #TASK = 'multi'

    '''
    Preparing data
    '''


    print('Reading in ' + source + ' training data...' + dataSet)
    if dataSet == 'WaseemHovy':
        if TASK == 'binary':
            IDsTrain,Xtrain,Ytrain = helperFunctions.read_corpus_WaseemHovy(trainPath)
        else:
            IDsTrain,Xtrain,Ytrain = helperFunctions.read_corpus_WaseemHovy(trainPath)
    else:
        exit()
        ## TODO: implement reading function for the Reddit data



    # Minimal preprocessing / cleaning
    Xtrain = helperFunctions.clean_samples(Xtrain)

    print(len(Xtrain), 'training samples!')
    '''
    Preparing vectorizer and classifier
    '''

    # Vectorizing data / Extracting features
    print('Preparing tools (vectorizer, classifier) ...')

    # unweighted word uni and bigrams
    ### This gives the stop_words may be inconsistent warning
    if source == 'Twitter':
        tokenizer = TweetTokenizer().tokenize
    else:
        tokenizer = None
        ### TODO: define tokenizer for Reddit data

    count_word = CountVectorizer(ngram_range=(1,2), stop_words=stop_words.get_stop_words('en'), tokenizer=tokenizer)
    count_char = CountVectorizer(analyzer='char', ngram_range=(3,7))

    # Getting embeddings
    
    # path_to_embs = 'embeddings/model_reset_random.bin'
    print('Getting pretrained word embeddings from {}...'.format(path_to_embs))
    embeddings, vocab = helperFunctions.load_embeddings(path_to_embs)
    print('Done')

    vectorizer = FeatureUnion([('word', count_word),
                                ('char', count_char),
                                ('word_embeds', features.Embeddings(embeddings, pool='max'))])


    # Set up SVM classifier with unbalanced class weights
    if TASK == 'binary':
        # cl_weights_binary = None
        cl_weights_binary = {'NOT':1/nonOffensiveRatio, 'OFF':1/offensiveRatio}
        clf = LinearSVC(class_weight=cl_weights_binary)
    else:
        # cl_weights_multi = None
        cl_weights_multi = {'OTHER':0.5,
                            'ABUSE':3,
                            'INSULT':3,
                            'PROFANITY':4}
        clf = LinearSVC(class_weight=cl_weights_multi)

    classifier = Pipeline([
                            ('vectorize', vectorizer),
                            ('classify', clf)])


    '''
    Actual training and predicting:
    '''

    print("10-fold cross validation results:")
    print(cross_validate(classifier, Xtrain, Ytrain,cv=10))
    print('Done.')






    #######
