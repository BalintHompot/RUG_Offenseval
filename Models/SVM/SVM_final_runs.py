'''
SVM systems for offenseval
'''
import argparse
import re
import statistics as stats
import stop_words
import json
import pickle
import gensim.models as gm


import features
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_validate
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
from itertools import combinations

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.tokenize import TweetTokenizer
from pandas import get_dummies
from joblib import dump

class posTagExtractor(TransformerMixin):
    def __init__(self, documents, labels):
        print("inititalizing POS tag extractor")
        self.taggedDocuments = self.transformToPosTags(documents)
        self.trainLabels = labels
        self.vectorizer = CountVectorizer(ngram_range=(1,3))

    def transformToPosTags(self, docs):
        for doc in docs:
            doc = [token.pos_ for token in nlp(doc)]
        return docs
    
    def transform(self, docs, *_):
        
        return self.vectorizer.transform(self.transformToPosTags(docs))

    def fit(self, *_):
        self.vectorizer.fit(self.taggedDocuments, self.trainLabels)
        print("posTagExtractor done fitting")
        return self


class frequencyFilter(TransformerMixin):
    def __init__(self, numberOfFeatures, embeds, analyzer, nGramMin, nGramMax):
        print("inititalizing freq filter")
        self.numberOfFeatures = numberOfFeatures
        self.oneHotList = {}
        self.separatedDocs = {}
        self.categoryEmbeddings = {}
        self.oneHotEnabled = False
        self.embeddingsEnabled = False
        self.embeddings = embeds
        self.analyzer = analyzer
        self.nGramMax = nGramMax
        self.nGramMin = nGramMin
        
    
    def getKMostImportantToken(self, docs):
        max_df = 0.01
        min_df = 0.0002
        k = self.numberOfFeatures    
        token_vect = TfidfVectorizer(analyzer=self.analyzer, ngram_range=(self.nGramMin, self.nGramMax), lowercase=False,
                                    tokenizer=TweetTokenizer().tokenize, min_df=min_df, max_df=max_df
                                    )
        
        tfidf = token_vect.fit_transform(docs)
        
        vocab = token_vect.vocabulary_
        inv_vocab = {index: word for word, index in vocab.items()}
            
        most_imp_ids = np.argsort(np.asarray(np.mean(tfidf, axis=0)).flatten())[::-1]
            
        most_imp = []
        for index in most_imp_ids:
            most_imp.append(inv_vocab[index])

        return most_imp[:k]

    def embedTransform(self, docs, *_):
        fullList = []
        for tweet in docs:
            tweetSimilarity = []
            tweetVectorList = []
            for token in TweetTokenizer().tokenize(tweet):
                try:
                    tweetVectorList.append(self.embeddings[token])
                except KeyError:
                    pass
            if len(tweetVectorList) == 0:
                tweetVectorList.append([0] * len(self.embeddings['cat']))
            for category, vectorList in self.categoryEmbeddings.items():
                similarity = cosine_similarity(np.asarray(tweetVectorList), np.asarray(vectorList))
                #tweetSimilarity += list(np.average(similarity, axis=0))
                #tweetSimilarity += list(np.median(similarity, axis=0))
                tweetSimilarity += list(np.amax(similarity, axis=0))
                tweetSimilarity += list(np.amin(similarity, axis=0))
            fullList.append(tweetSimilarity)
        return fullList


    def oneHotTransform(self, docs, *_):
        totalFreqList = []
        twt = TweetTokenizer()
        for text in docs:
            currentDocumentList = np.array([])
            for category in self.oneHotList.keys():
                categoryTokenList = np.array([0]*self.numberOfFeatures)
                #oneHotTerms = get_dummies(self.getKMostImportantToken(docs))
                oneHotTerms = self.oneHotList[category]
                for token in twt.tokenize(text):

                    try:
                        currToken = np.array(list(oneHotTerms[token]))
                    except KeyError:
                        currToken = np.array([0]*self.numberOfFeatures)
                    categoryTokenList += currToken
                categoryTokenList = [int(e>0) for e in categoryTokenList]
                currentDocumentList = np.concatenate((currentDocumentList, categoryTokenList))
            totalFreqList.append(list(currentDocumentList))
        return list(totalFreqList)

    def transform(self, docs, *_):
        ret = [[] for i in range(0, len(docs))]
        if self.oneHotEnabled:
                ohVectors = (self.oneHotTransform(docs))
        if self.embeddingsEnabled:
                embVectors = (self.embedTransform(docs))
        for doc in range(0, len(docs)):
            if self.oneHotEnabled:
                ret[doc] += (ohVectors[doc])
            if self.embeddingsEnabled:
                ret[doc] += (embVectors[doc])
        return ret

    def fit(self, X, y, *_):
        for docIndex in range(0, len(X)):

            currentCategory = y[docIndex]
            try:
                self.separatedDocs[currentCategory].append(X[docIndex])
            except KeyError:
                self.separatedDocs[currentCategory] = [X[docIndex]]
        
        for category, docList in self.separatedDocs.items():
            mostImportantTokens = self.getKMostImportantToken(docList)
            for token in mostImportantTokens:
                try:
                    self.categoryEmbeddings[category].append(self.embeddings[token])
                except KeyError:
                    try:
                        self.categoryEmbeddings[category] = [(self.embeddings[token])]
                    except:
                        pass

            self.oneHotList[category] = get_dummies(mostImportantTokens)
        print("posTagExtractor done fitting with oneHot " + str(self.oneHotEnabled) + " and embeddings " + str(self.embeddingsEnabled))
        return self

    def fit_transform(self, X, y, *_):
        self.fit(X, y)
        return self.transform(X)

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
        
        s = StandardScaler()
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

class linguisticFeatureExtractor(TransformerMixin):
    def __init__(self):
        print("inititalizing linguistic feature extractor")
        self.POSlist = ['X', 'INTJ' , 'ADP', 'NOUN', 'AUX', 'PRON', 'ADV', 'PROPN', 'CONJ', 'VERB', 'PUNCT', 'SCONJ', 'ADJ', 'DET', 'NUM', 'PART', 'SPACE']
        self.oneHotPOS = get_dummies(self.POSlist)
        self.depList = ['nk', '', 'ams', 'adc', 'oa2', 'avc', 'cvc', 'ph', 'uc', 'sbp', 'dm', 'mo', 'og', 'nmc', 'vo', 'ac', 'ph', 'ROOT', 'sb', 'cd', 'cj', 'oc', 'punct', 'op', 'svp', 'ju', 're', 'rs' , 'app', 'cp', 'ag', 'pd', 'cm', 'cc', 'oa', 'pnc', 'mnr', 'ep', 'pg', 'da', 'rc', 'pm', 'ng', 'par']
        self.oneHotDep = get_dummies(self.depList)

        self.featureList = []
        
    def setFeatureList(self, list):
        self.featureList = list

    def transform(self, doc, *_):
        returnList = []
        for text in doc:
            posVector = np.array([0] * len(self.POSlist))
            posNGramVector = np.array([])
            sentVector = np.array([0.0])
            lemmaVector = np.array([])
            complexityVector = np.array([0, 0, 0])
            depVector = np.array([0] * len(self.depList))

            for token in nlp(text):
                ##print("checking token")
                ##print(token.text, token.pos_)
                if "posTag" in self.featureList:
                    try:
                        currlist = list(self.oneHotPOS[token.pos_])
                    except KeyError:
                        print("pos key not found ")
                        print(token.pos_)
                        currlist = [0]*len(self.POSlist)
                    #print(currlist)
                    posVector += np.array(list(currlist))
                if "posTagNGrams" in self.featureList:
                    pass
                if "sentiment" in self.featureList:
                    sentVector += np.array([token.sentiment])
                if "sentenceComplexity" in self.featureList:
                    complexityVector += np.array([token.n_lefts, token.n_rights, len([t for t in token.subtree])])
                if "lemma" in self.featureList:
                    pass
                if "dep" in self.featureList:
                    try:
                        currlist = list(self.oneHotDep[token.dep_])
                    except KeyError:
                        print("dep key not found ")
                        print(token.dep_)
                        currlist = [0]*len(self.depList)
                    #print(currlist)
                    depVector += np.array(list(currlist))
            sentVector = np.divide(sentVector, len(text))
            complexityVector = np.divide(complexityVector, len(text))
            # print("pos list is ")
            # print(posVector)
            # print("sentiment is ")
            # print(sentVector)
            # print("complexity is ")
            # print(complexityVector)
            # print("appending ")
            toAppend = np.concatenate((posVector, posNGramVector, sentVector, lemmaVector, complexityVector, depVector))
            ##print(toAppend)
            returnList.append(toAppend)
        ##print("finished with document, returning")
        ##print(returnList)
        return returnList

    def fit(self, *_):
        print("lingfeatures done fitting with list " + str(self.featureList))
        return self

    def fit_transform(self, X, Y):
        return self.transform(X)

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

def evaluate(Ygold, Yguess):
    '''Evaluating model performance and printing out scores in readable way'''
    labs = sorted(set(Ygold + Yguess.tolist()))
    """ print('-'*50)
    print("Accuracy:", accuracy_score(Ygold, Yguess))
    print('-'*50)
    print("Precision, recall and F-score per class:") """

    # get all labels in sorted way
    # Ygold is a regular list while Yguess is a numpy array



    # printing out precision, recall, f-score for each class in easily readable way
    PRFS = precision_recall_fscore_support(Ygold, Yguess, labels=labs)
    """ print('{:10s} {:>10s} {:>10s} {:>10s}'.format("", "Precision", "Recall", "F-score"))
    for idx, label in enumerate(labs):
        print("{0:10s} {1:10f} {2:10f} {3:10f}".format(label, PRFS[0][idx],PRFS[1][idx],PRFS[2][idx]))

    print('-'*50)
    print("Average (macro) F-score:", stats.mean(PRFS[2]))
    print('-'*50)
    print('Confusion matrix:')
    print('Labels:', labs)
    print(confusion_matrix(Ygold, Yguess, labels=labs))
    print() """

    return [PRFS, labs]


def runFitting(params, objects):

    TASK = 'binary'
    #TASK = 'multi'

    '''
    Preparing data
    '''

    featureList = []

    if params["sentenceComplexityCheck"]:
        featureList.append("posTag")
    if params["embeddingsTermFreqFiltering"]:
        objects["freqFilter"].embeddingsEnabled = True
    if params["oneHotTermFreqFiltering"]:
        objects["freqFilter"].oneHotEnabled = True



    objects["liguisticFeatureExtractor"].setFeatureList(featureList)
    offenseval_train = '/Users/balinthompot/RUG/Honours/HateSpeech/offenseval-rug-master/Data/train/offenseval-training-v1.tsv'
    additional_train = '/Users/balinthompot/RUG/Honours/HateSpeech/offenseval-rug-master/Data/train/agr_en_train.csv'
    offenseval_test = '/Users/balinthompot/RUG/Honours/HateSpeech/offenseval-rug-master/Data/train/agr_en_dev.csv'
    additional_train_extra = '/Users/balinthompot/RUG/Honours/HateSpeech/offenseval-rug-master/Data/train/agr_en_dev.csv'

    #print('Reading in offenseval training data...')
    if TASK == 'binary':
        IDsTrain, Xtrain,Ytrain = read_corpus(offenseval_train)
    else:
        IDsTrain,Xtrain,Ytrain = read_corpus(offenseval_train, binary=False)

    #print('Reading in Test data...')
    #print('Reading in Test data...')
    #Xtest_raw = []
    #Xtest_raw, Y_test_dev = read_corpus(offenseval_test)
    additionalTrain, additionalLabels = read_corpus_otherSet(additional_train)
    additionalTrainExtra, additionalLabelsExtra = read_corpus_otherSet(additional_train_extra)
    Xtrain = Xtrain + additionalTrain + additionalTrainExtra
    Ytrain = Ytrain + additionalLabels + additionalLabelsExtra
    # Minimal preprocessing / cleaning

    Xtrain = clean_samples(Xtrain)
    #Xtest = clean_samples(Xtest_raw)

    #print(len(Xtrain), 'training samples!')
    #print(len(Xtest), 'test samples!')


    '''
    Preparing vectorizer and classifier
    '''

    # Vectorizing data / Extracting features
    #print('Preparing tools (vectorizer, classifier) ...')
    if params["tweetTokenization"]:
        count_word = CountVectorizer(ngram_range=(1,2), stop_words=stop_words.get_stop_words('de'), tokenizer=TweetTokenizer().tokenize)
    else:
        count_word = CountVectorizer(ngram_range=(1,2), stop_words=stop_words.get_stop_words('de'))
    count_char = CountVectorizer(analyzer='char', ngram_range=(3,7))

    

    #test = (count_word.fit_transform(Xtrain))
    #print(count_word.vocabulary_)

    # Getting embeddings
    
    embedder = features.Embeddings(objects["embeddings"], pool='max')

    vectorizer = FeatureUnion([('word', count_word),
                                ('char', count_char),
                                ('word_embeds', embedder )])
    
    if len(featureList) > 0:
        vectorizer.transformer_list.append(('lingFeats', objects["liguisticFeatureExtractor"]))

    if params["oneHotTermFreqFiltering"] or params["embeddingsTermFreqFiltering"]:
        vectorizer.transformer_list.append(('freqFilter', objects["freqFilter"]))

    if params["charNgramFreqFiltering"]:
        objects["charFreqFilter"].oneHotEnabled = True
        objects["charFreqFilter"].embeddingsEnabled = False
        vectorizer.transformer_list.append(('charfreqFilter', objects["charFreqFilter"]))

    if params["POStagCheck"]:
        vectorizer.transformer_list.append(('posTagger', posTagExtractor(Xtrain, Ytrain)))

    # Set up SVM classifier with unbalanced class weights
    """     if TASK == 'binary':
        # cl_weights_binary = None
        cl_weights_binary = {'OTHER':1, 'OFFENSE':10}
        clf = LinearSVC(class_weight=cl_weights_binary)
    else:
        # cl_weights_multi = None
        cl_weights_multi = {'OTHER':0.5,
                            'ABUSE':3,
                            'INSULT':3,
                            'PROFANITY':4}
        clf = LinearSVC(class_weight=cl_weights_multi) """
    clf = LinearSVC()
    #scaler = StandardScaler(with_mean=False)

    classifier = Pipeline([
                            ('vectorize', vectorizer),
                            #('scale', scaler),
                            ('classify', clf)])


    


    '''
    Actual training and predicting:
    '''

    print('Fitting on training data...')
    classifier.fit(Xtrain, Ytrain)
    print("storing")
    dump(classifier, "RUG_Offense_concatModel_additionalTraining.joblib")
    print("cross validating")
    ### predicting on set aside training data
    #print('Predicting on set aside data...')
    #Yguess = classifier.predict(XcustomTest)
    result = cross_validate(classifier, Xtrain, Ytrain,cv=10)
    #print(result)
    ########

    #print('Predicting...')
    #Yguess = classifier.predict(Xtest)


    """     '''
    Outputting in format required
    '''

    print('Outputting predictions...')

    outdir = '/Users/balinthompot/RUG/Honours/HateSpeech/offenseval-rug-master/Submission'
    fname = 'rug_fine_2.txt'

    with open(outdir + '/' + fname, 'w', encoding='utf-8') as fo:
        assert len(Yguess) == len(Xtest_raw), 'Unequal length between samples and predictions!'
        for idx in range(len(Yguess)):
            # print(Xtest_raw[idx] + '\t' + Yguess[idx] + '\t' + 'XXX', file=fo) # binary task (coarse)
            print(Xtest_raw[idx] + '\t' + 'XXX' + '\t' + Yguess[idx], file=fo) # multi task (fine)

    print('Done.')
    """
    return classifier


def mean(list):
    return sum(list)/len(list)


#######
from spacy import load as spacy_load
nlp = spacy_load('en')
path_to_embs = '/Users/balinthompot/RUG/Honours/HateSpeech/offenseval-rug-master/Resources/glove.twitter.27B.200d.txt'
# path_to_embs = 'embeddings/model_reset_random.bin'
#print('Getting pretrained word embeddings from {}...'.format(path_to_embs))
embeddings, vocab = load_embeddings(path_to_embs)
#print('Done')

print("starting")
params = {
    "tweetTokenization" : True, 
    "POStagCheck": True,
    "oneHotTermFreqFiltering": True, 
    "charNgramFreqFiltering": True,
    "sentenceComplexityCheck": False,
    "embeddingsTermFreqFiltering": True, 
    
}
objects = {
    "liguisticFeatureExtractor": linguisticFeatureExtractor(),
    "embeddings": embeddings,
    "freqFilter": frequencyFilter(1500, embeddings, "word", 1, 1),
    "charFreqFilter" : frequencyFilter(3200, embeddings, "char", 3, 7)
}
print("results of original parameters:")
resAndModel = runFitting(params, objects)



