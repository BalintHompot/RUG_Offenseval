'''
SVM systems for offenseval
'''
import features
import helperFunctions
import transformers

import pickle
import stop_words
from sklearn.model_selection import KFold, cross_validate
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from joblib import dump
from sklearn.linear_model import LogisticRegressionCV



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
    offenseval_train = './Data/train/offenseval-training-v1.tsv'


    #print('Reading in offenseval training data...')
    if TASK == 'binary':
        IDsTrain, Xtrain,Ytrain = helperFunctions.read_corpus(offenseval_train)
    else:
        IDsTrain,Xtrain,Ytrain = helperFunctions.read_corpus(offenseval_train, binary=False)


    Xtrain = helperFunctions.clean_samples(Xtrain)


    '''
    Preparing vectorizer and classifier
    '''

    # Vectorizing data / Extracting features
    #print('Preparing tools (vectorizer, classifier) ...')
    if params["tweetTokenization"]:
        count_word = transformers.CountVectorizer(ngram_range=(1,2), stop_words=stop_words.get_stop_words('en'), tokenizer=TweetTokenizer().tokenize)
    else:
        count_word = transformers.CountVectorizer(ngram_range=(1,2), stop_words=stop_words.get_stop_words('en'))
    count_char = transformers.CountVectorizer(analyzer='char', ngram_range=(3,7))

    
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
        vectorizer.transformer_list.append(('posTagger', transformers.posTagExtractor(Xtrain, Ytrain)))

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
    dump(classifier, "./Models/RUG_Offense_concatModel.joblib")
    ##print("cross validating")
    ### predicting on set aside training data
    #print('Predicting on set aside data...')
    #Yguess = classifier.predict(XcustomTest)
    print("cross validating")
    result = cross_validate(classifier, Xtrain, Ytrain,cv=10)
    print(result)
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
path_to_embs = './Resources/glove.twitter.27B.200d.txt'
# path_to_embs = 'embeddings/model_reset_random.bin'
#print('Getting pretrained word embeddings from {}...'.format(path_to_embs))
embeddings, vocab = helperFunctions.load_embeddings(path_to_embs)
#print('Done')

print("starting")
#### setting components for the concatenated model
params = {
    "tweetTokenization" : True, 
    "POStagCheck": True,
    "oneHotTermFreqFiltering": True, 
    "charNgramFreqFiltering": True,
    "sentenceComplexityCheck": False,
    "embeddingsTermFreqFiltering": True, 
    
}
objects = {
    "liguisticFeatureExtractor": transformers.linguisticFeatureExtractor(),
    "embeddings": embeddings,
    "freqFilter": transformers.frequencyFilter(1500, embeddings, "word", 1, 1),
    "charFreqFilter" : transformers.frequencyFilter(3200, embeddings, "char", 3, 7)
}
print("results of original parameters:")
resAndModel = runFitting(params, objects)
dump(resAndModel, "./Models/SVM_concat.joblib")



