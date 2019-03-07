from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold, cross_validate
from sklearn.feature_extraction.text import CountVectorizer
import re


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

def clean_samples_list(samples):
    '''
    Simple cleaning: removing URLs, line breaks, abstracting away from user names etc.
    '''

    new_samples = []
    for tw in samples:

        tw = re.sub(r'@\S+','User', tw)
        tw = re.sub(r'\|LBR\|', '', tw)
        tw = re.sub(r'http\S+\s?', '', tw)
        tw = re.sub(r'\#', '', tw)
        new_samples.append([tw])

    return new_samples

trainPath = '/Users/balinthompot/RUG/Honours/HateSpeech/offenseval-rug-master/Data/train/offenseval-training-v1.tsv'
otherPath = '/Users/balinthompot/RUG/Honours/HateSpeech/offenseval-rug-master/Data/train/agr_en_train.csv'
ids, tweets, labels = read_corpus(trainPath)

tweets_string = clean_samples(tweets)
tweet_list = clean_samples_list(tweets)
nb = MultinomialNB()
vect = CountVectorizer()
vect.fit(tweets_string)
nb.fit(vect.transform(tweets_string), labels)
print("starting cross validation")
print(cross_validate(nb,vect.transform(tweets_string), labels, cv = 10))