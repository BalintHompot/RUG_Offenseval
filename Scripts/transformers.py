from sklearn.base import TransformerMixin
from nltk.tokenize import TweetTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from pandas import get_dummies
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from spacy import load as spacy_load
from joblib import load
nlp = spacy_load('en')

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

            if len(tweetVectorList) == 0:
                tweetVectorList.append([0] * len(self.embeddings['cat']))
            similarity = cosine_similarity((tweetVectorList), (self.offenseWordEmbeddings))

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


class EnsembleVectorizer(TransformerMixin):
    def __init__(self, pathList):
        self.models = [load(p) for p in pathList]
        print("Ensemble vectorizer loaded all models")
    
    def transform(self, X):
        ret = []
        
        for tweet in X:
            vector = []
            for m in self.models:
                pred = m.predict([tweet])[0]
                if pred == 'OFF':
                    vector.append(1)
                else:
                    vector.append(0)
            ret.append(vector)
        return ret