import pandas as pd
import numpy as np
from os import listdir
from unidecode import unidecode
import nltk.data
from nltk.corpus import stopwords
from operator import itemgetter
import re
from nltk import bigrams
from nltk.util import ngrams
from sklearn.feature_extraction import DictVectorizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
cats = ['Rooms', 'Date', 'Location', 'Service', 'Business service', 'Author', 'Check in / front desk', 'No. Helpful', 'Cleanliness', 'Content', 'Value', 'No. Reader', 'Overall']
aspect = "Value"
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
seeds =  {"Value" : ["value", "price", "quality","worth"],
          "Rooms" : ["room", "suite", "view", "bed"],
          "Location" : ["location", "traffic", "minute", "restaurant"],
          "Cleanliness" : ["clean", "dirty", "maintain", "smell"],
          "Check in / front desk": ["stuff", "check", "help", "reservation"],
          "Service" : ["service", "food", "breakfast", "buffet"],
          "Business service" : ["business", "center", "computer", "internet"]
         }



def getBlankFrame():
    
    data = pd.DataFrame(columns=cats)
    
    return data


def addFileToData(filename, data):
    intColumns = ['No. Reader', 'No. Helpful', 'Cleanliness','Check in / front desk', 'Value', 'Overall', 'Service', 'Business service', 'Rooms', 'Location']
    characterThreshold = 60
    with open(filename, 'r') as content_file:
        content = content_file.read()
        
        #print(repr(content))
    if content.count("\r") > 0:
        reviews = content.split("\r\n\r\n")
    else:
        reviews = content.split("\n\n")
    
    for r in reviews:
        thisReview = pd.Series([None]*len(cats), cats)
        splt = r.split("\n")
        for s in splt:
            for c in cats:
                if "<"+c+">" in s:
                    value = s.replace('<'+c+'>', '')
                    if c in intColumns:
                        value = int(value)
                    if value == -1: #we dont want -1 as this is going to mess up averaging, take np.nan
                        value = np.nan

                    if c == "Content":
                        value = remove_non_ascii(value.lower())

                    thisReview[c] = value
                    
        if not thisReview["Content"] == None and len(thisReview["Content"]) > characterThreshold:
            #only add if theres content and its long enough
            data = data.append(thisReview, ignore_index=True)
    return data

def getStandardData(numFiles = 10):
    files = sorted(listdir('Review_Texts/'))
    df = getBlankFrame()

    for file in files[:numFiles]:
        df = addFileToData('Review_Texts/'+file, df)

    return df
      

def remove_non_ascii(text):
    return unidecode(unicode(text, encoding="utf-8"))




# alternate method to get keywords. LARA method based on Chi-square is basically same, but much better
def aspectSegmentationBayes(data, freq_threshold = 10, prob_threshold = 0.5, words_per_iter = 5, iters = 3):
    #break down reviews into sentences and break down each sentence into words using tokenizer and remove stopwords
    # returns list where each item is the list of words in that sentence
    sent_tokenized_reviews =  data['Content'].apply(lambda x: x.decode('utf-8')).apply(nltk.tokenize.sent_tokenize)
    sentences = [sentence for review in sent_tokenized_reviews for sentence in review]

    vectorizer =CountVectorizer(min_df = freq_threshold, binary=True,
                                                        ngram_range=(1,1),token_pattern = r'[a-zA-Z]{3,}',
                                                        stop_words=stopwords.words('english'))
    X = vectorizer.fit_transform(sentences)
    X = X.toarray() #convert sparse array to array

    count_sents_with_word = np.sum(X,axis=0) # works because binary

    # find Probability(sentence(S) has aspect(A) GIVEN S has word(W)) = count(S that have A and have W) / count(S that have W)

    for i in range(iters):
        for aspect in seeds:

            # if condition ensures code runs even if seed word not in our corpus by some chance
            seed_indices = [vectorizer.vocabulary_[word] for word in seeds[aspect] if vectorizer.vocabulary_.get(word) != None]
            count_sents_with_asp = np.sum(np.logical_or.reduce(X[:,seed_indices].T))
            count_sents_with_word_asp = np.sum(X[np.logical_or.reduce(X[:,seed_indices].T),:],axis=0)

            prob_asp_given_word = count_sents_with_word_asp.astype(float)/count_sents_with_word
            sorted_indices = np.argsort(prob_asp_given_word)[::-1]
            count = 0
            for j in sorted_indices:
                if count > words_per_iter or prob_asp_given_word[j] < prob_threshold:
                    break
                else:
                    if vectorizer.get_feature_names()[j] not in seeds[aspect]:
                        seeds[aspect].append(vectorizer.get_feature_names()[j])
                        count += 1
    return seeds

def filterReviewSentencesByWords(rev, words):
    #tokenize review into sentences
    sentences = tokenizer.tokenize(rev["Content"])
    
    #print(sentences)
    aspSentences = []
    for s in sentences:
        wordlist = re.sub("[^a-zA-Z]"," ", s).split()
        intersect = set(wordlist).intersection(words)
        #print(s)
        if len(intersect) != 0:
            aspSentences.append(s)

    #print(aspSentences)
    if len(aspSentences) == 0:
        #so no sentences contain the rating? what do we do ehre
        
        pass
    
    rev["aspectSentences"] = aspSentences

    return rev

def getAspectSentencesForReview(data, seedWords):
    #we have our seeds, now let's only keep the sentences which are relevant to that aspect in each review 
    
    data = data.apply(filterReviewSentencesByWords, axis=1, args=(seedWords,))
    return data

def getTrainingData(data):
    includeColumns = ["aspectRating", "aspectSentences", "Content", "Overall", ]
    data = getAspectSentencesForReview(data, seeds[aspect])
    data["aspectRating"] = data[aspect]
    return data[(pd.notnull(data[aspect])) & (data["aspectSentences"].apply(len) > 0)][includeColumns]

def unigram_creation(corpus):
    unigram_vectorizer = CountVectorizer(min_df=1,binary=False)
    X = unigram_vectorizer.fit_transform(corpus)
    X_1=X.toarray()
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf_X1 = transformer.fit_transform(X_1)
    return tfidf_X1.toarray()

def bigram_creation(corpus):
    bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)
    X_2 = bigram_vectorizer.fit_transform(corpus).toarray()
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf_X2 = transformer.fit_transform(X_2)
    return tfidf_X2.toarray()


def aspectSegmentationChiSquared(data, vocab=[], threshold=0, iterationLimit=3):
    #when we have the top chi-squared rated keywords, how many do we take
    keywordsToTake = 3
    
    reviews=data["Content"]
    #bootstrap iterations
    
    sent_tokenized_reviews = [nltk.tokenize.sent_tokenize(r.decode('utf-8')) for r in reviews]
    sentences = [sentence for r in sent_tokenized_reviews for sentence in r]
    
    vectorizer = CountVectorizer(min_df = 0, binary=True,
                                                        ngram_range=(1,1),token_pattern = r'[a-zA-Z]{3,}',
                                                        stop_words=stopwords.words('english'))
    X = vectorizer.fit_transform(sentences)
    X = X.toarray() #convert sparse array to array
    
    for i in range(0, iterationLimit):
        
        for aspect in seeds:
            seed_indices = [vectorizer.vocabulary_[word] for word in seeds[aspect] if vectorizer.vocabulary_.get(word) != None]
            row_mask = np.logical_or.reduce(X[:,seed_indices].T)
            sentences_with_aspect = X[row_mask,:]
            sentences_without_aspect = X[~row_mask,:]
            c_1 = np.sum(sentences_with_aspect,axis=0)
            c_2 = np.sum(sentences_without_aspect, axis=0)
            c_3 = len(sentences_with_aspect) - c_1
            c_4 = len(sentences_without_aspect) - c_2
            
            numer = 1.0*((c_1*c_4 - c_2 * c_3)**2)
            denom = 1.0*(c_1 + c_3)*(c_2 + c_4)*(c_1 + c_2)*(c_3 + c_4)
            
            csq = numer / denom
            
            count = 0
            sorted_indices = np.argsort(csq)[::-1]
            for x in sorted_indices:
                word = vectorizer.get_feature_names()[x]
                if word not in seeds[aspect]:
                    seeds[aspect].append(word)
                    count += 1
                if count == keywordsToTake:
                    break
                    
    return seeds


