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
def aspectSegmentationBayes(reviews, freq_threshold = .5, prob_threshold = 0.2, words_per_iter = 4, iters = 5):
    #break down reviews into sentences and break down each sentence into words using tokenizer and remove stopwords
    # returns list where each item is the list of words in that sentence
    sentence_words = []
    for review in reviews:
        review = review.decode('utf-8')
        sentences = nltk.tokenize.sent_tokenize(review)
        for sentence in sentences:
            sentence_words.append([x.lower() for x in nltk.tokenize.word_tokenize(sentence) if x not in stopwords.words('english') and len(x) > 2])    
    

    # find Probability(sentence(S) has aspect(A) GIVEN S has word(W)) = count(S that have A and have W) / count(S that have W)

    for i in range(iters):
        
        sents_with_word_asp = {}
        sents_with_word = {}
        sents_with_aspect = {}
        prob_asp_given_word = {}

        # calculates counts of (S that have W) and (S that have A and W)
        for sentence in sentence_words:
            for word in sentence:
                sents_with_word[word] = sents_with_word.get(word,0) + 1
                for aspect, aspect_words in seeds.items():
                    for aspect_word in aspect_words:
                        if aspect_word in sentence:
                            sents_with_word_asp[(word,aspect)] = sents_with_word_asp.get((word, aspect), 0) + 1
                            sents_with_aspect[aspect] = sents_with_aspect.get(aspect,0) + 1
                            break

        for (word, aspect), count in sents_with_word_asp.items():
            #susceptible to low frequencies. hence freq_threshold
            #freq_threshold ensures that count(S with  W) is atleast x% of count(S)
            if sents_with_word[word] > (freq_threshold/100.0)*len(sentence_words):
                prob_asp_given_word[(word,aspect)] = count/float(sents_with_word[word])

        prob_asp_given_word_sorted = sorted(prob_asp_given_word.items(), key=itemgetter(1),reverse=True)
        
        for aspect, word_list in seeds.items():
            count = 0
            for item in prob_asp_given_word_sorted:
                #item is of the form ((word,aspect),probability)
                if item[0][1] == aspect:
                    if item[0][0] not in word_list:
                        if count <= words_per_iter:
                            if item[1] >= prob_threshold:
                                seeds[aspect].append(item[0][0])
                                count += 1
                            else:
                                #because sorted, the others can't have higher probability
                                break
                        else:
                            # because limiit of words per aspect in this iteration has been reached
                            break

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



