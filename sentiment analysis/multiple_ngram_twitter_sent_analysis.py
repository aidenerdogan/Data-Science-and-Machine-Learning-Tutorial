#!/usr/bin/env python
# coding: utf-8

from nltk import ngrams
from nltk.corpus import stopwords
import string
import pandas as pd
import numpy as np
import timeit
from random import shuffle


stopwords_english = stopwords.words('english')

# clean words, i.e. remove stopwords and punctuation
def clean_words(words, stopwords_english):
    words_clean = []
#     print(type(words))
    for word in words:
        word = word.lower()
        if word not in stopwords_english and word not in string.punctuation:
            words_clean.append(word)
    return words_clean

# feature extractor function for unigram
def bag_of_words(words):
    words_dictionary = dict([word, True] for word in words)
    return words_dictionary

# feature extractor function for ngrams (bigram)
def bag_of_ngrams(words, n=2):
    words_ng = []
    for item in iter(ngrams(words, n)):
        words_ng.append(item)
    words_dictionary = dict([word, True] for word in words_ng)
    return words_dictionary
important_words = ['above', 'below', 'off', 'over', 'under', 'more', 'most', 'such', 'no', 'nor', 'not', 'only', 'so', 'than', 'too', 'very', 'just', 'but']
stopwords_english_for_bigrams = set(stopwords_english) - set(important_words)

# let's define a new function that extracts all features
# i.e. that extracts both unigram and bigrams features
def bag_of_all_words(words, n=2):
    words_clean = clean_words(words, stopwords_english)
    words_clean_for_bigrams = clean_words(words, stopwords_english_for_bigrams)

    unigram_features = bag_of_words(words_clean)
    bigram_features = bag_of_ngrams(words_clean_for_bigrams)

    all_features = unigram_features.copy()
    all_features.update(bigram_features)

    return all_features

tweet_data = pd.read_csv('clean_tweet2.csv',index_col=0)
tweet_data.head()

print(tweet_data.shape)

print(tweet_data.info())

print(tweet_data[tweet_data.isnull().any(axis=1)].head())

print(np.sum(tweet_data.isnull().any(axis=1)))

print(tweet_data.isnull().any(axis=0))

tweet_data.dropna(inplace=True)
tweet_data.reset_index(drop=True,inplace=True)
print(tweet_data.info())

print(tweet_data.shape)

# pos_reviews = np.array([row for row in tweet_data[tweet_data.sentiment == 1].text.str.split()])
pos_reviews = [row for row in tweet_data[tweet_data.sentiment == 1].text.str.split()]
# [value for (index, value) in pos_reviews.items()]
# pos_reviews

# pos_reviews = np.array([row for row in tweet_data[tweet_data.sentiment == 1].text.str.split()])
neg_reviews = [row for row in tweet_data[tweet_data.sentiment == 0].text.str.split()]
# [value for (index, value) in pos_reviews.items()]
# neg_reviews

pos_reviews_set = []
for words in pos_reviews:
#     print(words)
    pos_reviews_set.append((bag_of_all_words(words), 'pos'))
# pos_reviews_set

neg_reviews_set = []
for words in neg_reviews:
#     print(words)
    neg_reviews_set.append((bag_of_all_words(words), 'neg'))
# neg_reviews_set

print ('len of pos data',len(pos_reviews_set), 'len of beg data', len(neg_reviews_set)) # Output: (1000, 1000)

# radomize pos_reviews_set and neg_reviews_set
# doing so will output different accuracy result everytime we run the program
shuffle(pos_reviews_set)
shuffle(neg_reviews_set)

test_set = pos_reviews_set[:int((0.2*len(pos_reviews_set)))] + neg_reviews_set[:int((0.2*len(neg_reviews_set)))]
train_set = pos_reviews_set[int((0.2*len(pos_reviews_set))):] + neg_reviews_set[int((0.2*len(neg_reviews_set))):]

print('len of train data',len(test_set), 'len of test data', len(train_set))
print('\nTRANING MODEL\n')
from nltk import classify
from nltk import NaiveBayesClassifier
start = timeit.default_timer()
classifier = NaiveBayesClassifier.train(train_set)

accuracy = classify.accuracy(classifier, test_set)
stop = timeit.default_timer()
print('total time :', stop-start)
print('time for each row :',(stop-start)/len(test_set))
print('accuracy',accuracy)
print (classifier.show_most_informative_features(10))

print('\nExamles\n')
from nltk.tokenize import word_tokenize
custom_review = "I hated the film. It was a disaster. Poor direction, bad acting."
print('example 1 :',custom_review)
custom_review_tokens = word_tokenize(custom_review)
custom_review_set = bag_of_all_words(custom_review_tokens)
print (classifier.classify(custom_review_set))
# Negative review correctly classified as negative

# probability result
prob_result = classifier.prob_classify(custom_review_set)
print (prob_result)
print (prob_result.max())
print (prob_result.prob("neg"),'Negative')
print (prob_result.prob("pos"),'Positive')


custom_review = "It was a wonderful and amazing movie. I loved it. Best direction, good acting."
print('example 2 :',custom_review)
custom_review_tokens = word_tokenize(custom_review)
custom_review_set = bag_of_all_words(custom_review_tokens)

print (classifier.classify(custom_review_set))
# Positive review correctly classified as positive

# probability result
prob_result = classifier.prob_classify(custom_review_set)
print (prob_result)
print (prob_result.max())
print (prob_result.prob("neg"),'Negative')
print (prob_result.prob("pos"),'Positive')

custom_review = input('please enter a sentence :')
custom_review_tokens = word_tokenize(custom_review)
custom_review_set = bag_of_all_words(custom_review_tokens)

print (classifier.classify(custom_review_set))
# Positive review correctly classified as positive

# probability result
prob_result = classifier.prob_classify(custom_review_set)
print (prob_result)
print (prob_result.max())
print (prob_result.prob("neg"),'Negative')
print (prob_result.prob("pos"),'Positive')
