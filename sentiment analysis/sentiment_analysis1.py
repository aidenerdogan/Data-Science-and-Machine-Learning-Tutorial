#!/usr/bin/env python
# coding: utf-8

# In[268]:


from nltk import ngrams
from nltk.corpus import stopwords
import string
import pandas


# In[269]:


stopwords_english = stopwords.words('english')


# In[270]:


# clean words, i.e. remove stopwords and punctuation
def clean_words(words, stopwords_english):
    words_clean = []
#     print(type(words))
    for word in words:
        word = word.lower()
        if word not in stopwords_english and word not in string.punctuation:
            words_clean.append(word)
    return words_clean


# In[271]:


# feature extractor function for unigram
def bag_of_words(words):
    words_dictionary = dict([word, True] for word in words)
    return words_dictionary


# In[272]:


# feature extractor function for ngrams (bigram)
def bag_of_ngrams(words, n=2):
    words_ng = []
    for item in iter(ngrams(words, n)):
        words_ng.append(item)
    words_dictionary = dict([word, True] for word in words_ng)
    return words_dictionary


# In[273]:


from nltk.tokenize import word_tokenize
text = "It was a very good movie."
words = word_tokenize(text.lower())


# In[274]:


print(words)


# In[275]:


print(bag_of_ngrams(words))


# In[276]:


words_clean = clean_words(words, stopwords_english)
print (words_clean)


# In[277]:


important_words = ['above', 'below', 'off', 'over', 'under', 'more', 'most', 'such', 'no', 'nor', 'not', 'only', 'so', 'than', 'too', 'very', 'just', 'but']


# In[278]:


stopwords_english_for_bigrams = set(stopwords_english) - set(important_words)

words_clean_for_bigrams = clean_words(words, stopwords_english_for_bigrams)
print (words_clean_for_bigrams)


# In[279]:


unigram_features = bag_of_words(words_clean)
print(unigram_features)

# In[280]:


bigram_features = bag_of_ngrams(words_clean_for_bigrams)
print(bigram_features)


# In[281]:


# combine both unigram and bigram features
all_features = unigram_features.copy()
all_features.update(bigram_features)
print(all_features)


# In[282]:


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


# In[283]:


print(bag_of_all_words(words))


# In[284]:


from nltk.corpus import movie_reviews

pos_reviews = []
for fileid in movie_reviews.fileids('pos'):
    words = movie_reviews.words(fileid)
    pos_reviews.append(words)

neg_reviews = []
for fileid in movie_reviews.fileids('neg'):
    words = movie_reviews.words(fileid)
    neg_reviews.append(words)


# In[285]:


# print(pos_reviews[0])


# In[286]:


# positive reviews feature set
pos_reviews_set = []
for words in pos_reviews:
    pos_reviews_set.append((bag_of_all_words(words), 'pos'))

# negative reviews feature set
neg_reviews_set = []
for words in neg_reviews:
    neg_reviews_set.append((bag_of_all_words(words), 'neg'))


# In[287]:


# print(pos_reviews_set)


# In[288]:


tweet_data = pandas.read_csv('clean_tweet2.csv')


# In[292]:


pos_reviews = list(tweet_data[tweet_data.sentiment == 1].text.str.lower().str.split())
# print('pos_reviews\n\n',pos_reviews)


# In[290]:


neg_reviews = list(tweet_data[tweet_data.sentiment == 0].text.str.lower().str.split())
# len(neg_reviews)+len(pos_reviews)
# print(neg_reviews)


# In[300]:


# positive reviews feature set
for i in range(len(pos_reviews)):
    print(pos_reviews[i])
    bags = bag_of_all_words(pos_reviews[i])
    print(bags)
# pos_reviews_set2 = []
# for index in range(len(pos_reviews)-1):
#     pos_reviews_set2.append((bag_of_all_words(pos_reviews[index]), 'pos'))
# print(pos_reviews_set2)
# # negative reviews feature set
# neg_reviews_set2 = []
# for words in neg_reviews:
#     neg_reviews_set2.append((bag_of_all_words(words), 'neg'))
# print(neg_reviews_set2)

