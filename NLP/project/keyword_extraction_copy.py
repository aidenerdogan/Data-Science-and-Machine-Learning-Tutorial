"""
author: Dhaval Barot
Description: Keyword extraction from dataset
"""
import nltk
# nltk.download('wordnet')
# nltk.download('porter')
# nltk.download('stopwords')

import pandas
import pickle
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer


def read_csv(filename):
    """
    :param filename: filename of dataset
    :return: return dataset
    """
    dataset = pandas.read_csv(filename, delimiter=',')
    return dataset


def word_count(dataset, col_name):
    """
    :param dataset: put your dataset
    :param col_name: put column on which you want to extract keywords
    :return: new dataset with counts
    """
    # Fetch wordcount for each abstract
    dataset['word_count'] = dataset[col_name].apply(lambda x: len(str(x).split(" ")))
    return dataset

# Word Count
# print(dataset[['abstract1','word_count']].head())

# Descriptive statistics of word counts
# print(dataset.word_count.describe())


def identify_comm_words(dataset, len=10):
    """
    :param dataset: put your dataset
    :param len: put length of the list
    :return: list of common words
    """
    # Identify common words
    return pandas.Series(
        ' '.join(dataset).split()).value_counts()[:len]


def identify_unique_words(dataset, len):
    """
    :param dataset: put your dataset
    :param len: put length of the list
    :return: list of unique words
    """
    # Identify uncommon words
    return pandas.Series(' '.join(dataset['abstract']).split()).value_counts()[-len:]


def append_new_stop_words(stop_words, new_stop_words=[]):
    """
    :param stop_words: put stop word set
    :param new_stop_words: put list of stop words
    :return: returns new set of stop words
    """
    stop_words = set(stop_words.words()).union(new_stop_words)
    return stop_words


def remove_common_words(dataset, stop_words, num_of_rec=0):
    """
    :param dataset: put your dataset
    :param stop_words: put stop word set
    :param num_of_rec: number of records in set
    :return: keywords set with out common words
    """
    corpus = []
    for i in range(0, num_of_rec):
        # Remove punctuations
        text = re.sub('[^a-zA-Z]', ' ', dataset['abstract'][i])

        # Convert to lowercase
        text = text.lower()

        # remove tags
        text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)

        # remove special characters and digits
        text = re.sub("(\\d|\\W)+", " ", text)

        # Convert to list from string
        text = text.split()

        # Stemming
        ps = PorterStemmer()

        # Lemmatisation
        lem = WordNetLemmatizer()
        text = [lem.lemmatize(word) for word in text if word not in stop_words]
        text = " ".join(text)
        corpus.append(text)

    return corpus


# Function for sorting tf_idf in descending order
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


file_name = 'papers2.csv'
col_name = 'abstract'


def train_model(file_name, col_name):
    dataset = read_csv(file_name)

    dataset = word_count(dataset, col_name)

    stop_words = append_new_stop_words(stopwords)

    corpus = remove_common_words(dataset, stop_words, 3924)

    cv = CountVectorizer(max_df=0.8, stop_words=stop_words, max_features=10000, ngram_range=(1, 3))

    X = cv.fit_transform(corpus)
    # print('==============',list(cv.vocabulary_.keys()))

    filename = 'finalized_model.sav'
    pickle.dump(X, open(filename, 'wb'))
    print('=========',cv, X[:50])

    return cv, X

# ------------------------------------------------------------------------
cv, X = train_model(file_name, col_name)
#
# # get feature names
# feature_names = cv.get_feature_names()
#
#
# tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
# tfidf_transformer.fit(X)
# # ----------------input abstract--------------
# doc = 'present theory compositionality stochastic optimal control showing task optimal controller constructed certain ' \
#       'primitive primitive feedback controller pursuing agenda mixed proportion much progress making towards agenda ' \
#       'compatible agenda present task resulting composite control law provably optimal problem belongs certain class ' \
#       'class rather general yet number unique property bellman equation made linear even linear discrete dynamic give ' \
#       'rise compositionality developed special case linear dynamic gaussian noise framework yield analytical solution ' \
#       'linear mixture linear quadratic regulator without requiring final cost quadratic generally natural set control ' \
#       'primitive constructed applying svd green function bellman equation illustrate theory context human arm ' \
#       'movement idea optimality compositionality prominent field motor control yet hard reconcile work make possible '
#
# # generate tf-idf for the given document
# tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
#
# # sort the tf-idf vectors by descending order of scores
# sorted_items = sort_coo(tf_idf_vector.tocoo())
#
# # extract only the top n; n here is 10
# keywords = extract_topn_from_vector(feature_names, sorted_items, 50)
#
# # now print the results
# print("\nAbstract:")
# print(doc)
# print("\nKeywords:")
# for k in keywords:
#     print
# ------------------------------------------------------------------------


dataset = read_csv(file_name)

dataset = word_count(dataset, col_name)

stop_words = append_new_stop_words(stopwords)

corpus = remove_common_words(dataset, stop_words, 3924)

print(len(corpus))
wordlist = identify_comm_words(corpus, 10)

print(wordlist)