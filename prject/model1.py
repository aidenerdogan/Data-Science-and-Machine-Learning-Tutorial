import sqlite3
# get positive and negative words from text file
def get_tarin_data():
 positive_words = []
 negative_words = []
 for word in open('positive_words.txt', 'r',encoding = "ISO-8859-1").readlines()[35:]:
     positive_words.append(({word.rstrip(): True}, 'positive'))

 for word in open('negative_words.txt', 'r', encoding = "ISO-8859-1").readlines()[35:]:
     negative_words.append(({word.rstrip(): True}, 'negative'))
 all_words_with_sentiment = positive_words + negative_words
 return all_words_with_sentiment
# Navi Bayes Classifier
from nltk.classify import NaiveBayesClassifier
classifier = NaiveBayesClassifier.train(get_tarin_data())
# get data from sqlite3 darabase
def read_test_data():
 conn = sqlite3.connect('database.sqlite')
 c = conn.cursor()
 test_data = c.execute("SELECT * FROM Sentiment")
 return test_data

def to_dictionary(words):
 return dict([(word, True) for word in words])

test_data = []
# classify data
def predict_sentiment(text, expected_sentiment=None):
    classfy_text = to_dictionary(text.split())
    result = classifier.classify(classfy_text)
    test_data.append([classfy_text, expected_sentiment])
    return result

stopwords = ["wouldn't", 'own', 'ours', 'under', "who's", 'also', "here's", 'could', 'have', 'but', 'com', 'ourselves', "where's", 'at', "they'd", 'hence', 'doing', "you're", "how's", 'r', 'yours', 'against', 'same', "i'd", 'which', 'more', 'very', 'whom', 'ought', 'http', 'for', 'was', 'most', 'of', 'his', 'are', 'so', 'did', 'am', "she'll", 'been', 'up', 'does', 'after', 'few', 'until', "they're", 'it', 'both', 'their', 'the', "there's", "can't", 'into', "he's", 'they', 'through', "why's", 'get', 'all', 'them', 'herself', "we'll", 'if', 'can', 'out', "i'm", 'and', 'because', "it's", "they've", "isn't", 'were', 'some', 'than', "wasn't", 'too', "that's", "they'll", 'that', "haven't", 'such', "i've", 'while', 'why', 'like', 'should', 'to', "you'll", 'hers', 'you', 'then', 'however', 'down', 'before', "she's", 'any', "what's", "he'd", 'here', 'www', "didn't", "you've", 'cannot', 'above', 'her', 'from', "shan't", "you'd", 'yourselves', 'an', "aren't", 'only', 'theirs', "we'd", 'on', 'having', 'each', 'else', 'who', 'or', "i'll", 'had', 'as', "mustn't", "couldn't", "weren't", 'me', 'further', 'over', 'where', 'when', 'no', 'being', "she'd", 'about', 'those', 'in', 'she', "when's", 'has', 'its', 'how', "don't", "hadn't", 'just', "we're", 'him', 'himself', 'once', 'he', 'since', 'again', 'my', 'this', "we've", 'off', 'would', 'not', 'these', 'your', 'therefore', 'themselves', 'our', 'is', 'between', 'ever', "won't", "he'll", 'i', 'do', "let's", 'with', "hasn't", 'k', 'otherwise', "doesn't", 'we', 'by', 'itself', 'shall', 'be', "shouldn't", 'below', 'during', 'myself', 'nor', 'what', 'other', 'there', 'yourself', 'a', 'mightn', 'mustn', 't', "mightn't", 'haven', 'don', 'isn', 's', 'wouldn', 'will', 'needn', 'hadn', 'll', 'couldn', 'aren', 'wasn', 'ain', 'm', 'didn', 'won', "that'll", "should've", "needn't", 're', 'o', 'doesn', 've', 'y', 'ma', 'now', 'd', 'shan', 'shouldn', 'hasn', 'weren']

import collections
import nltk.classify
import nltk.metrics
# clean negative twitts (delete special charecters)
def clean_neg_data():
 neg_data = [row[15] for row in read_test_data() if row[5] == 'Negative']
 neg_cleaned_word = []
 for sent in neg_data:
  words = " ".join([word for word in sent.split() if 'http' not in word and not word.startswith('@') and not word.startswith('#') and word != 'RT' and len(word) >= 3 and word not in stopwords])
  neg_cleaned_word.append(words)
 return neg_cleaned_word

# clean positive twitts (delete special charecters)
# row[15] mean text
# row[5] mean sentiment
def clean_pos_data():
 pos_data = [row[15] for row in read_test_data() if row[5] == 'Positive']
 pos_cleaned_word = []
 for sent in pos_data:
  words = " ".join([word for word in sent.split() if 'http' not in word and not word.startswith('@') and not word.startswith('#') and word != 'RT' and len(word) >= 3 and word not in stopwords])
  pos_cleaned_word.append(words)
 return pos_cleaned_word

def sentiment_analysis_twits_by_labelled_words():
 expected_negative_set = collections.defaultdict(set)
 actual_negative_set = collections.defaultdict(set)
 for index, review in enumerate(clean_neg_data()):
  expected_negative_set['negative'].add(index)
  actual_sentiment = predict_sentiment(review, 'negative')
  actual_negative_set[actual_sentiment].add(index)
 print ("Total Positive found in negative reviews %s" % len(actual_negative_set['positive']))
 expected_positive_set = collections.defaultdict(set)
 actual_positive_set = collections.defaultdict(set)
 for index, review in enumerate(clean_pos_data()):
  expected_positive_set['positive'].add(index)
  actual_sentiment = predict_sentiment(review, 'positive')
  actual_positive_set[actual_sentiment].add(index)
 print ("Total Negative found in positive reviews %s" % len(actual_positive_set['negative']))


 print ('Accuracy: %.2f' % nltk.classify.util.accuracy(classifier, test_data))
 print ('Positive Precision: %.2f' % nltk.precision(expected_positive_set['positive'], actual_positive_set['positive']))
 print ('Positive Recall: %.2f' % nltk.recall(expected_positive_set['positive'], actual_positive_set['positive']))
 print ('Negative Precision: %.2f' % nltk.precision(expected_negative_set['negative'], actual_negative_set['negative']))
 print ('Negative Recall: %.2f' % nltk.recall(expected_negative_set['negative'], actual_negative_set['negative']))
sentiment_analysis_twits_by_labelled_words()

