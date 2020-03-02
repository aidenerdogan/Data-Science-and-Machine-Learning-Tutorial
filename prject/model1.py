import sqlite3

def get_data():
	pos_words = []
	neg_words = []

	for pos_word in open('positive-words.txt', 'r',encoding = "ISO-8859-1").readlines()[35:]:
	    pos_words.append(({pos_word.rstrip(): True}, 'Positive'))

	for neg_word in open('negative-words.txt', 'r', encoding = "ISO-8859-1").readlines()[35:]:
	    neg_words.append(({neg_word.rstrip(): True}, 'Negative'))
	all_words_with_sentiment = pos_words + neg_words
	return all_words_with_sentiment
# for i in get_data():
# 	print(i)
from nltk.classify import NaiveBayesClassifier
classifier = NaiveBayesClassifier.train(get_data())

def read_test_data():
	conn = sqlite3.connect('database.sqlite')
	c = conn.cursor()
	test_data = c.execute("SELECT * FROM Sentiment")
	return test_data
# 	for i in test_data:
# 		print(i[15])
print(read_test_data())