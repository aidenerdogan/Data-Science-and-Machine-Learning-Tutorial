import csv
from sklearn.feature_extraction.text import CountVectorizer
# what is difference between nltk NB and sklearn NB(BNB,MNB,GNB)
from sklearn.naive_bayes import BernoulliNB


# GET TRAIN DATA
# data info
# We are using labelled word for train model
# this data contains 160154 negative,90710 positive words
def get_train_data():
	# first 35 lines for data info
	for word in open('positive_words.txt','r',encoding='ISO-8859-1').readlines()[35:]:
		word = word.replace('\n', '').lower()
		if word in positive_words_counts:
			positive_words_counts[word] += 1
		else:
			positive_words_counts[word] = 1
		positive_words.append([word.rstrip(),1])
	for word in open('negative_words.txt', 'r', encoding = "ISO-8859-1").readlines()[35:]:
		word = word.replace('\n', '').lower()
		if word in negative_words_counts:
			negative_words_counts[word] += 1
		else:
			negative_words_counts[word] = 1
		negative_words.append([word.rstrip(),0])
	data = positive_words + negative_words

# TRAIN MODEL
# we are using BernoulliNB of Naive Bayes Classification algorithm for train model
def training_model(data, word_vectorizer):
		print('===Trainining Model...===')
		# train_text is data['text']
		train_text = [data[0] for data in data]
		# train_sentiment mean data sentiment class
		train_sentiment =[data[1] for data in data]
		# print(train_text[:20])
		train_text = word_vectorizer.fit_transform(train_text)
		# print(train_text[1])
		print('===Train Finished===')
		return BernoulliNB().fit(train_text,train_sentiment)
def get_test_data(file):
	with open(file,'r') as csv_file:
		csv_reader = csv.DictReader(csv_file)
		# [print(row) for row in csv_reader]
# get_test_data('Sentiment.csv')
if __name__ == '__main__':
	positive_words = []
	negative_words = []
	positive_words_counts = dict()
	negative_words_counts = dict()
	get_train_data()
	# print(negative_words_counts)
	# print(positive_words_counts)
	# print(positive_words)
	data = positive_words + negative_words
	word_vectorizer = CountVectorizer(binary = 'true')
	Bernoulli_classifier = training_model(data, word_vectorizer)
	result = Bernoulli_classifier.predict(word_vectorizer.transform(["I like this tv shows!"]))
	print('---Test of Vectorizer---\n',result[0])
	# txt = ["Dravid is a former Indian cricket player and captain","greatest batsmen in the history of cricket.","Dravid was one of the reasons behind India's great cricket history"]
	# count_train = word_vectorizer.fit_transform(txt) # this fits on given corpus to create vocabulary
	# print(count_train)