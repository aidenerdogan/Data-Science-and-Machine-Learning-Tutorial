import csv,re,random
# stopwords list
stopwords = ["wouldn't", 'own', 'ours', 'under', "who's", 'also', "here's", 'could', 'have', 'but', 'com', 'ourselves', "where's", 'at', "they'd", 'hence', 'doing', "you're", "how's", 'r', 'yours', 'against', 'same', "i'd", 'which', 'more', 'very', 'whom', 'ought', 'http', 'for', 'was', 'most', 'of', 'his', 'are', 'so', 'did', 'am', "she'll", 'been', 'up', 'does', 'after', 'few', 'until', "they're", 'it', 'both', 'their', 'the', "there's", "can't", 'into', "he's", 'they', 'through', "why's", 'get', 'all', 'them', 'herself', "we'll", 'if', 'can', 'out', "i'm", 'and', 'because', "it's", "they've", "isn't", 'were', 'some', 'than', "wasn't", 'too', "that's", "they'll", 'that', "haven't", 'such', "i've", 'while', 'why', 'like', 'should', 'to', "you'll", 'hers', 'you', 'then', 'however', 'down', 'before', "she's", 'any', "what's", "he'd", 'here', 'www', "didn't", "you've", 'cannot', 'above', 'her', 'from', "shan't", "you'd", 'yourselves', 'an', "aren't", 'only', 'theirs', "we'd", 'on', 'having', 'each', 'else', 'who', 'or', "i'll", 'had', 'as', "mustn't", "couldn't", "weren't", 'me', 'further', 'over', 'where', 'when', 'no', 'being', "she'd", 'about', 'those', 'in', 'she', "when's", 'has', 'its', 'how', "don't", "hadn't", 'just', "we're", 'him', 'himself', 'once', 'he', 'since', 'again', 'my', 'this', "we've", 'off', 'would', 'not', 'these', 'your', 'therefore', 'themselves', 'our', 'is', 'between', 'ever', "won't", "he'll", 'i', 'do', "let's", 'with', "hasn't", 'k', 'otherwise', "doesn't", 'we', 'by', 'itself', 'shall', 'be', "shouldn't", 'below', 'during', 'myself', 'nor', 'what', 'other', 'there', 'yourself', 'a', 'mightn', 'mustn', 't', "mightn't", 'haven', 'don', 'isn', 's', 'wouldn', 'will', 'needn', 'hadn', 'll', 'couldn', 'aren', 'wasn', 'ain', 'm', 'didn', 'won', "that'll", "should've", "needn't", 're', 'o', 'doesn', 've', 'y', 'ma', 'now', 'd', 'shan', 'shouldn', 'hasn', 'weren']
# GET TRAIN DATA
# data info
# We are using labelled word for train model
# this data contains 160154 negative,90710 positive words
def get_train_data():
	# first 35 lines for data info
	# positive_words['1'] = {"abc":1, "pqr":3}
	# print(len(positive_words['1']))
	for word in open('positive_words.txt','r',encoding='ISO-8859-1').readlines()[35:]:
		word = word.replace('\n', '').lower()
		if word in data_set[1]:
			data_set[1][word] += 1
		else:
			data_set[1][word] = 1
	for word in open('negative_words.txt', 'r', encoding = "ISO-8859-1").readlines()[35:]:
		word = word.replace('\n', '').lower()
		if word in data_set[0]:
			data_set[0][word] += 1
		else:
			data_set[0][word] = 1
	# print(data_set)
	return data_set
def get_unique(data_set):
	unique_list = list(data_set[1])
	# print(unique_list)
	[unique_list.append(word[0]) for word in data_set[0] if word[0] not in unique_list]
	return unique_list

def sentiment_text(text):
	words =[word.lower() for word in (re.findall(r"[\w']+|[.,!?;]", text.rstrip())) if len(word) >= 3 and word not in stopwords]
	return words

def get_P_positive(p_word_positive,words):
	p_word_positive = 0
	total = sum(data_set[0].values())+sum(data_set[1].values())
	for word in words:
		if word in data_set[1]:
			p_word_positive += (
				(data_set[1][word]/sum(data_set[1].values())) * (sum(data_set[1].values())/total)
				) /((data_set[1][word] + (data_set[0][word] if word in data_set[0] else 0))/total)
		# else:
		# 	p_word_positive += 1/((1/(sum(data_set[1].values())+sum(data_set[0].values())))+ len(get_unique(data_set)))
	return p_word_positive

def get_P_negative(p_word_negative,words):
	p_word_negative = 0
	total = sum(data_set[0].values())+ sum(data_set[1].values())
	for word in words:
		if word in data_set[0]:
			p_word_negative += ((data_set[0][word]/sum(data_set[0].values()))*(sum(data_set[0].values())/total)
				)/(((data_set[1][word] if word in data_set[1] else 0) +  data_set[0][word])/total)
		# else:
			# p_word_negative += 1/((1/(sum(data_set[1].values())+sum(data_set[0].values())))+ len(get_unique(data_set)))
	return p_word_negative

def get_accuracy(file):
	TP_TN = 0
	with open(file,'r',encoding = "ISO-8859-1") as csv_file:
		csv_reader = csv.DictReader(csv_file)
		for (i,row) in enumerate(csv_reader):
			if i<101:
				if get_sentiment(sentiment_text(row['SentimentText'])) == row['Sentiment']:
					TP_TN += 1
	return TP_TN/100


def get_sentiment(words):
	result = None
	p_word_positive,p_word_negative = 0.0,0.0
	p_word_positive = get_P_positive(p_word_positive,words)
	p_word_negative = get_P_negative(p_word_negative,words)
	print('pos',p_word_positive)
	print('neg',p_word_negative)
	# '1' mean positive, '0' mean negative
	if p_word_positive > p_word_negative:
		result = '1'
	elif p_word_positive < p_word_negative :
		result = '0'
	else:
		result = 'neatural'
	return result

if __name__ == '__main__':
	data_set = {1:{},0:{}}
	get_train_data()
	# text = input('pls input a text :')
	import timeit
	start = timeit.default_timer()
	# print(get_sentiment(sentiment_text(text)))
	print(get_accuracy('train.csv'))
	stop = timeit.default_timer()
	print('Time: ', stop - start)
	