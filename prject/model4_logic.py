import csv,re,random
# stopwords list
stopwords = ["wouldn't", 'own', 'ours', 'under', "who's", 'also', "here's", 'could', 'have', 'but', 'com', 'ourselves', "where's", 'at', "they'd", 'hence', 'doing', "you're", "how's", 'r', 'yours', 'against', 'same', "i'd", 'which', 'more', 'very', 'whom', 'ought', 'http', 'for', 'was', 'most', 'of', 'his', 'are', 'so', 'did', 'am', "she'll", 'been', 'up', 'does', 'after', 'few', 'until', "they're", 'it', 'both', 'their', 'the', "there's", "can't", 'into', "he's", 'they', 'through', "why's", 'get', 'all', 'them', 'herself', "we'll", 'if', 'can', 'out', "i'm", 'and', 'because', "it's", "they've", "isn't", 'were', 'some', 'than', "wasn't", 'too', "that's", "they'll", 'that', "haven't", 'such', "i've", 'while', 'why', 'like', 'should', 'to', "you'll", 'hers', 'you', 'then', 'however', 'down', 'before', "she's", 'any', "what's", "he'd", 'here', 'www', "didn't", "you've", 'cannot', 'above', 'her', 'from', "shan't", "you'd", 'yourselves', 'an', "aren't", 'only', 'theirs', "we'd", 'on', 'having', 'each', 'else', 'who', 'or', "i'll", 'had', 'as', "mustn't", "couldn't", "weren't", 'me', 'further', 'over', 'where', 'when', 'no', 'being', "she'd", 'about', 'those', 'in', 'she', "when's", 'has', 'its', 'how', "don't", "hadn't", 'just', "we're", 'him', 'himself', 'once', 'he', 'since', 'again', 'my', 'this', "we've", 'off', 'would', 'not', 'these', 'your', 'therefore', 'themselves', 'our', 'is', 'between', 'ever', "won't", "he'll", 'i', 'do', "let's", 'with', "hasn't", 'k', 'otherwise', "doesn't", 'we', 'by', 'itself', 'shall', 'be', "shouldn't", 'below', 'during', 'myself', 'nor', 'what', 'other', 'there', 'yourself', 'a', 'mightn', 'mustn', 't', "mightn't", 'haven', 'don', 'isn', 's', 'wouldn', 'will', 'needn', 'hadn', 'll', 'couldn', 'aren', 'wasn', 'ain', 'm', 'didn', 'won', "that'll", "should've", "needn't", 're', 'o', 'doesn', 've', 'y', 'ma', 'now', 'd', 'shan', 'shouldn', 'hasn', 'weren']
# GET TRAIN DATA
# data info
# We are using labelled word for train model
# this data contains 160154 negative,90710 positive words
def get_train_data():
	positive_words,negative_words = [],[]
	# first 35 lines for data info
	for word in open('positive_words.txt','r',encoding='ISO-8859-1').readlines()[35:]:
		word = word.replace('\n', '').lower()
		positive_words.append([word.rstrip(),1.0])
	for word in open('negative_words.txt', 'r', encoding = "ISO-8859-1").readlines()[35:]:
		word = word.replace('\n', '').lower()
		negative_words.append([word.rstrip(),0.0])
	or_data = positive_words + negative_words
	data = random.sample(or_data, len(or_data))
	return data

# TRAIN MODEL
# split data to training and test data via split_ratio
def split_data_set(data_set, split_ratio):
    trainSize = int(len(data_set) * split_ratio)
    trainSet = []
    copy = list(data_set)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]	

# get all polarities from data
def get_probability(data_test):
	# step 1 get P_class
	for word in training_set:
		if word[1] == 1.0:
			if word[0] in positive_words_counts:
					positive_words_counts[word[0]] += 1
			else:
					positive_words_counts[word[0]] = 1
		else:
			if word[0] in negative_words_counts:
				negative_words_counts[word[0]] += 1
			else:
				negative_words_counts[word[0]] = 1
	# laplace_smooting(data_test)
	P_positive = sum(positive_words_counts.values())/len(training_set)
	P_negative = sum(negative_words_counts.values())/len(training_set)
	# step 2 get P_words
	P_positive_words = dict()
	P_negative_words = dict()
	for word in positive_words_counts.keys():
		P_positive_words[word] = positive_words_counts[word]/len(training_set)
	for word in negative_words_counts.keys():
		P_negative_words[word] = negative_words_counts[word]/len(training_set)
	return P_positive,P_negative,P_positive_words,P_negative_words
def laplace_smooting(test_set):
	for word in test_set:
		if word[0] not in positive_words_counts:
			for word2 in positive_words_counts:
				positive_words_counts[word2] += 1
			positive_words_counts[word[0]] = 1
		if word[0] not in negative_words_counts:
			for word2 in negative_words_counts:
				negative_words_counts[word2] += 1
			negative_words_counts[word[0]] = 1

def get_predict(test_set):
	predict = []
	for word in test_set:
		p_word_positive,p_word_negative = 0.0,0.0
		if word[0] in P_positive_words:
			p_word_positive = (P_positive_words[word[0]]*P_positive)/(P_positive_words[word[0]]+P_negative_words[word[0]])
		else:
			pass
		if word[0] in P_negative_words:
			p_word_negative = (P_negative_words[word[0]]*P_negative)/(P_positive_words[word[0]]+P_negative_words[word[0]])
		else:
			pass
		if p_word_positive > p_word_negative:
			predict.append([word[0],1.0])
		else:
			predict.append([word[0],0.0])
	return predict

def get_accuracy(predict):
	count = 0
	TN,TP,FN,FP = 0,0,0,0
	for i in range(len(test_set)):
		if test_set[i][1] == predict[i][1]:
			if predict[i][1] == 1.0:
				TP += 1
			else:
				TN += 1
			count += 1
		else:
			if predict[i][1] == 0.0:
				FN += 1
			else:
				FP += 1 
	accuracy1 = (TN+TP)/(TN+TP+FN+FP)
	accuracy = count/len(test_set)
	preccision = TP/(FP+TP)
	specificity = TN/(TN+FP)
	sensivity = TP/(TP+FN)
	print('accuracy1',accuracy1)
	print('accuracy',accuracy)
	print('preccision',preccision)
	print('sepcificion',specificity)
	print('sensivity',sensivity)
	# return accuracy
def sentiment_text(text):
	words =[word.lower() for word in (re.findall(r"[\w']+|[.,!?;]", text.rstrip())) if len(word) >= 3 and word not in stopwords]
	print(words)
	return words
def get_sentiment(words):
	result = None
	p_word_positive,p_word_negative = 0.0,0.0
	# get_probability(words)
	for word in words:
		if word in P_positive_words:
			p_word_positive = (P_positive_words[word]*P_positive)/(P_positive_words[word]+P_negative_words[word])
			# p_word_positive = p_word_positive*P_positive_words[word]
		else:
			p_word_positive = p_word_positive*1
		if word in P_negative_words:
			p_word_negative = (P_negative_words[word]*P_negative)/(P_positive_words[word]+P_negative_words[word])
		else:
			p_word_negative = p_word_negative*1
	p_word_positive = p_word_positive*P_positive
	p_word_negative = p_word_negative*P_negative
	if p_word_positive > p_word_negative:
		result = 'positive'
	else:
		result = 'negative'
	print(result)
	return result

if __name__ == '__main__':
	data_set = get_train_data()
	positive_words_counts = dict()
	negative_words_counts = dict()
	training_set, test_set = split_data_set(data_set,0.67)
	print('Split {0} row into train = {1} and test {2} rows'.format(len(data_set),len(training_set),len(test_set)))
	P_positive, P_negative,P_positive_words,P_negative_words = get_probability(test_set)
	# print(P_positive_words)
	# print(P_negative_words)
	text = input('pls input a text :')
	get_predict(test_set)
	get_accuracy(get_predict(test_set))
	# sentiment_text(text)
	# laplace_smooting(sentiment_text(text))
	get_sentiment(sentiment_text(text))