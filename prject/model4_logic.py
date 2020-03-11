import csv,re, random,math
stopwords = ["wouldn't", 'own', 'ours', 'under', "who's", 'also', "here's", 'could', 'have', 'but', 'com', 'ourselves', "where's", 'at', "they'd", 'hence', 'doing', "you're", "how's", 'r', 'yours', 'against', 'same', "i'd", 'which', 'more', 'very', 'whom', 'ought', 'http', 'for', 'was', 'most', 'of', 'his', 'are', 'so', 'did', 'am', "she'll", 'been', 'up', 'does', 'after', 'few', 'until', "they're", 'it', 'both', 'their', 'the', "there's", "can't", 'into', "he's", 'they', 'through', "why's", 'get', 'all', 'them', 'herself', "we'll", 'if', 'can', 'out', "i'm", 'and', 'because', "it's", "they've", "isn't", 'were', 'some', 'than', "wasn't", 'too', "that's", "they'll", 'that', "haven't", 'such', "i've", 'while', 'why', 'like', 'should', 'to', "you'll", 'hers', 'you', 'then', 'however', 'down', 'before', "she's", 'any', "what's", "he'd", 'here', 'www', "didn't", "you've", 'cannot', 'above', 'her', 'from', "shan't", "you'd", 'yourselves', 'an', "aren't", 'only', 'theirs', "we'd", 'on', 'having', 'each', 'else', 'who', 'or', "i'll", 'had', 'as', "mustn't", "couldn't", "weren't", 'me', 'further', 'over', 'where', 'when', 'no', 'being', "she'd", 'about', 'those', 'in', 'she', "when's", 'has', 'its', 'how', "don't", "hadn't", 'just', "we're", 'him', 'himself', 'once', 'he', 'since', 'again', 'my', 'this', "we've", 'off', 'would', 'not', 'these', 'your', 'therefore', 'themselves', 'our', 'is', 'between', 'ever', "won't", "he'll", 'i', 'do', "let's", 'with', "hasn't", 'k', 'otherwise', "doesn't", 'we', 'by', 'itself', 'shall', 'be', "shouldn't", 'below', 'during', 'myself', 'nor', 'what', 'other', 'there', 'yourself', 'a', 'mightn', 'mustn', 't', "mightn't", 'haven', 'don', 'isn', 's', 'wouldn', 'will', 'needn', 'hadn', 'll', 'couldn', 'aren', 'wasn', 'ain', 'm', 'didn', 'won', "that'll", "should've", "needn't", 're', 'o', 'doesn', 've', 'y', 'ma', 'now', 'd', 'shan', 'shouldn', 'hasn', 'weren']
# GET TRAIN DATA
# data info
# We are using labelled word for train model
# this data contains 160154 negative,90710 positive words
def get_train_data():
	# first 35 lines for data info
	for word in open('positive_words.txt','r',encoding='ISO-8859-1').readlines()[35:]:
		word = word.replace('\n', '').lower()
		# if word in positive_words_counts:
		# 	positive_words_counts[word] += 1
		# else:
		# 	positive_words_counts[word] = 1
		positive_words.append([word.rstrip(),1.0])
	for word in open('negative_words.txt', 'r', encoding = "ISO-8859-1").readlines()[35:]:
		word = word.replace('\n', '').lower()
		# if word in negative_words_counts:
		# 	negative_words_counts[word] += 1
		# else:
		# 	negative_words_counts[word] = 1
		negative_words.append([word.rstrip(),0.0])
	# data = positive_words + negative_words

		# [print(row) for row in csv_reader]
#Veri setini train,test ayrımı
def split_data(data):
    trainSize = int(len(data) * 0.7)
    trainSet = []
    copy = list(data)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    # print(trainSet)
    return [trainSet, copy]
def seperateByClass(data):
    seperated = {}
    for i in range(len(data)):
        vector = data[i]
        # print('vector[-1]',vector[-1])
        if (vector[-1] not in seperated):
            seperated[vector[-1]] = []
        seperated[vector[-1]].append(vector)
    # print(seperated)
    return seperated

# def get():
# 	for word in trainingSet:
# 		if word[1] == 1.0:
# 			if word[0] in positive_words_counts:
# 				positive_words_counts[word[0]] += 1
# 			else:
# 				positive_words_counts[word[0]] = 1
# 		else:
# 			if word[0] in negative_words_counts:
# 				negative_words_counts[word[0]] += 1
# 			else:
# 				negative_words_counts[word[0]] = 1

# TRAIN MODEL
# we are using BernoulliNB of Naive Bayes Classification algorithm for train model
def get_result():
	test_data = testSet
	# print(test_data)
	True_Pos = 0
	True_Neg = 0
	results = []
	# print(test_data)
	for word in test_data:
		# print(word[0])
		if word[1] == 1.0:
			# print(word[0])
			if word[0] in positive_words:
				if word[0] in negative_words:
					# print(word[0])
					result = ((positive_words_counts[word[0]]/len(positive_words))*(len(positive_words)/(len(positive_words)+len(negative_words))))/((positive_words_counts[word[0]]+negative_words_counts[word[0]])/(len(positive_words)+len(negative_words)))
					# print(result)
				else:
					result = ((positive_words_counts[word[0]]/len(positive_words))*(len(positive_words)/(len(positive_words)+len(negative_words))))/((positive_words_counts[word[0]]+0)/(len(positive_words)+len(negative_words)))
				# print(result)
				if float(result)>0.7:
					# print(result)
					True_Pos += 1
					# print(True_Pos)
				else:
					True_Neg += 1
		elif word[1] == 1.0:
			if word[0] in negative_words_counts:
				result = None
				if word[0] in positive_words:
					# print(word[0])
					result = ((negative_words_counts[word[0]]/negative_words)*(len(negative_words)/(len(positive_words)+len(negative_words))))/((positive_words_counts[word[0]]+negtative_words_counts[word[0]])/(len(positive_words)+len(negative_words)))
				else:
					result = ((negative_words_counts[word[0]]/negative_words)*(len(negative_words)/(len(positive_words)+len(negative_words))))/((0+negtative_words_counts[word[0]])/(len(positive_words)+len(negative_words)))

				# print(result)
				if float(result)>0.7:
					# print(result)
					True_Neg += 1
				else:
					True_Pos += 1
					# print(True_Neg)
	print(True_Pos, True_Neg)
	accuracy = (True_Pos+True_Neg)/(len(positive_words)+len(negative_words))
	print(accuracy)
	# return accuracy


# def get_features():
# 	P_positive = len(positive_words)/(len(positive_words)+len(negative_words))
# 	P_negative = len(negative_words)/(len(positive_words)+len(negative_words))
# 	print(P_positive, P_negative)

# get_test_data('Sentiment.csv')
if __name__ == '__main__':
	positive_words = []
	negative_words = []
	positive_words_counts = dict()
	negative_words_counts = dict()
	get_train_data()
	data = positive_words + negative_words
	trainingSet, testSet = split_data(data)
	print('Split {0} row into train = {1} and test {2} rows'.format(len(data),len(trainingSet),len(testSet)))