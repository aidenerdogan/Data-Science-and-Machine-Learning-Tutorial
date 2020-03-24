import csv, re

stopwords = ["wouldn't", 'own', 'ours', 'under', "who's", 'also', "here's", 'could', 'have', 'but', 'com', 'ourselves', "where's", 'at', "they'd", 'hence', 'doing', "you're", "how's", 'r', 'yours', 'against', 'same', "i'd", 'which', 'more', 'very', 'whom', 'ought', 'http', 'for', 'was', 'most', 'of', 'his', 'are', 'so', 'did', 'am', "she'll", 'been', 'up', 'does', 'after', 'few', 'until', "they're", 'it', 'both', 'their', 'the', "there's", "can't", 'into', "he's", 'they', 'through', "why's", 'get', 'all', 'them', 'herself', "we'll", 'if', 'can', 'out', "i'm", 'and', 'because', "it's", "they've", "isn't", 'were', 'some', 'than', "wasn't", 'too', "that's", "they'll", 'that', "haven't", 'such', "i've", 'while', 'why', 'like', 'should', 'to', "you'll", 'hers', 'you', 'then', 'however', 'down', 'before', "she's", 'any', "what's", "he'd", 'here', 'www', "didn't", "you've", 'cannot', 'above', 'her', 'from', "shan't", "you'd", 'yourselves', 'an', "aren't", 'only', 'theirs', "we'd", 'on', 'having', 'each', 'else', 'who', 'or', "i'll", 'had', 'as', "mustn't", "couldn't", "weren't", 'me', 'further', 'over', 'where', 'when', 'no', 'being', "she'd", 'about', 'those', 'in', 'she', "when's", 'has', 'its', 'how', "don't", "hadn't", 'just', "we're", 'him', 'himself', 'once', 'he', 'since', 'again', 'my', 'this', "we've", 'off', 'would', 'not', 'these', 'your', 'therefore', 'themselves', 'our', 'is', 'between', 'ever', "won't", "he'll", 'i', 'do', "let's", 'with', "hasn't", 'k', 'otherwise', "doesn't", 'we', 'by', 'itself', 'shall', 'be', "shouldn't", 'below', 'during', 'myself', 'nor', 'what', 'other', 'there', 'yourself', 'a', 'mightn', 'mustn', 't', "mightn't", 'haven', 'don', 'isn', 's', 'wouldn', 'will', 'needn', 'hadn', 'll', 'couldn', 'aren', 'wasn', 'ain', 'm', 'didn', 'won', "that'll", "should've", "needn't", 're', 'o', 'doesn', 've', 'y', 'ma', 'now', 'd', 'shan', 'shouldn', 'hasn', 'weren']
def insert_from_csv(file):
	neg_file = open('negative_words.txt','a+')
	pos_file = open('positive_words.txt','a+')
	neg_file2 = open('negative_words2.txt','a+')
	pos_file2 = open('positive_words2.txt','a+')
	with open(file,'r',encoding='ISO-8859-1') as csv_file:
		csv_reader = csv.DictReader(csv_file)
		for row in csv_reader:
			words = [word for word in (re.findall(r"[\w']+|[.,!?;]", row['Text'].rstrip())) if len(word) >= 3 and word not in stopwords]
			# print(words)
			if row['Sentiment'] == '0':
				# print(words)
				[neg_file.write(word+'\n') for word in words]
				[neg_file2.write(word+'\n') for word in words]
			elif row['Sentiment'] == '1':
				[pos_file.write(word+'\n') for word in words]
				[pos_file2.write(word+'\n') for word in words]
	neg_file.close()
	pos_file.close()
	neg_file2.close()
	pos_file2.close()


insert_from_csv('train.csv')
# # insert_from_csv('imdb_labelled.csv')
# # insert_from_csv('yelp_labelled.csv')


# def insert_from_txt():
# 	neg_file = open('negative_words.txt','a+')
# 	pos_file = open('positive_words.txt','a+')
# 	pos_text_file = open('positive_words2.txt', encoding = "ISO-8859-1")
# 	for line  in pos_text_file.readlines():
# 		words = [word.lower() for word in (re.findall(r"[\w']+|[.,!?;]", line.rstrip())) if len(word) >= 3 and word not in stopwords and word not in pos_text_file]
# 		# print(words)
# 		[pos_file.write(word+'\n') for word in words]
# 	neg_text_file = open('negative_words2.txt', encoding = "ISO-8859-1")
# 	for line  in neg_text_file.readlines():
# 		words = [word.lower() for word in (re.findall(r"[\w']+|[.,!?;]", line.rstrip())) if len(word) >= 3 and word not in stopwords and word not in neg_text_file]
# 		# print(words)
# 		[neg_file.write(word+'\n') for word in words]
# 	neg_file.close()
# 	pos_file.close()
# 	neg_text_file.close()
# 	pos_text_file.close()

# # insert_from_txt()

# def insert_from_text_labelled(file):
# 	neg_file = open('negative_words.txt','a+')
# 	pos_file = open('positive_words.txt','a+')
# 	pos_text_file = open(file, encoding = "ISO-8859-1")
# 	for line  in pos_text_file.readlines():
# 		# print(line[-2:][0])
# 		words = [word for word in (re.findall(r"[\w']+|[.,!?;]", line.rstrip())) if len(word) >= 3 and word not in stopwords]
# 		# print(words)
# 		# print([word for word in words])
# 		# print(line[-2:][0])
# 		if line[-2:][0] == '0':
# 			# print(words)
# 			[neg_file.write(word.lower()+'\n') for word in words]
# 		elif line[1] == '1':
# 			# print(words)
# 			[pos_file.write(word.lower()+'\n') for word in words]
# 	neg_file.close()
# 	pos_file.close()
# insert_from_text_labelled('yelp_labelled.txt')