# import re for multiple splitting
import re

def segmentation(text):
	sentenceEnders = re.compile('[.!?] ')
	sentences = sentenceEnders.split(text)
	# sentences = re.split("[\.\!\?][\s]{1,2}",text)
	return sentences
def tokenize(text):
	# words = [re.sub(r'\w+',' ',sentence) for sentence in segmentation(text)]
	words = [re.split('[, .]',sentence) for sentence in segmentation(text)]
	# words = re.sub('['^A-Za-z0-9']+','',words)
	return words
def identify_stopwords(text):
	words = []
	stopwords = ["wouldn't", 'own', 'ours', 'under', "who's", 'also', "here's", 'could', 'have', 'but', 'com', 'ourselves', "where's", 'at', "they'd", 'hence', 'doing', "you're", "how's", 'r', 'yours', 'against', 'same', "i'd", 'which', 'more', 'very', 'whom', 'ought', 'http', 'for', 'was', 'most', 'of', 'his', 'are', 'so', 'did', 'am', "she'll", 'been', 'up', 'does', 'after', 'few', 'until', "they're", 'it', 'both', 'their', 'the', "there's", "can't", 'into', "he's", 'they', 'through', "why's", 'get', 'all', 'them', 'herself', "we'll", 'if', 'can', 'out', "i'm", 'and', 'because', "it's", "they've", "isn't", 'were', 'some', 'than', "wasn't", 'too', "that's", "they'll", 'that', "haven't", 'such', "i've", 'while', 'why', 'like', 'should', 'to', "you'll", 'hers', 'you', 'then', 'however', 'down', 'before', "she's", 'any', "what's", "he'd", 'here', 'www', "didn't", "you've", 'cannot', 'above', 'her', 'from', "shan't", "you'd", 'yourselves', 'an', "aren't", 'only', 'theirs', "we'd", 'on', 'having', 'each', 'else', 'who', 'or', "i'll", 'had', 'as', "mustn't", "couldn't", "weren't", 'me', 'further', 'over', 'where', 'when', 'no', 'being', "she'd", 'about', 'those', 'in', 'she', "when's", 'has', 'its', 'how', "don't", "hadn't", 'just', "we're", 'him', 'himself', 'once', 'he', 'since', 'again', 'my', 'this', "we've", 'off', 'would', 'not', 'these', 'your', 'therefore', 'themselves', 'our', 'is', 'between', 'ever', "won't", "he'll", 'i', 'do', "let's", 'with', "hasn't", 'k', 'otherwise', "doesn't", 'we', 'by', 'itself', 'shall', 'be', "shouldn't", 'below', 'during', 'myself', 'nor', 'what', 'other', 'there', 'yourself', 'a', 'mightn', 'mustn', 't', "mightn't", 'haven', 'don', 'isn', 's', 'wouldn', 'will', 'needn', 'hadn', 'll', 'couldn', 'aren', 'wasn', 'ain', 'm', 'didn', 'won', "that'll", "should've", "needn't", 're', 'o', 'doesn', 've', 'y', 'ma', 'now', 'd', 'shan', 'shouldn', 'hasn', 'weren']
	[[words.append(word) for word in sentence if word not in stopwords] for sentence in tokenize(text)]
	words = [re.sub('[^A-Za-z0-9]+','', word) for word in words]
	# [words.append(word) for word in tokenize(text) if word not in stopwords]
	return words
def get_stemlist():
	import csv
	with open('diffs.csv','r') as csv_file:
		dict_reader = csv.DictReader(csv_file)
		for voc in dict_reader:
			yield voc
def stemming(text):
	words = [word.lower() for word in identify_stopwords(text)]
	print(words)
	lst = []
	# vocs =  [row['voc'] for row in get_stemlist()]
	# print(vocs)
	for word in words:
		for voc in get_stemlist():
			if word == voc['voc']:
				words.remove(word)
				words.append(voc['output'])
	print(words)
		# words = [[voc['output'] for voc in dict_reader if voc['voc'] == word] for word in words]
if __name__=='__main__':
	text = input('Please inpiut a text :')
	stemming(text)