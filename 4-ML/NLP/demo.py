def segmentation(text):
	from nltk.tokenize import sent_tokenize
	import pandas as pd
	sentences = sent_tokenize(text)
	# # for i in sentences:
	# # 	print(i)
	# df = pd.DataFrame(sentences,columns=['sentence'])
	# # print(df['sentence'])
	# print('Your Text\n',text)
	# print(df['sentence'].str.split().head())
	# print(df['sentence'].str.split(expand=True).unstack().head())
	return sentences

def tokenize(text):
	import nltk
	tokens = []
	sentences = segmentation(text)
	for row in sentences:
		tokens.append(nltk.word_tokenize(row))
	print('Tokens\n',tokens,'\n')
	# print('Tokens len is {0}'.format(len(tokens)))
	return tokens
def identify_stopwords(text):
	from wordcloud import STOPWORDS
	stopwords = set(STOPWORDS)
	tokens = tokenize(text)
	sentences = []
	for token in tokens:
		sentences.append(list(set(token)-stopwords))
		# print(len(list(set(token)-stopwords)))
	print(sentences)
	# print('After identify stopwords len is {0}'.format(len(sentences)))
	return sentences
# where we remove word affixes to get to the base form of a word
def stemming(text):
	# stemming with nltk
	import nltk
	# snowball_stemmer_words = []
	snowball_stemmer = nltk.stem.SnowballStemmer('english')
	sentences = identify_stopwords(text)
	for words in sentences:
		# snowball_stemmer_words.append(snowball_stemmer.stem(word))
		print([snowball_stemmer.stem(word) for word in words])
# Where the root word is always a lexicographically correct word (present in the dictionary), but the root stem may not be so
def lemmatizer(text):
	# # lemmatizer with nltk
	# from nltk.stem import WordNetLemmatizer
	# lemmatizer = WordNetLemmatizer()
	# sentences = identify_stopwords(text)
	# lst =['a','n','v']
	# for words in sentences:
	# 	[print(word,':',lemmatizer.lemmatize(word,pos='v')) for word in words]
	# lemmatizer with spacy-1
	import spacy
	nlp = spacy.load('en_core_web_sm')
	text = nlp(text)
	# for word in doc:
	# 	print(word,':',word.lemma_)
	# lemmatizer-2
	text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
	print(text)

# word frequency-1
def vectorizer(text):
	from sklearn.feature_extraction.text import CountVectorizer
	sentences = segmentation(text)
	vector = CountVectorizer(min_df=0)
	vector_sentences = vector.fit_transform(sentences)
	print(vector_sentences)
	print(vector.get_feature_names())
	print()
	print(vector_sentences.toarray())
# Word freq-2 by Tf-IDF-TFIDF
# The final calculation of the Term Frequency(TF)-Inverse Doc Freq(IDF) is simply the multiplication of the TF and IDF terms: TF * IDF.
def TFIDF(text):
	from sklearn.feature_extraction.text import TfidfVectorizer
	import pandas as pd
	sentences = segmentation(text)
	vectorizer = TfidfVectorizer()
	vectors = vectorizer.fit_transform(sentences)
	feature_names = vectorizer.get_feature_names()
	dense_vec = vectors.todense()
	dense_list = dense_vec.tolist()
	tfidf_data = pd.DataFrame(dense_list, columns=feature_names)
	print(tfidf_data)
# Part of speech or POS is a grammatical role that explains how a particular word is used in a sentence.
def part_of_speech(text):
	import nltk
	for sentence in identify_stopwords(text):
		tagged = nltk.pos_tag(sentence)
		print(tagged)
# Dependency parsing is the process of extracting the dependency parse of a sentence to represent its grammatical structure. 
def dependency(text):
	import spacy
	nlp = spacy.load('en_core_web_sm')
	piano_doc = nlp(text)
	for token in piano_doc:
		print (token.text, token.tag_, token.head.text, token.dep_)
	from spacy import displacy
	displacy.serve(piano_doc, style='dep')
# Named Entity recognation (phycial names)
def NER(text):
	import spacy
	nlp = spacy.load('en_core_web_sm')
	piano_class_doc = nlp(text)
	for ent in piano_class_doc.ents:
		print(ent.text, ent.start_char, ent.end_char, ent.label_, spacy.explain(ent.label_))
	from spacy import displacy
	displacy.serve(piano_class_doc, style='ent')
def coreference(text):
	import spacy
	nlp = spacy.load('en')
	print('All cluster mentions')
	doc = nlp(text)
	import neuralcoref
	# print(doc._.has_coref)
	print(doc._.coref_clusters)	
	# for i in doc._.coref_cluster:
	# 	print(i.mentions)
# frequently-mentioned noun chunks in text/doc
# doc/test summarization with unsupervised learning techn	iqs
def summarization(text):
	from nltk.corpus import stopwords
	from nltk.stem import PorterStemmer
	from nltk.tokenize import word_tokenize, sent_tokenize
	# create frequence table
	freq_table = dict()
	stopwords = set(stopwords.words('english'))
	words = word_tokenize(text)
	ps = PorterStemmer()
	for word in words:
		word = ps.stem(word)
		if word in stopwords:
			continue
		if word in freq_table:
			freq_table[word] += 1
		else:
			freq_table[word] = 1
		# split to sentences
		sentences = sent_tokenize(text)
		# score sentences
		sentence_value = dict()
		for sentence in sentences:
			word_coun_in_sent = (len(word_tokenize(sentence)))
			for word_value in freq_table:
				if word_value in sentence.lower():
					if sentence[:10] in sentence_value:
						sentence_value[sentence[:10]] += freq_table[word_value]
					else:
						sentence_value[sentence[:10]] = freq_table[word_value]
		# find avarage score
		sum_values = 0
		for entry in sentence_value:
			sum_values += sentence_value[entry]
		avarage = int(sum_values / len(sentence_value))
		# generate summary
		sentence_count = 0
		summary = ''
		for sentence in sentences:
			if sentence[:10] in sentence_value and sentence_value[sentence[:10]] > (1.5*avarage):
				summary += ' '+ sentence
				sentence_count +=1
		print(summary)


def key_terms(text):
	import spacy
	import textacy.extract
	# Load the large English NLP model
	nlp = spacy.load('en_core_web_lg')
	# Parse the document with spaCy
	doc = nlp(text)
	# Extract noun chunks that appear
	noun_chunks = textacy.extract.noun_chunks(doc, min_freq=3)

	# Convert noun chunks to lowercase strings
	noun_chunks = map(str, noun_chunks)
	noun_chunks = map(str.lower, noun_chunks)

	# Print out any nouns that are at least 2 words long
	for noun_chunk in set(noun_chunks):
	    if len(noun_chunk.split(" ")) > 1:
	        print(noun_chunk)

if __name__=="__main__":
	text = input("enter at least 500 characters :")
	while True:
		if(len(text)>500):
			break
		else:
			text = input('Your text less than 500 ch, pls try again.')
	# print("----------------Word Tokinizer----------------")
	# tokenize(text)
	# print("----------------Identify Stopwords----------------")
	# identify_stopwords(text)
	print("----------------Stemming (Snowball-porter2)----------------")
	stemming(text)
	print("----------------Lemmaztization----------------")
	lemmatizer(text)
	# print("----------------Word Vectorization----------------")
	# vectorizer(text)
	# print("----------------Frequency(TF-IDF-TFIDF)----------------")
	# TFIDF(text)
	# print("----------------Part of Speech----------------")
	# part_of_speech(text)
	# print("----------------Dependency----------------")
	# dependency(text)
	# print("----------------Named Entity Recognation----------------")
	# NER(text)
	# print("----------------CoreFerence----------------")
	# coreference(text)
	# print("----------------Summarization----------------")
	# summarization(text)