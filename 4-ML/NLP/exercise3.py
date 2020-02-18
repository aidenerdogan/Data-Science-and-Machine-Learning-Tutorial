# import nltk

# step-1 tokenization
# text = "My name is Ahmet and I am working on Data Science"
# tokens = nltk.word_tokenize(text)
# print(tokens)

# step-2 remove stopwords
# from nltk.corpus import stopwords
# stop_words = stopwords.words('english')
# removed_stopwords = [word for word in tokens if word not in stop_words]
# print(removed_stopwords)
# try this steps with I'm and I am

# step-3 stemming
# sbs = nltk.stem.SnowballStemmer('english')
# [print(sbs.stem(word)) for word in removed_stopwords]
# lst = ['cook','cooks','cooked','cooking']
# [print(sbs.stem(word)) for word in lst]

# step-4 word embedding
"-----"

# step-5 term frequency-inverse document frequency
# Term Frequency (TF): (string counts)/(all string count of doc)
# Inverse Document Frequency (IDF): log(N/DF)
# TF-IDF: TF*IDF
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# def get_tf_idf(vectorizer):
# 	feature_names = vectorizer.get_feature_names()
# 	dense_vec = vectors.todense()
# 	dense_list = dense_vec.tolist()
# 	tfidf_data = pd.DataFrame(dense_list, columns=feature_names)
# 	return tfidf_data
# vectorizer = TfidfVectorizer()
# doc_1 = "TF-IDF uses statistics to measure how important a word is to " \
#         "a particular document"
# doc_2 = "The TF-IDF is perfectly balanced, considering both local and global " \
#         "levels of statistics for the target word."
# doc_3 = "Words that occur more frequently in a document are weighted higher, " \
#         "but only if they're more rare within the whole document."
# documents_list = [doc_1,doc_2,doc_3]
# vectors = vectorizer.fit_transform(documents_list)
# tfidf_data = get_tf_idf(vectorizer)
# # print(tfidf_data)

# # step-6 topic modelling
# from sklearn.decomposition import LatenDirichletAllocation as LDA
# num_topics = 3
# # Here we create and fit the LDA model
# # The "document_word_matrix" is a 2D array where each row is a document
# # and each column is a word. The cells contain the count of the word within
# # each document
# lda = LDA(n_components=num_topics, n_jobs=-1)
# lda.fit(tfidf_data)

# step-7: sentiment analysis
