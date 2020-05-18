# for geting data
import pandas as pd
# manipulation on data
import numpy as np
# plot somthing
import matplotlib.pyplot as plt
# clean data
import re
# use NLP lib
import nltk
"===get data==="
# read data
airline_tweets = pd.read_csv('usairline.csv')
# or
# data_source_url = "https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv"
# airline_tweets = pd.read_csv(data_source_url)
# check data
# print(airline_tweets.head())
"===plot data==="
# plot_size = plt.rcParams["figure.figsize"]
# print(plot_size[0])
# print(plot_size[1])
# plot_size[0] = 8
# plot_size[1] = 6
# plt.rcParams["figure.figsize"] = plot_size
# # plot with pie
# airline_tweets.airline.value_counts().plot(kind='pie', autopct='%1.0f%%')
# # plt.show()
# airline_tweets.airline_sentiment.value_counts().plot(kind='pie', autopct='%2.0f%%', colors=['yellow','green','black'])
# plt.show()
# groupby 'airline_sentiment','airline' or 'airline','airline_sentiment'
# airline_sentiment = airline_tweets.groupby(['airline','airline_sentiment']).airline_sentiment.count().unstack()
# airline_sentiment.plot(kind='bar')
# plt.show()
# for avarage and different plot use seaborn
# import seaborn as sns
# sns.barplot(x='airline_sentiment', y='airline_sentiment_confidence', data=airline_tweets)
# plt.show()
"===clean data==="
features = airline_tweets.iloc[:, 10]
labels = airline_tweets.iloc[:, 1]
processed_features = []

for sentence in range(0, len(features)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    processed_features.append(processed_feature)
# print(processed_features[:5])
"===create a model==="
# Representing Text in Numeric Form
# Bag of Words
# TF-IDF
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(processed_features).toarray()
# print(processed_features[:5])
# MODEL
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestClassifier

text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train, y_train)
predictions = text_classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))
# source: https://stackabuse.com/python-for-nlp-sentiment-analysis-with-scikit-learn/