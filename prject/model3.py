import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split # function for splitting data to train and test sets
import random
import nltk
import re, csv

def get_data():
    data = pd.read_csv('Sentiment.csv')
    # Keeping only the neccessary columns
    data = data[['text','sentiment']]
    # Splitting the dataset into train and test set
    return data
    # with open('Sentiment.csv','r') as csv_file:
    #     csv_reader = csv.DictReader(csv_file)
    #     for row in csv_reader:
    #         yield row

def define_datas():
    # define train data
    train,test = get_data()
    train_positive = train[ train['sentiment'] == 'Positive']
    train_positive = train_positive['text']
    train_negative = train[ train['sentiment'] == 'Negative']
    train_negative = train_negative['text']
    # define test data
    test_positive = test[ test['sentiment'] == 'Positive']
    test_positive = test_positive['text']
    test_negative = test[ test['sentiment'] == 'Negative']
    test_negative = test_negative['text']
    return test_positive, test_negative

stopwords = ["wouldn't", 'own', 'ours', 'under', "who's", 'also', "here's", 'could', 'have', 'but', 'com', 'ourselves', "where's", 'at', "they'd", 'hence', 'doing', "you're", "how's", 'r', 'yours', 'against', 'same', "i'd", 'which', 'more', 'very', 'whom', 'ought', 'http', 'for', 'was', 'most', 'of', 'his', 'are', 'so', 'did', 'am', "she'll", 'been', 'up', 'does', 'after', 'few', 'until', "they're", 'it', 'both', 'their', 'the', "there's", "can't", 'into', "he's", 'they', 'through', "why's", 'get', 'all_words', 'them', 'herself', "we'll", 'if', 'can', 'out', "i'm", 'and', 'because', "it's", "they've", "isn't", 'were', 'some', 'than', "wasn't", 'too', "that's", "they'll", 'that', "haven't", 'such', "i've", 'while', 'why', 'like', 'should', 'to', "you'll", 'hers', 'you', 'then', 'however', 'down', 'before', "she's", 'any', "what's", "he'd", 'here', 'www', "didn't", "you've", 'cannot', 'above', 'her', 'from', "shan't", "you'd", 'yourselves', 'an', "aren't", 'only', 'theirs', "we'd", 'on', 'having', 'each', 'else', 'who', 'or', "i'll", 'had', 'as', "mustn't", "couldn't", "weren't", 'me', 'further', 'over', 'where', 'when', 'no', 'being', "she'd", 'about', 'those', 'in', 'she', "when's", 'has', 'its', 'how', "don't", "hadn't", 'just', "we're", 'him', 'himself', 'once', 'he', 'since', 'again', 'my', 'this', "we've", 'off', 'would', 'not', 'these', 'your', 'therefore', 'themselves', 'our', 'is', 'between', 'ever', "won't", "he'll", 'i', 'do', "let's", 'with', "hasn't", 'k', 'otherwise', "doesn't", 'we', 'by', 'itself', 'shall_words', 'be', "shouldn't", 'below', 'during', 'myself', 'nor', 'what', 'other', 'there', 'yourself', 'a', 'mightn', 'mustn', 't', "mightn't", 'haven', 'don', 'isn', 's', 'wouldn', 'will', 'needn', 'hadn', 'll', 'couldn', 'aren', 'wasn', 'ain', 'm', 'didn', 'won', "that'll", "should've", "needn't", 're', 'o', 'doesn', 've', 'y', 'ma', 'now', 'd', 'shan', 'shouldn', 'hasn', 'weren']
def clean_data():
    tweet_data = []
    for index, row in train.iterrows():
        words = re.findall(r"[\w']+|[.,!?;]", row[0].rstrip())
        words_filtered = [e.lower() for e in words if len(e) >= 3]
        words_cleaned = [word for word in words_filtered
            if 'http' not in word
            and not word.startswith('@')
            and not word.startswith('#')
            and word != 'RT']
        words_without_stopwords = [word for word in words_cleaned if not word in stopwords]
        tweet_data.append((words_without_stopwords, row[1]))
    return tweet_data

# Extracting word features
def get_words_in_tweet_data(tweet_data):
    all_words = []
    for (words, sentiment) in tweet_data:
        all_words.extend(words)
    return all_words

def get_features(word_list):
    word_list = nltk.FreqDist(word_list)
    features = word_list.keys()
    return features
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['contains(%s)' % word] = (word in document_words)
    return features
if __name__ == '__main__':
    # data = [row.split() for row in get_data()]
    data = get_data()
    train, test = train_test_split(data, test_size=0.2)
    # random.shuffle(data)
    # train = data[:int((len(data)+1)*.80)] #Remaining 80% to training set
    # test = data[int(len(data)*.80+1):] #Splits 20% data to test set
    # Removing neutral sentiments
    # train = train[train.sentiment != "Neutral"]
    w_features = get_features(get_words_in_tweet_data(clean_data()))
    # Training the Naive Bayes classifier
    training_set = nltk.classify.apply_features(extract_features,clean_data())
    # print(training_set)
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    test_word = input('pls input a twitt or a text :')
    features_test_word = extract_features(test_word.split())
    # print (features_test_word)
    print (classifier.classify(features_test_word))