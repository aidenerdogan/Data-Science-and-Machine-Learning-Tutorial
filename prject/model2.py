
positive_words = []
negative_words = []
for word in open('positive_words.txt', 'r',encoding = "ISO-8859-1").readlines()[35:]:
  positive_words.append(({word.rstrip(): True}, 'positive'))

for word in open('negative_words.txt', 'r', encoding = "ISO-8859-1").readlines()[35:]:
  negative_words.append(({word.rstrip(): True}, 'negative'))

words_with_sentiment = positive_words + negative_words

# print (words_with_sentiment)

all_words = []
for words, sentiment in words_with_sentiment:
    all_words.extend(words)

import nltk

fd = nltk.FreqDist(all_words)
word_features = fd.keys()

# print (word_features)

# Extract Features


def extract_features(document):
    unique_words_in_document = set(document)
    features = {}
    for word_feature in word_features:
        features['contains(%s)' % word_feature] = (word_feature in unique_words_in_document)
    return features


training_set = nltk.classify.apply_features(extract_features, words_with_sentiment)

print (training_set)

classifier = nltk.classify.NaiveBayesClassifier.train(training_set)

classifier.show_most_informative_features(10)

test_word = "#RajaNatwarlal is a pathetic movie"

features_test_word = extract_features(test_word.split())
print (features_test_word)
print (classifier.classify(features_test_word))
