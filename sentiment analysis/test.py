# dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

"""About data:
Data has sentiment(1 means positive 0 means negative)
and text(from twitter) and 100000 rows.
"""
# Read Data
my_data = pd.read_csv('clean_tweet2.csv',index_col=0)
# check data
# print(my_data)

# DATA CLEANNING
print('\n=== CLEANNING DATA ===')
print(my_data[my_data.isnull().any(axis=1)].head())
print(np.sum(my_data.isnull().any(axis=1)))
my_data.dropna(inplace=True)
my_data.reset_index(drop=True,inplace=True)
my_data.info()

train_data = my_data[:80000]
test_data = my_data[80000:]
print(train_data)
print(test_data)

from nltk import classify
from nltk import NaiveBayesClassifier
classifier = NaiveBayesClassifier.train(train_data)

print("Accuracy is:", classify.accuracy(classifier, test_data))

print(classifier.show_most_informative_features(10))
