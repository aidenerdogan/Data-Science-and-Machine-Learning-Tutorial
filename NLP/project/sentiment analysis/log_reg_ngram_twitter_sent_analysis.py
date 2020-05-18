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
my_data = pd.read_csv('clean_tweets.csv',index_col=0)
# check data
# print(my_data)

# DATA CLEANNING
print('\n=== CLEANNING DATA ===')
print(my_data[my_data.isnull().any(axis=1)].head())
print(np.sum(my_data.isnull().any(axis=1)))
my_data.dropna(inplace=True)
my_data.reset_index(drop=True,inplace=True)
my_data.info()
x = my_data.text
y = my_data.sentiment

# model dependencies
from sklearn.model_selection import train_test_split
SEED = 200
x_train,x_validation_and_test, y_train, y_validation_and_test = train_test_split(
    x, y, test_size=0.02, random_state=SEED)
x_validation, x_test, y_validation, y_test = train_test_split(
    x_validation_and_test,y_validation_and_test, test_size=.5, random_state=SEED)
print('\n=== ABOUT DATA OF MODEL ===')
print ("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_train),
                                                                             (len(x_train[y_train == 0]) / (len(x_train)*1.))*100,
                                                                            (len(x_train[y_train == 1]) / (len(x_train)*1.))*100))
print ("Validation set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_validation),
                                                                             (len(x_validation[y_validation == 0]) / (len(x_validation)*1.))*100,
                                                                            (len(x_validation[y_validation == 1]) / (len(x_validation)*1.))*100))
print ("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_test),
                                                                             (len(x_test[y_test == 0]) / (len(x_test)*1.))*100,
                                                                            (len(x_test[y_test == 1]) / (len(x_test)*1.))*100))

def train_test_and_evaluate(pipeline, x_train, y_train, x_test, y_test):
    if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.5:
        null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
    else:
        null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
    sentiment_fit = pipeline.fit(x_train, y_train)
    y_pred = sentiment_fit.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    conmat = np.array(confusion_matrix(y_test, y_pred, labels=[0,1]))
    confusion = pd.DataFrame(conmat, index=['negative', 'positive'],
                         columns=['predicted_negative','predicted_positive'])
    print ("null accuracy: {0:.2f}%".format(null_accuracy*100))
    print ("accuracy score: {0:.2f}%".format(accuracy*100))
    if accuracy > null_accuracy:
        print ("model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100))
    elif accuracy == null_accuracy:
        print ("model has the same accuracy with the null accuracy")
    else:
        print ("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100))
    print ("-"*80)
    print ("Confusion Matrix\n")
    print (confusion)
    print ("-"*80)
    print ("Classification Report\n")
    print (classification_report(y_test, y_pred, target_names=['negative','positive']))
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
lr = LogisticRegression()

# print('\n========== UNI_GRAM ==========')

# ug_cvec = CountVectorizer(max_features=80000)
# ug_pipeline = Pipeline([
#         ('vectorizer', ug_cvec),
#         ('classifier', lr)
#     ])
# train_test_and_evaluate(ug_pipeline, x_train, y_train, x_validation, y_validation)

print('\n========== BI_GRAM ==========')

bg_cvec = CountVectorizer(max_features=80000,ngram_range=(1, 2))
bg_pipeline = Pipeline([
        ('vectorizer', bg_cvec),
        ('classifier', lr)
    ])
train_test_and_evaluate(bg_pipeline, x_train, y_train, x_validation, y_validation)

# print('\n========== TRI_GRAM ==========')

# tg_cvec = CountVectorizer(max_features=80000,ngram_range=(1, 3))
# tg_pipeline = Pipeline([
#         ('vectorizer', tg_cvec),
#         ('classifier', lr)
#     ])
# train_test_and_evaluate(tg_pipeline, x_train, y_train, x_validation, y_validation)
