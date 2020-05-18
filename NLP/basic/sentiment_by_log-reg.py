
# about data: data is 50k text about IMDB movies,25k test, 25k train and in train 12.5k positive, 12.5k negative data
# get data from text
train_data = [line.strip() for line in open('movie_data/full_train.txt','r')]
test_data = [line.strip() for line in open('movie_data/full_test.txt', 'r')]
# print(test_data)

# cleaning data
import re
REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
def clean(text):
	text = [REPLACE_NO_SPACE.sub("", line.lower()) for line in text]
	text = [REPLACE_WITH_SPACE.sub(" ", line) for line in text]
	return text
train_clean = clean(train_data)
test_clean = clean(test_data)
# print(test_clean)

# vectorization
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(binary=True)
cv.fit(train_clean)
X = cv.transform(train_clean)
X_test = cv.transform(test_clean)
# print(X_test)

# classification building
from sklearn.linear_model import LogisticRegression
from	sklearn.metrics import accuracy_score
from	sklearn.model_selection import train_test_split

target = [1 if i < 12500 else 0 for i in range(25000)]
X_train,X_val, y_train, y_val =train_test_split(X, target, train_size = 0.75)
for c in [0.01,0.05,0.25,0.5,1]:
	lr = LogisticRegression(C=c)
	lr.fit(X_train,y_train)
	# print("Accuracy for C=%s: %s" %(c, accuracy_score(y_val,lr.predict(X_val))))
# train final model
final_model = LogisticRegression(C=0.05)
final_model.fit(X,target)
print('Final Accuary: %s' % accuracy_score(target, final_model.predict(X_test)))

feature_to_coef = {word: coef for word, coef in zip(cv.get_feature_names(), final_model.coef_[0])}
for best_positive in sorted(feature_to_coef.items(), key=lambda x:x[1],reverse=True)[:5]:
	print(best_positive)

for best_ngative in sorted(feature_to_coef.items(), key=lambda x:x[1])[:5]:
	print(best_ngative)
SOURCE = https://towardsdatascience.com/sentiment-analysis-with-python-part-1-5ce197074184

# "===========PART 2=========="
# from nltk.corpus import stopwords
# # removal stopwords
# english_stop_words = stopwords.words('english')
# def remove_stop_words(corpus):
#     removed_stop_words = []
#     for review in corpus:
#         removed_stop_words.append(
#             ' '.join([word for word in review.split() 
#                       if word not in english_stop_words])
#         )
#     return removed_stop_words
# print(len(train_clean[0]))
# no_stop_words = remove_stop_words(train_clean)
# print(len(no_stop_words[0]))
# # stemmimg
# def get_stemmed_text(corpus):
#     from nltk.stem.porter import PorterStemmer
#     stemmer = PorterStemmer()
#     return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]

# stemmed_reviews = get_stemmed_text(train_clean)
# print(len(stemmed_reviews[0]))
# # lemmatization
# def get_lemmatized_text(corpus):
#     from nltk.stem import WordNetLemmatizer
#     lemmatizer = WordNetLemmatizer()
#     return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

# lemmatized_reviews = get_lemmatized_text(train_clean)
# print(len(lemmatized_reviews[0]))

# didn't finish, i'll continue on it
# source:https://towardsdatascience.com/sentiment-analysis-with-python-part-2-4f71e7bde59a
# what is this Logistic Regression : https://blog.quantinsti.com/machine-learning-logistic-regression-python/