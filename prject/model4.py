from sklearn.feature_extraction.text import CountVectorizer
# what is difference between nltk NB and sklearn NB(BNB,MNB,GNB)
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt

# GET TRAIN DATA
# data info
# We are using labelled word for train model
# this data contains 160154 negative,90710 positive words
def get_train_data():
	positive_words = []
	negative_words = []
	# first 35 lines for data info
	for word in open('positive_words.txt','r',encoding='ISO-8859-1').readlines()[35:]:
		positive_words.append([word.rstrip(),1])
	for word in open('negative_words.txt', 'r', encoding = "ISO-8859-1").readlines()[35:]:
		negative_words.append([word.rstrip(),0])
	data = positive_words + negative_words
	return data

# TRAIN MODEL
# we are using BernoulliNB of Naive Bayes Classification algorithm for train model
def training_model(data, word_vectorizer):
		print('===Trainining Model...===')
		# train_text is data['text']
		train_text = [data[0] for data in data]
		# train_sentiment mean data sentiment class
		train_sentiment =[data[1] for data in data]
		train_text = word_vectorizer.fit_transform(train_text)
		print('===Train Finished===')
		return BernoulliNB().fit(train_text,train_sentiment)
word_vectorizer = CountVectorizer(binary = 'true')
Bernoulli_classifier = training_model(get_train_data(), word_vectorizer)
result = Bernoulli_classifier.predict(word_vectorizer.transform(["I like this tv shows!"]))
print('---Test of Vectorizer---\n',result[0])
def text_analysis(Bernoulli_classifier, word_vectorizer, text):
    return text, Bernoulli_classifier.predict(word_vectorizer.transform([text]))

new_result = text_analysis(Bernoulli_classifier, word_vectorizer, "Best brand ever")
new_result
def get_result(result):
    text, analysis_result = result
    print_text = "Positive" if analysis_result[0] == 1 else "Negative"
    print(text, ":", print_text)
print('===Sentiment Text of Examples===')    
get_result(new_result) 
get_result( text_analysis(Bernoulli_classifier, word_vectorizer,"this is the best film"))
get_result( text_analysis(Bernoulli_classifier, word_vectorizer,"this is the worst film"))
get_result( text_analysis(Bernoulli_classifier, word_vectorizer,"awesome!"))
get_result( text_analysis(Bernoulli_classifier, word_vectorizer,"5/20"))
get_result( text_analysis(Bernoulli_classifier, word_vectorizer,"to much bad"))
get_result( text_analysis(Bernoulli_classifier, word_vectorizer,"nice girl"))
stopwords = ["wouldn't", 'own', 'ours', 'under', "who's", 'also', "here's", 'could', 'have', 'but', 'com', 'ourselves', "where's", 'at', "they'd", 'hence', 'doing', "you're", "how's", 'r', 'yours', 'against', 'same', "i'd", 'which', 'more', 'very', 'whom', 'ought', 'http', 'for', 'was', 'most', 'of', 'his', 'are', 'so', 'did', 'am', "she'll", 'been', 'up', 'does', 'after', 'few', 'until', "they're", 'it', 'both', 'their', 'the', "there's", "can't", 'into', "he's", 'they', 'through', "why's", 'get', 'all', 'them', 'herself', "we'll", 'if', 'can', 'out', "i'm", 'and', 'because', "it's", "they've", "isn't", 'were', 'some', 'than', "wasn't", 'too', "that's", "they'll", 'that', "haven't", 'such', "i've", 'while', 'why', 'like', 'should', 'to', "you'll", 'hers', 'you', 'then', 'however', 'down', 'before', "she's", 'any', "what's", "he'd", 'here', 'www', "didn't", "you've", 'cannot', 'above', 'her', 'from', "shan't", "you'd", 'yourselves', 'an', "aren't", 'only', 'theirs', "we'd", 'on', 'having', 'each', 'else', 'who', 'or', "i'll", 'had', 'as', "mustn't", "couldn't", "weren't", 'me', 'further', 'over', 'where', 'when', 'no', 'being', "she'd", 'about', 'those', 'in', 'she', "when's", 'has', 'its', 'how', "don't", "hadn't", 'just', "we're", 'him', 'himself', 'once', 'he', 'since', 'again', 'my', 'this', "we've", 'off', 'would', 'not', 'these', 'your', 'therefore', 'themselves', 'our', 'is', 'between', 'ever', "won't", "he'll", 'i', 'do', "let's", 'with', "hasn't", 'k', 'otherwise', "doesn't", 'we', 'by', 'itself', 'shall', 'be', "shouldn't", 'below', 'during', 'myself', 'nor', 'what', 'other', 'there', 'yourself', 'a', 'mightn', 'mustn', 't', "mightn't", 'haven', 'don', 'isn', 's', 'wouldn', 'will', 'needn', 'hadn', 'll', 'couldn', 'aren', 'wasn', 'ain', 'm', 'didn', 'won', "that'll", "should've", "needn't", 're', 'o', 'doesn', 've', 'y', 'ma', 'now', 'd', 'shan', 'shouldn', 'hasn', 'weren']
print('===Testing Model...===')
import re,csv
def get_evaluation_data(file):
    evaluation_data = []
    with open(file,'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            words = " ".join([word for word in (re.findall(r"[\w']+|[.,!?;]", row['text'].rstrip())) if len(word) >= 3 and word not in stopwords])
            if row['sentiment'] == 'Positive':
                evaluation_data.append([words,1])
            else:
                evaluation_data.append([words,0])
    #         # print(words)
    #         if row['sentiment'] == 'Negative':
    #             # print(words)
    #             [neg_file.write(word+'\n') for word in words]
    #             elif row['sentiment'] == 'Positive':
    #                 [pos_file.write(word+'\n') for word in words
    return evaluation_data
# print(get_evaluation_data('Sentiment.csv'))
def simple_evaluation(evaluation_data):
    evaluation_text     = [evaluation_data[0] for evaluation_data in evaluation_data]
    evaluation_result   = [evaluation_data[1] for evaluation_data in evaluation_data]

    total = len(evaluation_text)
    corrects = 0
    for index in range(0, total):
        analysis_result = text_analysis(Bernoulli_classifier, word_vectorizer, evaluation_text[index])
        text, result = analysis_result
        corrects += 1 if result[0] == evaluation_result[index] else 0

    return corrects * 100 / total

simple_evaluation(get_evaluation_data('Sentiment.csv'))
def create_confusion_matrix(evaluation_data):
    evaluation_text     = [evaluation_data[0] for evaluation_data in evaluation_data]
    actual_result       = [evaluation_data[1] for evaluation_data in evaluation_data]
    prediction_result   = []
    for text in evaluation_text:
        analysis_result = text_analysis(Bernoulli_classifier, word_vectorizer, text)
        prediction_result.append(analysis_result[1][0])
    
    matrix = confusion_matrix(actual_result, prediction_result)
    return matrix
    
confusion_matrix_result = create_confusion_matrix(get_evaluation_data('Sentiment.csv'))
print('===Confusion Matrix===')
print(pd.DataFrame(confusion_matrix_result, columns=["Negatives", "Positives"],index=["Negatives", "Positives"]))
print('===Plotting===')
classes = ["Negatives", "Positives"]

plt.figure()
plt.imshow(confusion_matrix_result, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Sentiment Analysis")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

text_format = 'd'
thresh = confusion_matrix_result.max() / 2.
for row, column in itertools.product(range(confusion_matrix_result.shape[0]), range(confusion_matrix_result.shape[1])):
    plt.text(column, row, format(confusion_matrix_result[row, column], text_format),
             horizontalalignment="center",
             color="white" if confusion_matrix_result[row, column] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

TN_true_negatives = confusion_matrix_result[0][0]
FN_false_negatives = confusion_matrix_result[0][1]
FP_false_positives = confusion_matrix_result[1][0]
TP_true_positives = confusion_matrix_result[1][1]

accuracy = (TP_true_positives + TN_true_negatives) / (TP_true_positives + TN_true_negatives + FP_false_positives + FN_false_negatives)
precision = TP_true_positives / (TP_true_positives + FP_false_positives)
recall_sensitivity = TP_true_positives / (TP_true_positives + FN_false_negatives)
specificity = TN_true_negatives/(TP_true_positives+FN_false_negatives)
f1_score = 2*(recall_sensitivity * precision) / (recall_sensitivity + precision)
print('===Resutls===')
print('Accuracy:',accuracy)
print('Precision:',precision)
print('Recall_Sensitivity:',recall_sensitivity)
print('F1 Score:',f1_score)
print('Specificity',specificity)
