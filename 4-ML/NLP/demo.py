def segmentation(text):
	from nltk.tokenize import sent_tokenize
	import pandas as pd
	sentences = sent_tokenize(text)
	# for i in sentences:
	# 	print(i)
	df = pd.DataFrame(sentences,columns=['sentence'])
	# print(df['sentence'])
	print('Your Text\n',text)
	print(df['sentence'].str.split().head())
	print(df['sentence'].str.split(expand=True).unstack().head())

if __name__=="__main__":
	text = input("enter at least 500 characters :")
	stemming(identify_stopwords(tokenize(text))
	)