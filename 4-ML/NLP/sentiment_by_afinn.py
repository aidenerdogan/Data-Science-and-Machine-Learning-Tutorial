from afinn import Afinn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('news.csv')
# print(data)
af = Afinn()
articles = data['full_text']
# compute sentiment scores (polarity) and labels
sentiment_scores = [af.score(article) for article in articles]
sentiment_category = ['positive' if score > 0 else 'negative' if score < 0 else 'natural' for score in sentiment_scores]

# sentiment statistics per news category
df =  pd.DataFrame([list(data['news_category']), sentiment_scores, sentiment_category]).T
df.columns = ['news_category', 'sentiment_score', 'sentiment_category']
df['sentiment_score'] = df.sentiment_score.astype(float)
# print(df.groupby(by=['news_category']).describe())

# visualizing news sentiments
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
sp = sns.stripplot(x='news_category',y='sentiment_score',hue='news_category', data=df, ax=ax1)
bp = sns.boxplot(x='news_category',y='sentiment_score',hue='news_category', data=df, palette='Set2', ax=ax2)
t = f.suptitle('Visializing News Sentiment', fontsize=14)
# plt.show()
fc = sns.factorplot(x='news_category', hue='sentiment_category', data=df, kind='count', palette={"negative": "#FE2020", 
                             "positive": "#BADD07", 
                             "natural": "#68BFF5"})
# plt.show()
# most articles for techonology
# pos_idx = df[(df.news_category=='technology') & (df.sentiment_score==6)].index[0]
# neg_idx = df[(df.news_category=='technology') & (df.sentiment_score==-15)].index[0]
# print('Most Negeative Tech News Article: ', data.iloc[neg_idx][['news_article']][0])
# print()
# print('Most Positive Tech News Article:', data.iloc[pos_idx][['news_article']][0])

# most articles for world news
# pos_idx = df[(df.news_category=='world') & (df.sentiment_score == 16)].index[0]
# neg_idx = df[(df.news_category=='world') & (df.sentiment_score == -12)].index[0]

# print('Most Negative World News Article:', data.iloc[neg_idx][['news_article']][0])
# print()
# print('Most Positive World News Article:', data.iloc[pos_idx][['news_article']][0])

# source = https://towardsdatascience.com/a-practitioners-guide-to-natural-language-processing-part-i-processing-understanding-text-9f4abfd13e72