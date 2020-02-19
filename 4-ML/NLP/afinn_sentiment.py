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
plt.show()
