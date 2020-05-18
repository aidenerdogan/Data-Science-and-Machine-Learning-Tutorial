from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('news.csv')
# print(data)
articles = data['clean_text']

# compute sentiment scores (polarity) and labels
sentiment_scores_tb = [round(TextBlob(article).sentiment.polarity, 3) for article in articles]
sentiment_category_tb = ['positive' if score > 0 
                             else 'negative' if score < 0 
                                 else 'neutral' 
                                     for score in sentiment_scores_tb]


# sentiment statistics per news category
df = pd.DataFrame([list(data['news_category']), sentiment_scores_tb, sentiment_category_tb]).T
df.columns = ['news_category', 'sentiment_score', 'sentiment_category']
df['sentiment_score'] = df.sentiment_score.astype('float')
# print(df.groupby(by=['news_category']).describe())

# fc = sns.factorplot(x="news_category", hue="sentiment_category", 
                    # data=df, kind="count", 
                    # palette={"negative": "#FE2020", 
                    #          "positive": "#BADD07", 
                    #          "neutral": "#68BFF5"})
# plt.show()


pos_idx = df[(df.news_category=='world') & (df.sentiment_score == 0.7)].index[0]
neg_idx = df[(df.news_category=='world') & (df.sentiment_score == -0.296)].index[0]

print('Most Negative World News Article:', data.iloc[neg_idx][['news_article']][0])
print()
print('Most Positive World News Article:', data.iloc[pos_idx][['news_article']][0])

# source = https://towardsdatascience.com/a-practitioners-guide-to-natural-language-processing-part-i-processing-understanding-text-9f4abfd13e72