import pandas as pd
import emoji
import re
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tyler_application import TweetPreprocessor, analyze_tweets, display_confusion_matrix
"""
raw = pd.read_csv('twitter_training.csv', names = ['Number','Topic','Sentiment','Tweet'])
preprocessor = TweetPreprocessor()
processed = preprocessor.clean_data(raw)
sia = SentimentIntensityAnalyzer()
analyzed = analyze_tweets(processed, sia)
print(analyzed)
"""

# Define class and methods for preprocessing twitter df and tweets

class Twitter_Preprocessor(): # Brisca's preprocessor
    '''
    Class that can perform preprocessing of twitter dataframe and tweets
    '''
    def preprocess_tweet(self, tweet): # called within preprocess_df method
        '''
        Parameters
            tweet: string
        Returns
            preprocessed_tweet: list of tokens, lowercased, emojis as strings,
                                non-ascii removed, additional whitespace removed
        '''
        # Lowercase tweets
        preprocessed_tweet = tweet.lower()
        # Convert emojis to strings
        preprocessed_tweet = emoji.demojize(preprocessed_tweet)
        # Remove non-ASCII characters and fill with whitespace
        preprocessed_tweet = ''.join([i if ord(i) < 128 else ' '
                                    for i in preprocessed_tweet])
        # Remove additional whitespace
        preprocessed_tweet = re.sub(' +', ' ', preprocessed_tweet)
        # Tokenize tweet into list of strings
        preprocessed_tweet = preprocessed_tweet.split()

        return preprocessed_tweet


    def filter_short_tweets(self, tweet): # called within preprocess_df method
        '''
        Add docstring
        '''
        # Filter out tweets with < 3 tokens (empty and messy data)
        is_long_enough = len(tweet) > 2
        return is_long_enough


    def preprocess_df(self, dataframe): # main method
        '''
        Parameters
            dataframe: pd.DataFrame of twitter data
        Returns
            preprocessed_df: pd.DataFrame with rows with nan and len < 3 removed,
                            tweets with "Irrelevant" sentiment removed, sentiment
                            converted to integers, tweets preprocessed
        '''
        # Remove rows with nan values in relevant columns
        preprocessed_df = dataframe.dropna(subset=["gold_sentiment", "tweet"])
        # Remove messy data: rows with empty "tweet" cell or < 3 tokens
        preprocessed_df = preprocessed_df[preprocessed_df["tweet"].apply(self.filter_short_tweets)]
        # Remove rows with sentiment = 'Irrelevant'
        preprocessed_df = preprocessed_df[preprocessed_df["gold_sentiment"] !=
                                        "Irrelevant"]
        # Convert sentiment to integers
        sentiment_to_num = {"Positive": 1, "Neutral": 0, "Negative": -1}
        preprocessed_df["gold_sentiment"] = preprocessed_df["gold_sentiment"].map(sentiment_to_num).astype(int)
        # Preprocessing tweets
        preprocessed_df["tweet"] = preprocessed_df["tweet"].apply(self.preprocess_tweet)

        return preprocessed_df
    

# using Brisca's preprocessor instead
col_names = ["tweet_id", "topic", "gold_sentiment", "tweet"]
raw = pd.read_csv('twitter_training.csv', names = col_names, encoding = 'utf-8')
preprocessor = Twitter_Preprocessor()
processed = preprocessor.preprocess_df(raw)
processed['tweet'] = processed['tweet'].apply(" ".join)
sia = SentimentIntensityAnalyzer()
analyzed = analyze_tweets(processed, sia, column_names={"Topic": "topic", "Tweet": "tweet"})


disp_norm_gold = display_confusion_matrix(analyzed["gold_sentiment"], analyzed["Comp"], normalize_by='true')
disp_norm_gold.ax_.set_title('VADER sentiment vs gold sentiment, row normalized')

disp_norm_nltk = display_confusion_matrix(analyzed["gold_sentiment"], analyzed["Comp"], normalize_by='pred')
disp_norm_nltk.ax_.set_title('VADER sentiment vs gold sentiment, column normalized')

print(type(disp_norm_gold))

plt.show()