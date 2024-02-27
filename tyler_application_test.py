import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from tyler_application import TweetPreprocessor, analyze_tweets
raw = pd.read_csv('twitter_training.csv', names = ['Number','Topic','Sentiment','Tweet'])
preprocessor = TweetPreprocessor()
processed = preprocessor.clean_data(raw)
sia = SentimentIntensityAnalyzer()
analyzed = analyze_tweets(processed, sia)
print(analyzed)