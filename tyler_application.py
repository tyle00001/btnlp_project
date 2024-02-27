import pandas as pd
import re
from pandas.api.types import is_string_dtype 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer() # initialize a Sentiment Intensity Analyzer
STOPWORDS = set(stopwords.words('english'))

class Tweet_Preprocessor:
    def __init__(self, column_names = {'Topic':'Topic','Tweet':'Tweet'}):
        # check column_names
        if type(column_names) != dict:
            raise TypeError('column_names must be dict')
        # check values 
        if 'Tweet' not in column_names.keys() or 'Topic' not in column_names.keys():
            raise ValueError('If passing non-default names for "Topic" and "Tweet" columns, column_names must contain "Topic" and "Tweet" as keys')
        self.topic = column_names['Topic']
        self.tweet = column_names['Tweet']
    
    def __remove_non_ASCII(self, string: str) -> str:
        return re.sub('[^\x00-\x7F]',' ', string)

    def __remove_additional_whitespace(self, string: str):
        return re.sub(' {2,}', ' ' ,string)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
  
        """
        if type(df) != pd.DataFrame:
            raise TypeError('Please pass a pandas DataFrame')
        if self.topic not in df.columns or self.tweet not in df.columns: # check that necessary columns as present
            raise ValueError(f"DataFrame must have {self.topic} and {self.tweet} as column names")
    
        df = df.dropna(axis = 0, subset = [self.topic, self.tweet]) # drop rows with one or more NaN values

        # check that both Topic and Tweet columns only contain strings
        if not is_string_dtype(df[self.topic]):
            raise TypeError(f'The {self.column_names["Topic"]} column contained one or more non-str values')
        if not is_string_dtype(df[self.tweet]):
            raise TypeError(f'The {self.column_names["Tweet"]} column contained one or more non-str values')
        
        # lowercase both Topic and Tweet columns
        df[self.topic] = df[self.topic].apply(str.lower)
        df[self.tweet] = df[self.tweet].apply(str.lower)

        df[self.tweet] = df[self.tweet].apply(self.__remove_non_ASCII).apply(self.__remove_additional_whitespace)

        df[self.tweet] = df[self.tweet].apply(word_tokenize) # tokenize tweets
        df[self.tweet] = df[self.tweet].apply(lambda x: [w for w in x if w not in STOPWORDS]) # remove stopwords
        df[self.tweet] = df[self.tweet].apply(lambda x: " ".join(x)) # recombine lists into a single string, separated by spaces
       
        return df
    

def analyze_tweets(twitter_data: pd.DataFrame, sia: SentimentIntensityAnalyzer, column_names = {'Topic':'Topic','Tweet':'Tweet'}) -> pd.DataFrame:
    """
    Parameters
        twitter_data: pd.DataFrame
        sia: SentimentIntensityAnalyzer
        column_names: dict
            a dictionary providing the names of the Topic and Tweet columns
    Returns
        df: the same dataframe 
            without the Sentiment column
            with negative, neutral, positive and composite sentiment indicated in the Neg, Neu, Pos and Comp columns, respectively
    """    
    if type(sia) != SentimentIntensityAnalyzer:
        raise TypeError('sia must be a SentimentIntensityAnalyzer')
    
    if type(column_names) != dict:
        raise TypeError('column_names must be dict')
    if 'Tweet' not in column_names.keys() or 'Topic' not in column_names.keys(): # check that Tweet and Topic columns are present
        raise ValueError('If passing non-default names for "Topic" and "Tweet" columns, column_names must contain "Topic" and "Tweet" as keys')
    topic = column_names['Topic']
    tweet = column_names['Tweet']

    if type(twitter_data) != pd.DataFrame:
        raise TypeError('tweets must be a pandas.DataFrame')
    if topic not in twitter_data.columns or tweet not in twitter_data.columns: # check that necessary columns as present
        raise ValueError(f"DataFrame must have {topic} and {tweet} as column names")

    twitter_data[['Neg','Neu','Pos','Comp']] = pd.json_normalize(twitter_data[tweet].apply(lambda x: sia.polarity_scores(x)))
    return twitter_data

