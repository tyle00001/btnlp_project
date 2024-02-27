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
        self.column_names = column_names
    
    def __remove_non_ASCII(self, string: str) -> str:
        return re.sub('[^\x00-\x7F]',' ', string)

    def __remove_additional_whitespace(self, string: str):
        return re.sub(' {2,}', ' ' ,string)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
  
        """
        if type(df) != pd.DataFrame:
            raise TypeError('Please pass a pandas DataFrame')
        if self.column_names['Topic'] not in df.columns or self.column_names['Tweet'] not in df.columns: # check that necessary columns as present
            raise ValueError(f'DataFrame must have {self.column_names['Topic']} and {self.column_names['Tweet']} as column names')
    
        df = df.dropna(axis = 0, subset = [self.column_names['Topic'], self.column_names['Tweet']]) # drop rows with one or more NaN values

        # check that both Topic and Tweet columns only contain strings
        if not is_string_dtype(df[self.column_names['Topic']]):
            raise TypeError(f'The {self.column_names['Topic']} column contained one or more non-str values')
        if not is_string_dtype(df[self.column_names['Tweet']]):
            raise TypeError(f'The {self.column_names['Tweet']} column contained one or more non-str values')
        
        # lowercase both Topic and Tweet columns
        df[[self.column_names['Topic'], self.column_names['Tweet']]] = df[[self.column_names['Topic'], self.column_names['Tweet']]].apply(str.lower)

        df[self.column_names['Tweet']] = df[self.column_names['Tweet']].apply(self.__remove_non_ASCII)
        df[self.column_names['Tweet']] = df[self.column_names['Tweet']].apply(self.__remove_additional_whitespace)

        df[self.column_names['Tweet']] = df[self.column_names['Tweet']].apply(word_tokenize) # tokenize tweets
        df[self.column_names['Tweet']] = df[self.column_names['Tweet']].apply(lambda x: w for w in x if w not in STOPWORDS) # remove stopwords
        df[self.column_names['Tweet']] = df[self.column_names['Tweet']].apply(lambda x: " ".join(x)) # recombine lists into a single string, separated by spaces
       
        return df
    

def analyze_tweets(tweets: pd.DataFrame, sia: SentimentIntensityAnalyzer, column_names = {'Topic':'Topic','Tweet':'Tweet'}) -> pd.DataFrame:
    """
    Parameters
        tweets: pd.DataFrame
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
    
    if type(tweets) != pd.DataFrame:
        raise TypeError('tweets must be a pandas.DataFrame')
    if column_names['Topic'] not in tweets.columns or column_names['Tweet'] not in tweets.columns: # check that necessary columns as present
        raise ValueError(f'DataFrame must have {column_names['Topic']} and {column_names['Tweet']} as column names')

    tweets[['Neg','Neu','Pos','Comp']] = pd.json_normalize(tweets['Content'].apply(lambda x: sia.polarity_scores(x)))
    return tweets

