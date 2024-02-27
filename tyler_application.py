import pandas as pd
import re
from pandas.api.types import is_string_dtype 
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer() # initialize a Sentiment Intensity Analyzer

class Tweet_Preprocessor:
    def __init__(self, column_names = {'Topic':'Topic','Tweet':'Tweet'}):
        # check column_names
        if type(column_names) != list:
            raise TypeError('columns must be list')
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
        Parameters
            df: pandas.DataFrame
                a dataframe with four columns whose elements are all strings or NaN
        Returns 
            pandas.DataFrame
                the same dataframe with
                    first column (dtype: a numeric type), containing numerics
                    second column (dtype: object), containing strings
                    third column (dtype: object), containing strings
                    fourth column (dtype: object), containing lists of strings
        Raises
            TypeError
                if type(df) != pandas.DataFrame
            ValueError
                if df does not have exactly four columns
        """
        if type(df) != pd.DataFrame:
            raise TypeError('Please pass a pandas DataFrame')
        if self.column_names['Topic'] not in df.columns or self.column_names['Tweet'] not in df.columns: # check that necessary columns as present
            raise ValueError(f'DataFrame must have {self.column_names['Topic']} and {self.column_names['Tweet']} as column names')
    
        df = df.dropna(axis = 0, subset = [self.column_names['Topic'], self.column_names['Tweet']]) # drop rows with one or more NaN values

        if not is_string_dtype(df[self.column_names['Topic']]):
            raise TypeError(f'The {self.column_names['Topic']} column contained one or more non-str values')
        if not is_string_dtype(df[self.column_names['Tweet']]):
            raise TypeError(f'The {self.column_names['Tweet']} column contained one or more non-str values')
        
        df.iloc[:,0] = pd.to_numeric(pre_df.iloc[:,0])

        df.iloc[:,3] = df.iloc[:,3].apply(str.lower)

        df.iloc[:,3] = df.iloc[:,3].apply(self.__remove_non_ASCII)

        df.iloc[:,3] = df.iloc[:,3].apply(self.__remove_additional_whitespace)

        df.iloc[:,3] = df.iloc[:,3].apply(word_tokenize)
        
        return df
    

def analyze_tweets(tweets: pd.DataFrame, sia: SentimentIntensityAnalyzer) -> pd.DataFrame:
    """
    Parameters
        tweets: pd.DataFrame
        sia: SentimentIntensityAnalyzer
    Returns
        df: the same dataframe 
            without the Sentiment column
            with negative, neutral, positive and composite sentiment indicated in the Neg, Neu, Pos and Comp columns, respectively
    """
    #tweets['Sentiment'] = tweets['Content'].apply(lambda x: sia.polarity_scores(x)['compound'])
    tweets[['Neg','Neu','Pos','Comp']] = pd.json_normalize(tweets['Content'].apply(lambda x: sia.polarity_scores(x)))
    tweets = tweets.drop('Sentiment',axis=1)
    return tweets

