import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

sia = SentimentIntensityAnalyzer() # initialize a Sentiment Intensity Analyzer

class CleanHerIHardlyKnowHer:
    def __init__(self, path, column_names):
        if type(path) != str:
            raise TypeError(f'{path} is not str') 
        if type(column_names) != list:
            raise TypeError('columns must be list')
        if not all(isinstance(element, str) for element in column_names):
            raise TypeError('columns must be list(str); check that all elements are str')
        if len(column_names) != 4:
            raise ValueError('exactly four column names are required')
        self.path = path
        self.column_names = column_names
        
    def read_csv(self) -> pd.DataFrame:
        """
        Parameters
            path: str
                A string representing a valid filepath to a csv file
            columns: list(str)
                A list of four strings for the DataFrame column names
        Returns
            pd.DataFrame
        Raises
            TypeError
                if type(path) != str
                if type(columns) != list
                if any element of columns is not str
            ValueError
                if len(columns) != 4
            FileNotFoundError
                if the filepath given by path doesn't need lead to a file
        """            
        try:
            df = pd.read_csv(self.path, header = None, names = self.column_names, on_bad_lines = 'warn')
        except FileNotFoundError:
            raise FileNotFoundError('File not found')
        else: 
            return df

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
            raise TypeError('df was not a pandas DataFrame')
        if len(df.columns) != 4:
            raise ValueError('df did not have exactly four columns')
    
        df = df.dropna(axis = 0) # drop rows with one or more NaN values

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

