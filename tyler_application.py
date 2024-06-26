import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_string_dtype
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import ConfusionMatrixDisplay

STOPWORDS = set(stopwords.words("english"))


def main():
    """
    Parameters:
        pee_pee: str
    Returns:
        poo_poo: str
    Raises:
        Check
    """
    raw = pd.read_csv(
        "twitter_training.csv", names=["Number", "Topic", "Sentiment", "Tweet"]
    )
    preprocessor = TweetPreprocessor()
    processed = preprocessor.clean_data(raw)
    sia = SentimentIntensityAnalyzer()
    analyzed = analyze_tweets(processed, sia)
    confusion_matrix = display_confusion_matrix(
        analyzed["Sentiment"], analyzed["Comp"], normalize_by="true"
    )
    confusion_matrix.ax_.set_title("VADER sentiment vs gold sentiment, row normalized")
    plt.show()


class TweetPreprocessor:
    """
    A class for preprocessing tweets
    """
    def __init__(self, topic="Topic", tweet="Tweet"):
        """
        Parameters:
            topic: str
                this will be considered the name of the Topic column by the TweetPreprocessor
            tweet: str
                this will be considered the name of the Tweet column by the TweetPreprocessor
        """
        if not isinstance(topic, str):
            raise TypeError("topic must be a str")
        if not isinstance(tweet, str):
            raise TypeError("tweet must be a str")
        self.topic = topic
        self.tweet = tweet

    def remove_non_ascii(self, string: str) -> str:
        """
        Parameters
            string: str
        Returns
            string without non-ascii characters
        """
        return re.sub("[^\x00-\x7F]", " ", string)

    def remove_additional_whitespace(self, string: str):
        """
        Parameters
            string: str
        Returns
            string without additional whitespace
        """
        return re.sub(" {2,}", " ", string)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
            df: pandas.Dataframe
        Returns
            df with the strings in Tweet column lowercase,
            without additional whitespace and without stopwords
        Raises
            TypeError
                if df is not a pandas DataFrame
                if the Topic column contains types other than NaN or string
                if the Tweet column contains types other than NaN or string
            ValueError
                if df does not have columns with the Tweet and Topic labels
                that this Tweet_Preprocessor with initialized with
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Please pass a pandas DataFrame")
        if self.topic not in df.columns or self.tweet not in df.columns:
            raise ValueError(
                f"DataFrame must have {self.topic} and {self.tweet} as column names"
            )

        df = df.dropna(axis=0, subset=[self.topic, self.tweet])

        if not is_string_dtype(df[self.topic]):
            raise TypeError(
                f"The {self.topic} column contained one or more non-str values"
            )
        if not is_string_dtype(df[self.tweet]):
            raise TypeError(
                f"The {self.tweet} column contained one or more non-str values"
            )

        df[self.topic] = df[self.topic].apply(str.lower)
        df[self.tweet] = df[self.tweet].apply(str.lower)

        df[self.tweet] = (
            df[self.tweet]
            .apply(self.remove_non_ascii)
            .apply(self.remove_additional_whitespace)
        )

        df[self.tweet] = df[self.tweet].apply(word_tokenize)  # tokenize tweets
        df[self.tweet] = df[self.tweet].apply(
            lambda x: [w for w in x if w not in STOPWORDS]
        )  # remove stopwords
        df[self.tweet] = df[self.tweet].apply(" ".join)

        return df


def analyze_tweets(
    twitter_data: pd.DataFrame,
    sia: SentimentIntensityAnalyzer,
    column_names={"Topic": "Topic", "Tweet": "Tweet"},
) -> pd.DataFrame:
    """
    Parameters
        twitter_data: pd.DataFrame
        sia: SentimentIntensityAnalyzer
        column_names: dict
            a dictionary providing the names of the Topic and Tweet columns
    Returns
        df: the same dataframe
            without the Sentiment column
            with negative, neutral, positive and composite sentiment
            indicated in the Neg, Neu, Pos and Comp columns, respectively
    """
    if not isinstance(sia, SentimentIntensityAnalyzer):
        raise TypeError("sia must be a SentimentIntensityAnalyzer")

    if not isinstance(column_names, dict):
        raise TypeError("column_names must be dict")
    if (
        "Tweet" not in column_names.keys() or "Topic" not in column_names.keys()
    ):  # check that Tweet and Topic columns are present
        raise ValueError(
            'If passing non-default names for "Topic" and "Tweet" columns,'
            'column_names must contain "Topic" and "Tweet" as keys'
        )
    topic = column_names["Topic"]
    tweet = column_names["Tweet"]

    if not isinstance(twitter_data, pd.DataFrame):
        raise TypeError("tweets must be a pandas.DataFrame")
    if (
        topic not in twitter_data.columns or tweet not in twitter_data.columns
    ):  # check that necessary columns as present
        raise ValueError(f"DataFrame must have {topic} and {tweet} as column names")

    twitter_data[["Neg", "Neu", "Pos", "Comp"]] = pd.json_normalize(
        twitter_data[tweet].apply(lambda x: sia.polarity_scores(x))
    )
    return twitter_data


def display_confusion_matrix(test: pd.Series, pred: pd.Series, normalize_by="true"):
    """
    Parameters
        test: pd.Series
            a series of test values
        pred: pd.Serties
            a series of predicted values
        normalize_by: str
            possible values: 'true' or 'pred'
            indicates whether to normalize by test or by pred
    Returns
        ConfusionMatrixDisplay
    """
    gold = np.select(
        condlist=[test == "Positive", test == "Negative"], choicelist=[-1, 1], default=0
    )
    d_scores = np.select(
        condlist=[pred < -0.25, pred > 0.25], choicelist=[-1, 1], default=0
    )
    return ConfusionMatrixDisplay.from_predictions(
        gold, d_scores, normalize=normalize_by
    )


if __name__ == "__main__":
    main()
