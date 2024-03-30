"""
This is a module for dataloading, preprocessing, sentiment analysis and evaluation
"""

# Imports and downloads
import re
import numpy as np
import pandas as pd
import emoji
import matplotlib.pyplot as plt
from pandas.api.types import is_string_dtype
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import ConfusionMatrixDisplay
from textblob import TextBlob
from transformers import pipeline

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
    #### Tyler's main funciton ####
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

    #### Brisca's main function ####

    # Load twitter dataset
    col_names = ["tweet_id", "topic", "gold_sentiment", "tweet"]
    pre_df = pd.read_csv("twitter_training.csv", names=col_names, encoding="utf-8")
    # Preprocessing
    preprocessor = TwitterPreprocessor()
    df_tweets = preprocessor.preprocess_df(pre_df)
    # Sentiment analysis
    my_analyzer = MySentimentAnalyzer()
    nltk_analyzed_tweets = my_analyzer.nltk_analyze_tweets(df_tweets)
    textblob_analyzed_tweets = my_analyzer.textblob_analyze_tweets(df_tweets)
    # The pipeline analyzer on the entire dataset takes forever (something > 45mins)
    # The plots in the repo are based on the first 10,000 lines (takes ca 15mins)
    # Here, I'm only runnning it on 1000
    pipeline_analyzed_tweets = my_analyzer.pipeline_analyze_tweets(
        df_tweets.iloc[:1000]
    )
    # Compare nltk and textblob predictions
    evaluator = Evaluator()
    evaluator.compare_predictions(
        nltk_analyzed_tweets["discrete_score"],
        "nltk",
        textblob_analyzed_tweets["discrete_score"],
        "textblob",
    )
    evaluator.compare_predictions(
        nltk_analyzed_tweets["discrete_score"],
        "nltk",
        textblob_analyzed_tweets["discrete_score"],
        "textblob",
        normalization="true",
    )
    evaluator.compare_predictions(
        nltk_analyzed_tweets["discrete_score"],
        "nltk",
        textblob_analyzed_tweets["discrete_score"],
        "textblob",
        normalization="pred",
    )
    # Check accuracy of each model's predictions
    evaluator.eval_predictions(
        nltk_analyzed_tweets["gold_sentiment"],
        nltk_analyzed_tweets["discrete_score"],
        "nltk",
        normalization="true",
    )
    evaluator.eval_predictions(
        textblob_analyzed_tweets["gold_sentiment"],
        textblob_analyzed_tweets["discrete_score"],
        "textblob",
        normalization="true",
    )
    evaluator.eval_predictions(
        textblob_analyzed_tweets["gold_sentiment"].iloc[:1000],
        pipeline_analyzed_tweets["hf_sentiment"],
        "huggingface",
        normalization="true",
    )

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


#### Brisca's code
# Imports and downloads


class TwitterPreprocessor:
    """
    Class that can perform preprocessing of twitter dataframe and tweets
    """

    def preprocess_tweet(self, tweet):  # called within preprocess_df method
        """
        Parameters
            tweet: string
        Returns
            preprocessed_tweet: list of tokens, lowercased, emojis as strings,
                                non-ascii removed, additional whitespace removed
        """
        # Lowercase tweets
        preprocessed_tweet = tweet.lower()
        # Convert emojis to strings like "👍" -> ":thumbsup:"
        preprocessed_tweet = emoji.demojize(preprocessed_tweet)
        # Remove non-ASCII characters and fill with whitespace
        preprocessed_tweet = "".join(
            [i if ord(i) < 128 else " " for i in preprocessed_tweet]
        )
        # Remove additional whitespace
        preprocessed_tweet = re.sub(" +", " ", preprocessed_tweet)
        # Tokenize tweet into list of strings
        preprocessed_tweet = preprocessed_tweet.split()

        return preprocessed_tweet

    def filter_short_tweets(self, tweet):  # called within preprocess_df method
        """
        Add docstring
        """
        # Filter out tweets with < 3 tokens (empty and messy data)
        is_long_enough = len(tweet) > 2
        return is_long_enough

    def preprocess_df(self, dataframe):  # main method
        """
        Parameters
            dataframe: pd.DataFrame of twitter data
        Returns
            preprocessed_df: pd.DataFrame with rows with nan and len < 3 removed,
                             tweets with "Irrelevant" sentiment removed, sentiment
                             converted to integers, tweets preprocessed
        """
        # Remove rows with nan values in relevant columns
        preprocessed_df = dataframe.dropna(subset=["gold_sentiment", "tweet"])
        # Remove messy data: rows with empty "tweet" cell or < 3 tokens
        preprocessed_df = preprocessed_df[
            preprocessed_df["tweet"].apply(self.filter_short_tweets)
        ]
        # Remove rows with sentiment = 'Irrelevant'
        preprocessed_df = preprocessed_df[
            preprocessed_df["gold_sentiment"] != "Irrelevant"
        ]
        # Convert sentiment to integers
        sentiment_to_num = {"Positive": 1, "Neutral": 0, "Negative": -1}
        preprocessed_df["gold_sentiment"] = (
            preprocessed_df["gold_sentiment"].map(sentiment_to_num).astype(int)
        )
        # Preprocessing tweets
        preprocessed_df["tweet"] = preprocessed_df["tweet"].apply(self.preprocess_tweet)

        return preprocessed_df


class MySentimentAnalyzer:
    """
    Class that can perform sentiment analysis of twitter data using various
    libraries (nltk/vadar, Spacy/ textblob, and Huggingface a pipeline)
    """

    def cont_to_discrete_score(self, score):
        """
        The scores from the nltk and the textblob analyzers are floats between -1 and
        1. However, the gold sentiment are discrete numbers -1, 0, or 1, so we convert
        the float scores to discrete numbers to be able to compare them against the
        gold sentiment

        Parameter
            score: float between -1 and 1
        Returns
            discrete_score: -1 for scores from -1 to -0.3333
                            0 for scores from -0.3333 to 0.3333
                            1 for scores from 0.3333 to 1
        """
        if score < -0.25:
            return -1
        if score > 0.25:
            return 1
        else:
            return 0

    # Sentiment analysis with NLTK/VADAR
    def nltk_analyze_tweets(
        self, twitter_data, sentiment_analyzer_for_nltk=SentimentIntensityAnalyzer()
    ):
        """
        Parameters
            twitter_data: pd.DataFrame of tweets
            sentiment_analyzer: SentimentIntensityAnalyzer object
        Returns
            analyzed_tweets: new dataframe like twitter_data with negative, neutral,
                            positive and a composite nltk based sentiment,
                            respectively coded in the "neg", "neu", "pos" and
                            "comp" columns
        """
        analyzed_tweets = twitter_data.copy()

        analyzed_tweets["tweet"] = analyzed_tweets["tweet"].apply(
            " ".join
        )  # join strings again for polarity_scores to work
        analyzed_tweets[["neg", "neu", "pos", "comp"]] = pd.json_normalize(
            analyzed_tweets["tweet"].apply(sentiment_analyzer_for_nltk.polarity_scores)
        )
        # Convert sentiment analysis "comp" score to numbers -1, 0, or 1
        analyzed_tweets["discrete_score"] = analyzed_tweets["comp"].apply(
            self.cont_to_discrete_score
        )
        return analyzed_tweets

    # Sentiment analysis with SpaCy/ textblob
    def textblob_analyze_tweets(self, twitter_data):
        """
        Parameters
              twitter_data: pd.DataFrame of tweets
        Returns
              analyzed_tweets: new dataframe like twitter_data with textblob based
                              sentiment polarity score coded in "polarity" column
        """
        analyzed_tweets = twitter_data.copy()
        analyzed_tweets["tweet"] = analyzed_tweets["tweet"].apply(
            " ".join
        )  # join strings again for sentiment.polarity to work
        analyzed_tweets["polarity"] = analyzed_tweets["tweet"].apply(
            lambda x: TextBlob(x).sentiment.polarity
        )
        # Convert continuous sentiment analysis "polarity" score to numbers -1, 0, or 1
        analyzed_tweets["discrete_score"] = analyzed_tweets["polarity"].apply(
            self.cont_to_discrete_score
        )

        return analyzed_tweets

    # Sentiment analysis with Huggingface pipeline
    def pipeline_analyze_tweets(
        self, twitter_data, sentiment_pipeline=pipeline("sentiment-analysis")
    ):
        """
        Parameters
            twitter_data: pd.DataFrame of tweets
            sentiment_pipeline: transformers TextClassificationPipeline
        Returns
            analyzed_tweets: new dataframe like twitter_data with positive or
                            negative sentiment coded as 1 or -1, respectively, in
                            "hf_sentiment" column
        """
        analyzed_tweets = twitter_data.copy()
        analyzed_tweets["tweet"] = analyzed_tweets["tweet"].str.join(" ")
        pipeline_output = sentiment_pipeline(
            analyzed_tweets["tweet"].tolist()
        )  # outputs list of dict for each tweet with keys "labe" and "score"
        # Extract sentiment labels from output
        pipeline_sentiments_as_labels = [output["label"] for output in pipeline_output]
        # Convert "POSITIVE", "NEGATIVE" labels to numbers (there is no neutral here)
        sentiment_to_num = {"POSITIVE": 1, "NEGATIVE": -1}
        # Add column with sentiments to df, return new df
        analyzed_tweets["hf_sentiment"] = [
            sentiment_to_num[label] for label in pipeline_sentiments_as_labels
        ]
        return analyzed_tweets


class Evaluator:
    """
    Class with methods to compare and evaluate prediictions
    """

    def compare_predictions(
        self, predictions_1, name_1, predictions_2, name_2, normalization=None
    ):
        """
        Check how similar two predictions are

        Parameters
            predictions_1, predictions_2: columns of pd DataFrame with discrete
                                          scores
            name_1, name_2: strings of model names, e.g. "nltk" or "textblob" -
                            these are used in the plot title
            normalization: None (dafault), by row if "true", by column if "pred"
        Displays
            a sklearn ConfusionMatrixDisplay object with title
        """

        # Set normalization and value format according to parameters
        if normalization == "true":
            val_format = None
            normalization_comment = ", row normalized"
        elif normalization == "pred":
            val_format = None
            normalization_comment = ", col normalized"
        else:
            val_format = "d"
            normalization_comment = ""

        # Create confusion matrix plot
        disp = ConfusionMatrixDisplay.from_predictions(
            predictions_1,
            predictions_2,
            values_format=val_format,
            normalize=normalization,
        )
        disp.ax_.set_title(
            f"{name_1} ('True') vs {name_2} ('Pred'){normalization_comment}"
        )

    def eval_predictions(self, gold_labels, predictions, name, normalization="true"):
        """
        Evaluate how accurate predictions are

        Parameters
            gold_labels, predictions: columns of pd DataFrame with discrete
                                      scores
            name: string of model name, e.g. "nltk" or "textblob" - this is used
                  in the plot title
            normalization: None (dafault), by row if "true", by column if "pred"
        Displays
            a sklearn ConfusionMatrixDisplay object with title
        """

        # Set normalization and value format according to parameters
        if normalization == "true":
            val_format = None
            normalization_comment = ", row normalized"
        elif normalization == "pred":
            val_format = None
            normalization_comment = ", col normalized"
        else:
            val_format = "d"
            normalization_comment = ""

        disp = ConfusionMatrixDisplay.from_predictions(
            gold_labels, predictions, values_format=val_format, normalize=normalization
        )
        disp.ax_.set_title(f"gold vs {name} predicted sentiment{normalization_comment}")


if __name__ == "__main__":
    main()
