"""
This module dataloading, preprocessing, sentiment analysis and evaluation
"""

# Imports and downloads
import re
import emoji
import nltk
import pandas as pd

from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import ConfusionMatrixDisplay
from textblob import TextBlob
from transformers import pipeline

nltk.download("vader_lexicon")


def main():
    """
    Loads dataset, performs preprocessing and sentiment analysis, displays plots to
    compare and evaluate predictions

    """
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