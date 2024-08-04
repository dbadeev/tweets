from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.text import text_to_word_sequence
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api


class mean_vectorizer(object):
	def __init__(self, word2vec):
		self.word2vec = word2vec
		self.dim = len(next(iter(word2vec.values())))

	def fit(self, X):
		return self

	def transform(self, X):
		return np.array([
			np.mean([self.word2vec[w] for w in words if w in self.word2vec]
					or [np.zeros(self.dim)], axis=0)
			for words in X
		])


def w2v(vectorizer, df, df_df):
    """
    This function preprocesses the input data for word2vec vectorization.
    It takes three parameters:
    - vectorizer (str): The type of vectorizer to use, either 'w2v' or 'glove-25', 'glove-100', or 'glove-200'.
    - df (pandas DataFrame): The input dataframe containing the tweets.
    - df_df (pandas DataFrame): The input dataframe containing the corresponding sentiment types.

    The function returns three values:
    - tweets_lists (list): A list of lists, where each inner list is a sequence of words from a tweet.
    - sent_type (list): A list containing the sentiment types of the corresponding tweets.
    - w2v_ (dict): A dictionary containing the word2vec vectors for the words in the input data.

    If the vectorizer is 'w2v', it trains a new Word2Vec model on the input data.
    If the vectorizer is 'glove-25', 'glove-100', or 'glove-200', it loads the corresponding glove vectors from the
    Gensim downloader.

    If an unknown vectorizer is provided, the function prints an error message and exits with a status code of 1.
    """
    tweets_lists = []
    w2v_ = {}
    vect = {'glove-25': 'glove-twitter-25',
            'glove-100': 'glove-twitter-100',
            'glove-200': 'glove-twitter-200'}

    sent_type = df.iloc[:, 1].astype(int).tolist()

    df_df_ = df_df.iloc[:, 0].astype(str).tolist()
    tweets_lists = [text_to_word_sequence(tw) for tw in df_df_]

    if vectorizer == 'w2v':
        w2v_model = Word2Vec(sentences=tweets_lists, vector_size=100,
                         					 		window=5, min_count=1)
        w2v_ = dict(zip(w2v_model.wv.key_to_index, w2v_model.wv.vectors))
    elif vectorizer in ['glove-25', 'glove-100', 'glove-200']:
        w2v_corpus = api.load(vect[vectorizer])
        w2v_ = dict(zip(w2v_corpus.key_to_index, w2v_corpus.vectors))
    else:
        print(f'Unknown vectorizer. Aborting...')
        exit(1)

    return tweets_lists, sent_type, w2v_


def model_preprocessing(clf, df, df_df):
    """
    This function preprocesses the input data for various vectorizers and sentiment classifiers.
    It takes three parameters:
    - clf (sklearn.base.BaseEstimator): A scikit-learn classifier object.
    - df (pandas.DataFrame): The input dataframe containing the tweets.
    - df_df (pandas.DataFrame): The input dataframe containing the corresponding sentiment types.

    The function returns a pandas DataFrame, df_res, which contains the accuracy scores of the classifier for each
    combination of vectorizer and preprocessor.

    The function first initializes a list of vectorizers and a list of preprocessors. It then creates a DataFrame,
    df_res, with the vectorizers as columns and the preprocessors as indices.

    For each combination of vectorizer and preprocessor, the function calls the w2v function to preprocess the input
    data and obtain the word2vec vectors. It then uses the mean_vectorizer class to transform the input data into a
    mean vector representation.

    The function then splits the transformed data into training and testing sets using scikit-learn's train_test_split
    function. It trains the classifier on the training set and predicts the sentiment types for the testing set.

    The accuracy score of the classifier's predictions is then stored in the corresponding cell of the df_res DataFrame.

    Finally, the function returns the df_res DataFrame containing the accuracy scores for each combination of
    vectorizer and preprocessor.
    """
    vectorizers = ['w2v', 'glove-25', 'glove-100', 'glove-200']
    preprocessors = list(df_df.columns)

    df_res = pd.DataFrame(index=preprocessors, columns=vectorizers)
    for vectorizer in vectorizers:
        for preprocessor in preprocessors:
            if preprocessor != 'original':
                tweets_lists, sent_type, w2v_ = w2v(vectorizer, df, df_df)
                data_mean = mean_vectorizer(w2v_).fit(tweets_lists).transform(
                    tweets_lists)
                X_train, X_test, y_train, y_test = train_test_split(
                    data_mean, sent_type, test_size=0.2, random_state=21)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                df_res[vectorizer][preprocessor] = accuracy_score(y_test, y_pred)
    return df_res
