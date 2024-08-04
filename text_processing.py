import text_cleaninig
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.stem import WordNetLemmatizer


def word_exists(df, column):
    """
    This function takes a DataFrame 'df' and a column name 'column' from the DataFrame.
    It uses CountVectorizer to create a document-term matrix (DTM) from the specified column.
    The DTM is then transformed into a DataFrame with the same index as the original DataFrame.
    For each column in the DTM, the function applies a lambda function to replace 0 values with 1.
    This function does not return a DataFrame with an additional 'type' column.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the text data.
    column (str): The name of the column in the DataFrame from which the text data will be extracted.

    Returns:
    pandas.DataFrame: A DataFrame containing the document-term matrix with 0 values replaced by 1.
    """
    cv = CountVectorizer()
    cv_matrix = cv.fit_transform(df[column])
    # create document term matrix
    df_dtm = pd.DataFrame(cv_matrix.toarray(), index=df['tweets'],
                                                  columns=cv.get_feature_names_out())
    columns = df_dtm.columns
    for col in columns:
        df_dtm[col] = df_dtm[col].apply(lambda item: item if item == 0 else 1)
#         df_dtm[col] = pd.to_numeric(df_dtm[col],downcast='integer')
#     df = df.set_index('tweets')
#     df_dtm = pd.concat([df_dtm, df['type']], axis=1)

    return df_dtm


def word_count(df, column):
    """
    This function takes a DataFrame 'df' and a column name 'column' from the DataFrame.
    It uses CountVectorizer to create a document-term matrix (DTM) from the specified column.
    The DTM is then transformed into a DataFrame with the same index as the original DataFrame.
    For each column in the DTM, the function applies a lambda function to replace 0 values with 1.
    This function does not return a DataFrame with an additional 'type' column.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the text data.
    column (str): The name of the column in the DataFrame from which the text data will be extracted.

    Returns:
    pandas.DataFrame: A DataFrame containing the document-term matrix with 0 values replaced by 1.
    """
    cv = CountVectorizer()
    cv_matrix = cv.fit_transform(df[column])
    # create document term matrix
    df_dtm = pd.DataFrame(cv_matrix.toarray(), index=df['tweets'],
                                                  columns=cv.get_feature_names_out())
    columns = df_dtm.columns
    for col in columns:
        df_dtm[col] = df_dtm[col].apply(lambda item: item if item == 0 else 1)
#     df_dtm[col] = pd.to_numeric(df_dtm[col],downcast='integer')
#     df = df.set_index('tweets')
#     df_dtm = pd.concat([df_dtm, df['type']], axis=1)

    return df_dtm


def tfidf(df, column):
    """
    This function takes a DataFrame 'df' and a column name 'column' from the DataFrame.
    It uses TfidfVectorizer to create a document-term matrix (DTM) from the specified column.
    The DTM is then transformed into a DataFrame with the same index as the original DataFrame.
    The function does not return a DataFrame with an additional 'type' column.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the text data.
    column (str): The name of the column in the DataFrame from which the text data will be extracted.

    Returns:
    pandas.DataFrame: A DataFrame containing the document-term matrix.
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df[column])
    df_dtm = pd.DataFrame(data=X.todense(), index=df['tweets'],
                                                  columns=vectorizer.get_feature_names_out())
#     columns = df_dtm.columns
#     for col in columns:
#         df_dtm[col] = pd.to_numeric(df_dtm[col],downcast='float')
#     df = df.set_index('tweets')

#     df_dtm = pd.concat([df_dtm, df['type']], axis=1)
    return df_dtm


def stem_text(text, stemmer, misspelling=False):
    """
    This function takes a text string and a stemmer object as input.
    It cleans the input text using the `text_cleaninig.clean` function,
    splits the cleaned text into words, and then applies the stemming operation
    to each word using the provided stemmer. The stemmed words are then joined back into a single string.

    Parameters:
    text (str): The input text string to be stemmed.
    stemmer (object): A stemmer object, such as `nltk.stem.SnowballStemmer` or `nltk.stem.WordNetLemmatizer`.
    misspelling (bool, optional): A flag indicating whether to apply misspelling corrections to the input text.
        Defaults to `False`.

    Returns:
    str: A string containing the stemmed version of the input text.
    """
    text = text_cleaninig.clean(text, misspelling=misspelling)
    text = text.split()
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)

    return text


def lem_text(text, misspelling=False):
    """
    This function takes a text string and a boolean flag indicating whether to apply misspelling corrections.
    It cleans the input text using the `text_cleaninig.clean` function,
    splits the cleaned text into words, and then applies lemmatization to each word using the WordNetLemmatizer.
    The lemmatized words are then joined back into a single string.

    Parameters:
    text (str): The input text string to be lemmatized.
    misspelling (bool, optional): A flag indicating whether to apply misspelling corrections to the input text.
        Defaults to `False`.

    Returns:
    str: A string containing the lemmatized version of the input text.
    """
    wordnet_lemmatizer = WordNetLemmatizer()
    text = text_cleaninig.clean(text, misspelling=misspelling)
    text = text.split()
    lemmed_words = [wordnet_lemmatizer.lemmatize(word) for word in text]
    text = " ".join(lemmed_words)

    return text
