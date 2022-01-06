import text_cleninig
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.stem import WordNetLemmatizer


def word_exists(df, column):
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
    # initialize
    #     cv = CountVectorizer(stop_words='english')
    cv = CountVectorizer()
    cv_matrix = cv.fit_transform(df[column])
    # create document term matrix
    df_dtm = pd.DataFrame(cv_matrix.toarray(), index=df['tweets'],
						  columns=cv.get_feature_names_out())
#     columns = df_dtm.columns
#     for col in columns:
#         df_dtm[col] = pd.to_numeric(df_dtm[col],downcast='integer')
#     df = df.set_index('tweets')

#     df_dtm = pd.concat([df_dtm, df['type']], axis=1)

    return df_dtm


def tfidf(df, column):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df[column])
#     df_dtm = pd.DataFrame(cv_matrix.toarray(), index=df['tweets'],
#     columns=cv.get_feature_names_out())

    df_dtm = pd.DataFrame(data=X.todense(), index=df['tweets'],
						  columns=vectorizer.get_feature_names_out())
#     columns = df_dtm.columns
#     for col in columns:
#         df_dtm[col] = pd.to_numeric(df_dtm[col],downcast='float')
#     df = df.set_index('tweets')

#     df_dtm = pd.concat([df_dtm, df['type']], axis=1)
    return df_dtm


def stem_text(text, stemmer, misspelling=False):
    text = text_cleninig.clean(text, misspelling=misspelling)
    text = text.split()
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)

    return text


def lem_text(text, misspelling=False):
    wordnet_lemmatizer = WordNetLemmatizer()
    text = text_cleninig.clean(text, misspelling=misspelling)
    text = text.split()
    lemmed_words = [wordnet_lemmatizer.lemmatize(word) for word in text]
    text = " ".join(lemmed_words)

    return text
