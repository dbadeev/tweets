from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import GridSearchCV
import warnings
import numpy as np


def model_preprocessing(clf, df, df_df):
    """
    This function performs model preprocessing by training a machine learning model on different combinations of
    vectorizers and preprocessors.

    Parameters:
    - clf (sklearn.base.BaseEstimator): A scikit-learn estimator to be trained on the data.
    - df (pandas.DataFrame): A pandas DataFrame containing the target variable 'type'.
    - df_df (pandas.DataFrame): A pandas DataFrame containing the vectorizers and preprocessors as columns and index
    respectively.

    Returns:
    - df_res (pandas.DataFrame): A pandas DataFrame containing the accuracy scores of the trained models for each
    combination of vectorizers and preprocessors.
    """
    vectorizers = df_df.columns
    preprocessors = df_df.index

    df_res = pd.DataFrame(index=preprocessors, columns=vectorizers)
    for vectorizer in vectorizers:
        for preprocessor in preprocessors:
            X_train, X_test, y_train, y_test = \
                train_test_split(df_df[vectorizer][preprocessor],
                                 df['type'], test_size=0.2, random_state=21)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            df_res[vectorizer][preprocessor] = accuracy_score(y_test, y_pred)
    return df_res


def grid_search_all(clf, df, df_df, parameters):
    """
    This function performs a grid search for a given machine learning model (clf) on all combinations of vectorizers and
    preprocessors specified in the df_df DataFrame. The function uses the GridSearchCV class from scikit-learn to
    search for the best hyperparameters for each combination of vectorizer and preprocessor. The results are printed
    to the console.

    Parameters:
    - clf (sklearn.base.BaseEstimator): A scikit-learn estimator to be trained on the data.
    - df (pandas.DataFrame): A pandas DataFrame containing the target variable 'type'.
    - df_df (pandas.DataFrame): A pandas DataFrame containing the vectorizers and preprocessors as columns and
      index respectively.
    - parameters (dict): A dictionary containing the hyperparameters to be searched for each vectorizer and
      preprocessor combination.

    Returns:
    None. The function prints the accuracy scores and best parameters for each combination of vectorizer and
    preprocessor to the console.
    """
    vectorizers = df_df.columns
    preprocessors = df_df.index

    for preprocessor in preprocessors:
        for vectorizer in vectorizers:
            gs = GridSearchCV(clf,
                              param_grid=parameters,
                              scoring='accuracy',
                              cv=5)
            # X_train, X_test, y_train, y_test = \
            #     train_test_split(df_df[vectorizer][preprocessor],
            #                      df['type'], test_size=0.2, random_state=21)
            gs.fit(df_df[vectorizer][preprocessor], df['type'])
            print(f'{preprocessor}, {vectorizer}:\n'
                  f'accuracy: {gs.best_score_:.6f},\n'
                  f'params = {gs.best_params_}\n')


def grid_search(clf, df, df_x, parameters):
    """
    This function performs a grid search for a given machine learning model (clf) on the input data (df_x) and target
    variable (df['type']).
    The function uses the GridSearchCV class from scikit-learn to search for the best hyperparameters for the given
    model.

    Parameters:
    - clf (sklearn.base.BaseEstimator): A scikit-learn estimator to be trained on the data.
    - df (pandas.DataFrame): A pandas DataFrame containing the target variable 'type'.
    - df_x (pandas.DataFrame): A pandas DataFrame containing the input features to be used for training the model.
    - parameters (dict): A dictionary containing the hyperparameters to be searched for the given model.

    Returns:
    None. The function prints the accuracy score and best parameters for the trained model to the console.
    """
    gs = GridSearchCV(clf,
                        param_grid=parameters,
                        scoring='accuracy',
                        # n_jobs=-1,
                        cv=5)
    gs.fit(df_x, df['type'])
    print(f'accuracy: {gs.best_score_:.6f},\nparams = {gs.best_params_}')


def custom_grid_search():
    pass