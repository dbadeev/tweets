from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd


def model_preprocessing(clf, df, df_df):
    vectorizers = df_df.columns
    preprocessors = df_df.index

    df_res = pd.DataFrame(index=preprocessors, columns=vectorizers)
    for vectorizer in vectorizers:
        for preprocessor in preprocessors:
            if preprocessor != 'any other ideas':
                X_train, X_test, y_train, y_test = \
					train_test_split(df_df[vectorizer][preprocessor],
								 df['type'], test_size=0.2, random_state=21)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                df_res[vectorizer][preprocessor] = \
												accuracy_score(y_test, y_pred)
    return df_res
