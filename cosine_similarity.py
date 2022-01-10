import pandas as pd
from scipy import spatial


def csim_calc(df, num_tweets: int):
	df_csim = pd.DataFrame(index=range(1, 11),
						   columns=['tweet1', 'tweet2', 'cos_sim'])
	df_csim['cos_sim'] = 0.
	for i in range(num_tweets - 1):
		for j in range(i + 1, num_tweets - 1):
			r = df.iloc[i, :].values
			s = df.iloc[j, :].values
			cnt_cos = 1 - spatial.distance.cosine(r, s)
			if cnt_cos > df_csim['cos_sim'][9]:
				df_csim['cos_sim'][9] = cnt_cos
				df_csim['tweet1'][9] = df.index[i]
				df_csim['tweet2'][9] = df.index[j]
				df_csim.sort_values(by='cos_sim', ascending=False)
	return df_csim


def cosine_similarity(df_df, num_tweets: int):
	vectorizers = df_df.columns
	preprocessors = df_df.index

	df_res = pd.DataFrame(index=preprocessors, columns=vectorizers)

	for vectorizer in vectorizers:
		for preprocessor in preprocessors:
			if preprocessor != 'any other ideas':
				df_res[vectorizer][preprocessor] = \
						csim_calc(df_df[vectorizer][preprocessor], num_tweets)
	return df_res