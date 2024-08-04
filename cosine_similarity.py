import pandas as pd
import numpy as np


def csim_calc(df, df_df):
	df_csim = pd.DataFrame(index=range(1, 11), columns=['tweets', 'cos_sim'])
	df_csim['cos_sim'] = 0.

	df_sim = df_df.iloc[:, 1:]
	similarity = np.dot(df_sim, df_sim.T)
	# squared magnitude of preference vectors (number of occurrences)
	square_mag = np.diag(similarity)
	# inverse squared magnitude
	inv_square_mag = 1 / square_mag
	# if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
	inv_square_mag[np.isinf(inv_square_mag)] = 0
	# inverse of the magnitude
	inv_mag = np.sqrt(inv_square_mag)
	# cosine similarity (elementwise multiply by inverse magnitudes)
	cosine = similarity * inv_mag
	sim = cosine.T * inv_mag
	n_rows = sim.shape[0] - 1
	for i in range(n_rows - 1):
		for j in range(i + 1, n_rows - 1):
			if sim[i][j] > df_csim['cos_sim'].iloc[9]:
				lst_base = str(i) + '. ' + df['tweets'].iloc[i] + \
							  ' - ' + str(j) + '. ' + df['tweets'].iloc[j]
				lst_cnt = str(i) + '. ' + df_df.index[i] + \
							' - ' + str(j) + '. ' + df_df.index[j]

				lst = lst_cnt + '\n' + lst_base
				df_csim['cos_sim'].iloc[9] = sim[i][j]
				df_csim['tweets'].iloc[9] = lst
				df_csim.sort_values(by='cos_sim', ascending=False, inplace=True)
		if df_csim['cos_sim'].iloc[9] >= 1.0:
			break
	return df_csim


def cosine_similarity(df, df_df):
	vectorizers = df_df.columns
	preprocessors = df_df.index
	cols = [(vect + ' + ' + prep) for vect in vectorizers
			   							for prep in preprocessors]
	df_res = pd.DataFrame(index=range(1, 11), columns=cols)

	for vectorizer in vectorizers:
		for preprocessor in preprocessors:
			df_csim = csim_calc(df, df_df[vectorizer][preprocessor])
			df_res.loc[:, vectorizer + ' + ' + preprocessor] = df_csim.loc[:,
				'tweets']
			# print(f"{df_res.loc[:, vectorizer + ' + ' + preprocessor]}")
	return df_res


def print_cossim(df):
	for col in df.columns:
		print(f'{col}')
		for line in range(1, 11):
			print(f'\t{line}.\n{df.loc[line, col]}')
		print('\n')
