import pandas as pd
import sklearn.preprocessing
import sklearn.naive_bayes
import numpy as np
import math

# read data
df = pd.read_csv('cleaned2.csv')
df = df.drop(['HRHHID', 'PTDTRACE'], axis = 1)
colnames = df.columns

# initialize ordinal encoder
enc = sklearn.preprocessing.OrdinalEncoder()

# fit ordinal encoder to features
enc.fit(df)

# transform df to work with CategoricalNB
df = pd.DataFrame(enc.transform(df), index = None, columns = colnames, dtype = str)

# create white and non-white subsets -- train/test splits within that

# first create model for white subset
np.random.seed(1)
white_df = df.loc[df['race-binary'] == '1.0'].reset_index(drop = True)
n_white = white_df.shape[0]
training_indices = np.random.choice(range(n_white), size = math.floor(0.8 * n_white))
test_indices = list(set(range(n_white)).difference(training_indices))
white_df_train = white_df.iloc[training_indices].reset_index(drop = True)
white_df_test = white_df.iloc[test_indices].reset_index(drop = True)

white_model = sklearn.naive_bayes.CategoricalNB().fit(X = white_df_train.drop(['HEFAMINC'], axis = 1).values,
                                                 y = white_df_train['HEFAMINC'].values)

test_results = white_model.predict(white_df_test.drop(['HEFAMINC'], axis = 1).values)
test_set_performance = np.mean(test_results == white_df_test['HEFAMINC'])

# model for non-white subset
np.random.seed(7)
non_white_df = df.loc[df['race-binary'] == '0.0'].reset_index(drop = True)
n_non_white = non_white_df.shape[0]
training_indices_nw = np.random.choice(range(n_non_white), size = math.floor(0.8 * n_non_white))
test_indices_nw = list(set(range(n_non_white)).difference(training_indices_nw))
non_white_df_train = non_white_df.iloc[training_indices_nw].reset_index(drop = True)
non_white_df_test = non_white_df.iloc[test_indices_nw].reset_index(drop = True)

non_white_model = sklearn.naive_bayes.CategoricalNB().fit(X = non_white_df_train.drop(['HEFAMINC'], axis = 1).values,
                                                 y = non_white_df_train['HEFAMINC'].values)

test_results_nw = non_white_model.predict(non_white_df_test.drop(['HEFAMINC'], axis = 1).values)
test_set_performance_nw = np.mean(test_results_nw == non_white_df_test['HEFAMINC'])
