import pandas as pd
import sklearn.preprocessing
import sklearn.naive_bayes
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import math

# read data
df_train = pd.read_csv('train.csv', dtype = str)
df_test = pd.read_csv('test.csv', dtype = str)
df_test = df_test.drop(0)

colnames = df_test.columns
df_train.columns = colnames

race_dict = {True: '1.0', False: '0.0'}
df_train['race-binary'] = df_train['race']=='1'
df_train = df_train.replace({'race-binary': race_dict})
df_test['race-binary'] = df_test['race']=='1'
df_test = df_test.replace({'race-binary': race_dict})

# tracking selection stats
selection_w = []
selection_nw = []
overall_selection = []
s_ratios = []

# first we want to fit a simple Naive Bayes classifier using race

# using full race info
multirace_model = sklearn.naive_bayes.CategoricalNB().fit(X = df_train.drop(['income', 'race-binary'], axis = 1).values,
                                                 y = df_train['income'].values)

multirace_results = multirace_model.predict(df_test.drop(['income', 'race-binary'], axis = 1).values)

# using binary race only
basic_model = sklearn.naive_bayes.CategoricalNB().fit(X = df_train.drop(['income', 'race'], axis = 1).values,
                                                 y = df_train['income'].values)

basic_results = basic_model.predict(df_test.drop(['income', 'race'], axis = 1).values)
test_set_performance = np.mean(basic_results == df_test['income'])
df_test['predictions'] = basic_results


# check selection rates
df_train['predictions'] = debiased_model.predict(df_train.drop(['income', 'race-binary'], axis = 1).values)
selection_tot = np.mean(df_train['predictions'].astype(float))
overall_selection.append(selection_tot)
selection_white = np.mean(df_train.loc[df_test['race-binary'] == '1.0']['predictions'].astype(float))
selection_w.append(selection_white)
selection_oth = np.mean(df_train.loc[df_test['race-binary'] == '0.0']['predictions'].astype(float))
selection_nw.append(selection_oth)

selection_ratio = selection_oth/selection_white

s_ratios.append(selection_ratio)

# now we want to iterate, dropping variables correlated with race

selection_ratio = 0
vars_to_drop = ['income', 'race-binary', 'race']

while selection_ratio < 4/5:
#while len(vars_to_drop) < df_test.shape[1]-1:
    debiased_model = sklearn.naive_bayes.CategoricalNB().fit(X = df_train.drop(vars_to_drop, axis = 1).values,
                                                             y = df_train['income'].values)

    test_results = debiased_model.predict(df_test.drop(vars_to_drop + ['predictions'], axis = 1).values)
    test_set_performance = np.mean(test_results == df_test['income'])
    df_test['predictions'] = test_results


    # check selection rates
    df_train['predictions'] = debiased_model.predict(df_train.drop(vars_to_drop + ['predictions'], axis = 1).values)
    selection_tot = np.mean(df_train['predictions'].astype(float))
    overall_selection.append(selection_tot)
    selection_white = np.mean(df_train.loc[df_test['race-binary'] == '1.0']['predictions'].astype(float))
    selection_w.append(selection_white)
    selection_oth = np.mean(df_train.loc[df_test['race-binary'] == '0.0']['predictions'].astype(float))
    selection_nw.append(selection_oth)

    selection_ratio = selection_oth/selection_white
    s_ratios.append(selection_ratio)

    if selection_ratio < 4/5:
    #if len(vars_to_drop) < df_cv.shape[1]-1:

        names = []
        v_coefs = []
        for i in range(0,df_train.shape[1]-1) :
            if colnames[i] in vars_to_drop:
                continue
            else:
                names.append(colnames[i])
                v_coefs.append(cramers_v(df_train.iloc[:,i],df_train['race-binary']))

        corr_data = {'varname': names, 'correlation': v_coefs}
        corr_df = pd.DataFrame(corr_data).sort_values(by=['correlation'], ascending=False)

        # drop variable with the highest correlation to race
        vars_to_drop.append(corr_df.iloc[0,0])

        print ('Dropping ' + corr_df.iloc[0,0])

#export variables to compare
df_export = df_test[['race', 'income']]
df_export['basic']  = basic_results
df_export['multirace'] = multirace_results
df_export['hirevue'] = test_results


# helper function to define Cramer's V
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))
