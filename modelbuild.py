import pandas as pd
import sklearn.preprocessing
import sklearn.naive_bayes
import numpy as np
import scipy.stats as ss
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

# split into training, cross validation (for checking 4/5 compliance), and test
np.random.seed(1)
n_df = df.shape[0]
training_indices = np.random.choice(range(n_df), size = math.floor(0.8 * n_df), replace = False)
other_indices = list(set(range(n_df)).difference(training_indices))
cv_indices = np.random.choice(other_indices, size = math.floor(0.1 * n_df), replace = False)
test_indices = list(set(other_indices).difference(cv_indices))


df_train = df.iloc[training_indices].reset_index(drop = True)
df_cv = df.iloc[cv_indices].reset_index(drop = True)
df_test = df.iloc[test_indices].reset_index(drop = True)


basic_model = sklearn.naive_bayes.CategoricalNB().fit(X = df_train.drop(['HEFAMINC'], axis = 1).values,
                                                 y = df_train['HEFAMINC'].values)

test_results = basic_model.predict(df_test.drop(['HEFAMINC'], axis = 1).values)
test_set_performance = np.mean(test_results == df_test['HEFAMINC'])
df_test['predictions'] = test_results

print(test_set_performance)

# looking at accuracy by race
df_test['falsepos'] = (df_test['predictions'] == '1.0') & (df_test['HEFAMINC']== '0.0')
df_test['falseneg'] = (df_test['predictions'] == '0.0')& (df_test['HEFAMINC'] == '1.0')
df_test['accuracy'] = df_test['predictions'] == df_test['HEFAMINC']

accuracy_white = np.mean(df_test.loc[df_test['race-binary'] == '1.0']['accuracy'].astype(float))
accuracy_nonwhite = np.mean(df_test.loc[df_test['race-binary'] == '0.0']['accuracy'].astype(float))

falsepos_white = np.mean(df_test.loc[df_test['race-binary'] == '1.0']['falsepos'].astype(float))
falsepos_nonwhite = np.mean(df_test.loc[df_test['race-binary'] == '0.0']['falsepos'].astype(float))

falseneg_white = np.mean(df_test.loc[df_test['race-binary'] == '1.0']['falseneg'].astype(float))
falseneg_nonwhite = np.mean(df_test.loc[df_test['race-binary'] == '0.0']['falseneg'].astype(float))

print(accuracy_white)
print(accuracy_nonwhite)
print(falsepos_white)
print(falsepos_nonwhite)
print(falseneg_white)
print(falseneg_nonwhite)

# check selection rates
cv_results = basic_model.predict(df_cv.drop(['HEFAMINC'], axis = 1).values)
df_cv['predictions'] = cv_results
selection_white = np.mean(df_cv.loc[df_cv['race-binary'] == '1.0']['predictions'].astype(float))
selection_oth = np.mean(df_cv.loc[df_cv['race-binary'] == '0.0']['predictions'].astype(float))

selection_ratio = selection_white/selection_oth
print(selection_ratio)


# now we want to iterate

accuracies = []
s_ratios = []
selection_ratio = 0
vars_to_drop = ['HEFAMINC', 'race-binary']

while selection_ratio < 4/5:
    debiased_model = sklearn.naive_bayes.CategoricalNB().fit(X = df_train.drop(vars_to_drop, axis = 1).values,
                                                             y = df_train['HEFAMINC'].values)

    test_results = debiased_model.predict(df_test.drop(vars_to_drop + ['predictions'], axis = 1).values)
    test_set_performance = np.mean(test_results == df_test['HEFAMINC'])
    df_test['predictions'] = test_results
    accuracies.append(test_set_performance)

    # check selection rates

    cv_results = debiased_model.predict(df_cv.drop(vars_to_drop + ['predictions'], axis = 1).values)
    df_cv['predictions'] = cv_results
    selection_white = np.mean(df_cv.loc[df_cv['race-binary'] == '1.0']['predictions'].astype(float))
    selection_oth = np.mean(df_cv.loc[df_cv['race-binary'] == '0.0']['predictions'].astype(float))


    selection_ratio = selection_white/selection_oth
    s_ratios.append(selection_ratio)

    if selection_ratio < 4/5:

        names = []
        v_coefs = []
        for i in range(0,df_cv.shape[1]-1) :
            if colnames[i] in vars_to_drop:
                continue
            else:
                names.append(colnames[i])
                v_coefs.append(cramers_v(df_cv.iloc[:,i],df_cv['race-binary']))

        corr_data = {'varname': names, 'correlation': v_coefs}
        corr_df = pd.DataFrame(corr_data).sort_values(by=['correlation'], ascending=False)

        # drop variable with the highest correlation to race
        vars_to_drop.append(corr_df.iloc[0,0])

        print ('Dropping ' + corr_df.iloc[0,0])

# looking at accuracy by race
df_test['falsepos'] = (df_test['predictions'] == '1.0') & (df_test['HEFAMINC']== '0.0')
df_test['falseneg'] = (df_test['predictions'] == '0.0')& (df_test['HEFAMINC'] == '1.0')
df_test['accuracy'] = df_test['predictions'] == df_test['HEFAMINC']

accuracy_white = np.mean(df_test.loc[df_test['race-binary'] == '1.0']['accuracy'].astype(float))
accuracy_nonwhite = np.mean(df_test.loc[df_test['race-binary'] == '0.0']['accuracy'].astype(float))

falsepos_white = np.mean(df_test.loc[df_test['race-binary'] == '1.0']['falsepos'].astype(float))
falsepos_nonwhite = np.mean(df_test.loc[df_test['race-binary'] == '0.0']['falsepos'].astype(float))

falseneg_white = np.mean(df_test.loc[df_test['race-binary'] == '1.0']['falseneg'].astype(float))
falseneg_nonwhite = np.mean(df_test.loc[df_test['race-binary'] == '0.0']['falseneg'].astype(float))

print(accuracy_white)
print(accuracy_nonwhite)
print(falsepos_white)
print(falsepos_nonwhite)
print(falseneg_white)
print(falseneg_nonwhite)


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
