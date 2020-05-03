import pandas as pd
import sklearn.preprocessing
import sklearn.naive_bayes
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import math

# read data
df = pd.read_csv('cleaned_acs_2.csv')
df = df.drop(['race-binary'], axis = 1)
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


# tracking accuracy stats
accuracy_dict = dict()
falsepos_dict = dict()
falseneg_dict = dict()
# tracking selection stats
selection_dict = dict()
s_ratios = []

# first we want to fit a simple Naive Bayes classifier using race

basic_model = sklearn.naive_bayes.CategoricalNB().fit(X = df_train.drop(['PINCP'], axis = 1).values,
                                                 y = df_train['PINCP'].values)

test_results = basic_model.predict(df_test.drop(['PINCP'], axis = 1).values)
test_set_performance = np.mean(test_results == df_test['PINCP'])
df_test['predictions'] = test_results

accuracy_dict.setdefault('overall', []).append(test_set_performance)


df_test['accuracy'] = df_test['predictions'] == df_test['PINCP']
df_test['falsepos'] = (df_test['predictions'] == '1.0') & (df_test['PINCP']== '0.0')
df_test['falseneg'] = (df_test['predictions'] == '0.0')& (df_test['PINCP'] == '1.0')

falsepos_tot = np.mean(df_test['falsepos'].astype(float))
falsepos_dict.setdefault('overall', []).append(falsepos_tot)

falseneg_tot = np.mean(df_test['falseneg'].astype(float))
falseneg_dict.setdefault('overall', []).append(falseneg_tot)


# looking at accuracy by race
races = ['1.0', '8.0', '0.0', '5.0', '2.0', '7.0', '6.0', '4.0', '3.0']

for race in races:
    acc = np.mean(df_test.loc[df_test['RAC1P'] == race]['accuracy'].astype(float))
    accuracy_dict.setdefault(race, []).append(acc)

    fp = np.mean(df_test.loc[df_test['RAC1P'] == race]['falsepos'].astype(float))
    falsepos_dict.setdefault(race, []).append(fp)

    fn = np.mean(df_test.loc[df_test['RAC1P'] == race]['falseneg'].astype(float))
    falseneg_dict.setdefault(race, []).append(fn)


# check selection rates
cv_results = basic_model.predict(df_cv.drop(['PINCP'], axis = 1).values)
df_cv['predictions'] = cv_results
selection_tot = np.mean(df_cv['predictions'].astype(float))
selection_dict.setdefault('overall', []).append(selection_tot)
for race in races:
    s_rate = np.mean(df_cv.loc[df_cv['RAC1P'] == race]['predictions'].astype(float))
    selection_dict.setdefault(race, []).append(s_rate)

selection_rates = df_cv.astype(float).groupby('RAC1P')['predictions'].mean()
selection_ratio = min(selection_rates)/max(selection_rates)
s_ratios.append(selection_ratio)

# now we want to iterate, dropping variables correlated with race

selection_ratio = 0
vars_to_drop = ['PINCP', 'RAC1P']

while selection_ratio < 4/5:
#while len(vars_to_drop) < df_cv.shape[1]-1:
    debiased_model = sklearn.naive_bayes.CategoricalNB().fit(X = df_train.drop(vars_to_drop, axis = 1).values,
                                                             y = df_train['PINCP'].values)

    test_results = debiased_model.predict(df_test.drop(vars_to_drop + ['predictions', 'accuracy', 'falsepos', 'falseneg'], axis = 1).values)
    test_set_performance = np.mean(test_results == df_test['PINCP'])
    df_test['predictions'] = test_results
    accuracy_dict.setdefault('overall', []).append(test_set_performance)


    df_test['accuracy'] = df_test['predictions'] == df_test['PINCP']
    df_test['falsepos'] = (df_test['predictions'] == '1.0') & (df_test['PINCP']== '0.0')
    df_test['falseneg'] = (df_test['predictions'] == '0.0')& (df_test['PINCP'] == '1.0')

    falsepos_tot = np.mean(df_test['falsepos'].astype(float))
    falsepos_dict.setdefault('overall', []).append(falsepos_tot)

    falseneg_tot = np.mean(df_test['falseneg'].astype(float))
    falseneg_dict.setdefault('overall', []).append(falseneg_tot)


    # looking at accuracy by race
    race_list = ['1.0', '8.0', '0.0', '5.0', '2.0', '7.0', '6.0', '4.0', '3.0']

    for race in races:
        acc = np.mean(df_test.loc[df_test['RAC1P'] == race]['accuracy'].astype(float))
        accuracy_dict.setdefault(race, []).append(acc)

        fp = np.mean(df_test.loc[df_test['RAC1P'] == race]['falsepos'].astype(float))
        falsepos_dict.setdefault(race, []).append(fp)

        fn = np.mean(df_test.loc[df_test['RAC1P'] == race]['falseneg'].astype(float))
        falseneg_dict.setdefault(race, []).append(fn)

    # check selection rates
    cv_results = debiased_model.predict(df_cv.drop(vars_to_drop + ['predictions'], axis = 1).values)
    df_cv['predictions'] = cv_results
    selection_tot = np.mean(df_cv['predictions'].astype(float))
    selection_dict.setdefault('overall', []).append(selection_tot)

    for race in races:
        s_rate = np.mean(df_cv.loc[df_cv['RAC1P'] == race]['predictions'].astype(float))
        selection_dict.setdefault(race, []).append(s_rate)

    selection_rates = df_cv.astype(float).groupby('RAC1P')['predictions'].mean()
    selection_ratio = min(selection_rates)/max(selection_rates)
    s_ratios.append(selection_ratio)

    if selection_ratio < 4/5:
    #if len(vars_to_drop) < df_cv.shape[1]-1:

        names = []
        v_coefs = []
        for i in range(0,df_cv.shape[1]-1) :
            if colnames[i] in vars_to_drop:
                continue
            else:
                names.append(colnames[i])
                v_coefs.append(cramers_v(df_cv.iloc[:,i],df_cv['RAC1P']))

        corr_data = {'varname': names, 'correlation': v_coefs}
        corr_df = pd.DataFrame(corr_data).sort_values(by=['correlation'], ascending=False)

        # drop variable with the highest correlation to race
        vars_to_drop.append(corr_df.iloc[0,0])

        print ('Dropping ' + corr_df.iloc[0,0])


# let's plot some results
xlist = range(0, df_cv.shape[1]-2)

# plotting selection rates
overall = plt.plot(xlist, overall_selection, 'k:', label = 'Overall')
white = plt.plot(xlist, selection_w, 'b', label = 'White')
nonwhite = plt.plot(xlist, selection_nw, 'g', label = 'Nonwhite')
plt.axvline(x=6, color='r')
plt.legend(loc="upper right")
plt.ylabel('Selection Rate')
plt.xlabel('# of Variables Dropped')
plt.show()
plt.close()

#plotting accuracy
overall = plt.plot(xlist, accuracies, 'k:', label = 'Overall')
white = plt.plot(xlist, accuracies_w, 'b', label = 'White')
nonwhite = plt.plot(xlist, accuracies_nw, 'g', label = 'Nonwhite')
plt.axvline(x=6, color='r')
plt.legend(loc="upper right")
plt.ylabel('Accuracy')
plt.xlabel('# of Variables Dropped')
plt.show()
plt.close()

# plotting false positives
overall = plt.plot(xlist, falsepos, 'k:', label = 'Overall')
white = plt.plot(xlist, falsepos_w, 'b', label = 'White')
nonwhite = plt.plot(xlist, falsepos_nw, 'g', label = 'Nonwhite')
plt.axvline(x=6, color='r')
plt.legend(loc="upper right")
plt.ylabel('False Positive Rate')
plt.xlabel('# of Variables Dropped')
plt.show()
plt.close()

# plotting false negatives
overall = plt.plot(xlist, falseneg, 'k:', label = 'Overall')
white = plt.plot(xlist, falseneg_w, 'b', label = 'White')
nonwhite = plt.plot(xlist, falseneg_nw, 'g', label = 'Nonwhite')
plt.axvline(x=6, color='r')
plt.legend(loc="upper right")
plt.ylabel('False Negative Rate')
plt.xlabel('# of Variables Dropped')
plt.show()
plt.close()


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
