import pandas as pd
import sklearn.preprocessing
import sklearn.naive_bayes
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import math

# read data
df = pd.read_csv('cleaned_acs_2.csv')
#df = df.drop(['RAC1P'], axis = 1)
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
accuracies = []
accuracies_w = []
accuracies_nw = []

falsepos = []
falsepos_w = []
falsepos_nw = []

falseneg = []
falseneg_w = []
falseneg_nw = []

ppv = []
ppv_w = []
ppv_nw = []

npv = []
npv_w = []
npv_nw = []

# tracking selection stats
selection_w = []
selection_nw = []
overall_selection = []
s_ratios = []

# first we want to fit a simple Naive Bayes classifier using race

basic_model = sklearn.naive_bayes.CategoricalNB().fit(X = df_train.drop(['PINCP', 'RAC1P'], axis = 1).values,
                                                 y = df_train['PINCP'].values)

test_results = basic_model.predict(df_test.drop(['PINCP', 'RAC1P'], axis = 1).values)
test_set_performance = np.mean(test_results == df_test['PINCP'])
df_test['predictions'] = test_results

accuracies.append(test_set_performance)

df_test['accuracy'] = df_test['predictions'] == df_test['PINCP']
df_test['truepos'] = (df_test['predictions'] == '1.0') & (df_test['PINCP']== '1.0')
df_test['falsepos'] = (df_test['predictions'] == '1.0') & (df_test['PINCP']== '0.0')
df_test['trueneg'] = (df_test['predictions'] == '0.0')& (df_test['PINCP'] == '0.0')
df_test['falseneg'] = (df_test['predictions'] == '0.0')& (df_test['PINCP'] == '1.0')

falsepos_tot = np.mean(df_test['falsepos'].astype(float))
falsepos.append(falsepos_tot)
falseneg_tot = np.mean(df_test['falseneg'].astype(float))
falseneg.append(falseneg_tot)

ppv_overall = np.sum(df_test['truepos'])/(np.sum(df_test['falsepos']) + np.sum(df_test['truepos']))
ppv.append(ppv_overall)
npv_overall = np.sum(df_test['trueneg'])/(np.sum(df_test['falseneg']) + np.sum(df_test['trueneg']))
npv.append(npv_overall)


# looking at accuracy by race
accuracy_white = np.mean(df_test.loc[df_test['race-binary'] == '1.0']['accuracy'].astype(float))
accuracies_w.append(accuracy_white)
accuracy_nonwhite = np.mean(df_test.loc[df_test['race-binary'] == '0.0']['accuracy'].astype(float))
accuracies_nw.append(accuracy_nonwhite)

falsepos_white = np.mean(df_test.loc[df_test['race-binary'] == '1.0']['falsepos'].astype(float))
falsepos_w.append(falsepos_white)
falsepos_nonwhite = np.mean(df_test.loc[df_test['race-binary'] == '0.0']['falsepos'].astype(float))
falsepos_nw.append(falsepos_nonwhite)

falseneg_white = np.mean(df_test.loc[df_test['race-binary'] == '1.0']['falseneg'].astype(float))
falseneg_w.append(falseneg_white)
falseneg_nonwhite = np.mean(df_test.loc[df_test['race-binary'] == '0.0']['falseneg'].astype(float))
falseneg_nw.append(falseneg_nonwhite)

ppv_white = np.sum(df_test.loc[df_test['race-binary'] == '1.0']['truepos'])/(np.sum(df_test.loc[df_test['race-binary'] == '1.0']['falsepos']) + np.sum(df_test.loc[df_test['race-binary'] == '1.0']['truepos']))
ppv_w.append(ppv_white)
ppv_nonwhite = np.sum(df_test.loc[df_test['race-binary'] == '0.0']['truepos'])/(np.sum(df_test.loc[df_test['race-binary'] == '0.0']['falsepos']) + np.sum(df_test.loc[df_test['race-binary'] == '0.0']['truepos']))
ppv_nw.append(ppv_nonwhite)


npv_white = np.sum(df_test.loc[df_test['race-binary'] == '1.0']['trueneg'])/(np.sum(df_test.loc[df_test['race-binary'] == '1.0']['falseneg']) + np.sum(df_test.loc[df_test['race-binary'] == '1.0']['trueneg']))
npv_w.append(npv_white)
npv_nonwhite = np.sum(df_test.loc[df_test['race-binary'] == '0.0']['trueneg'])/(np.sum(df_test.loc[df_test['race-binary'] == '0.0']['falseneg']) + np.sum(df_test.loc[df_test['race-binary'] == '0.0']['trueneg']))
npv_nw.append(npv_nonwhite)

# check selection rates
cv_results = basic_model.predict(df_cv.drop(['PINCP', 'RAC1P'], axis = 1).values)
df_cv['predictions'] = cv_results
selection_tot = np.mean(df_cv['predictions'].astype(float))
overall_selection.append(selection_tot)
selection_white = np.mean(df_cv.loc[df_cv['race-binary'] == '1.0']['predictions'].astype(float))
selection_w.append(selection_white)
selection_oth = np.mean(df_cv.loc[df_cv['race-binary'] == '0.0']['predictions'].astype(float))
selection_nw.append(selection_oth)

selection_ratio = selection_oth/selection_white

s_ratios.append(selection_ratio)


# now we want to iterate, dropping variables correlated with race

selection_ratio = 0
vars_to_drop = ['PINCP', 'race-binary', 'RAC1P']

#while selection_ratio < 4/5:
while len(vars_to_drop) < df_cv.shape[1]-1:
    debiased_model = sklearn.naive_bayes.CategoricalNB().fit(X = df_train.drop(vars_to_drop, axis = 1).values,
                                                             y = df_train['PINCP'].values)

    test_results = debiased_model.predict(df_test.drop(vars_to_drop + ['predictions', 'accuracy', 'falsepos', 'falseneg','truepos', 'trueneg'], axis = 1).values)
    test_set_performance = np.mean(test_results == df_test['PINCP'])
    df_test['predictions'] = test_results
    accuracies.append(test_set_performance)

    df_test['accuracy'] = df_test['predictions'] == df_test['PINCP']
    df_test['truepos'] = (df_test['predictions'] == '1.0') & (df_test['PINCP']== '1.0')
    df_test['falsepos'] = (df_test['predictions'] == '1.0') & (df_test['PINCP']== '0.0')
    df_test['trueneg'] = (df_test['predictions'] == '0.0')& (df_test['PINCP'] == '0.0')
    df_test['falseneg'] = (df_test['predictions'] == '0.0')& (df_test['PINCP'] == '1.0')

    falsepos_tot = np.mean(df_test['falsepos'].astype(float))
    falsepos.append(falsepos_tot)
    falseneg_tot = np.mean(df_test['falseneg'].astype(float))
    falseneg.append(falseneg_tot)

    ppv_overall = np.sum(df_test['truepos'])/(np.sum(df_test['falsepos']) + np.sum(df_test['truepos']))
    ppv.append(ppv_overall)
    npv_overall = np.sum(df_test['trueneg'])/(np.sum(df_test['falseneg']) + np.sum(df_test['trueneg']))
    npv.append(npv_overall)

    # looking at accuracy by race
    accuracy_white = np.mean(df_test.loc[df_test['race-binary'] == '1.0']['accuracy'].astype(float))
    accuracies_w.append(accuracy_white)
    accuracy_nonwhite = np.mean(df_test.loc[df_test['race-binary'] == '0.0']['accuracy'].astype(float))
    accuracies_nw.append(accuracy_nonwhite)

    falsepos_white = np.mean(df_test.loc[df_test['race-binary'] == '1.0']['falsepos'].astype(float))
    falsepos_w.append(falsepos_white)
    falsepos_nonwhite = np.mean(df_test.loc[df_test['race-binary'] == '0.0']['falsepos'].astype(float))
    falsepos_nw.append(falsepos_nonwhite)

    falseneg_white = np.mean(df_test.loc[df_test['race-binary'] == '1.0']['falseneg'].astype(float))
    falseneg_w.append(falseneg_white)
    falseneg_nonwhite = np.mean(df_test.loc[df_test['race-binary'] == '0.0']['falseneg'].astype(float))
    falseneg_nw.append(falseneg_nonwhite)

    ppv_white = np.sum(df_test.loc[df_test['race-binary'] == '1.0']['truepos'])/(np.sum(df_test.loc[df_test['race-binary'] == '1.0']['falsepos']) + np.sum(df_test.loc[df_test['race-binary'] == '1.0']['truepos']))
    ppv_w.append(ppv_white)
    ppv_nonwhite = np.sum(df_test.loc[df_test['race-binary'] == '0.0']['truepos'])/(np.sum(df_test.loc[df_test['race-binary'] == '0.0']['falsepos']) + np.sum(df_test.loc[df_test['race-binary'] == '0.0']['truepos']))
    ppv_nw.append(ppv_nonwhite)

    npv_white = np.sum(df_test.loc[df_test['race-binary'] == '1.0']['trueneg'])/(np.sum(df_test.loc[df_test['race-binary'] == '1.0']['falseneg']) + np.sum(df_test.loc[df_test['race-binary'] == '1.0']['trueneg']))
    npv_w.append(npv_white)
    npv_nonwhite = np.sum(df_test.loc[df_test['race-binary'] == '0.0']['trueneg'])/(np.sum(df_test.loc[df_test['race-binary'] == '0.0']['falseneg']) + np.sum(df_test.loc[df_test['race-binary'] == '0.0']['trueneg']))
    npv_nw.append(npv_nonwhite)

    # check selection rates
    cv_results = debiased_model.predict(df_cv.drop(vars_to_drop + ['predictions'], axis = 1).values)
    df_cv['predictions'] = cv_results
    selection_tot = np.mean(df_cv['predictions'].astype(float))
    overall_selection.append(selection_tot)
    selection_white = np.mean(df_cv.loc[df_cv['race-binary'] == '1.0']['predictions'].astype(float))
    selection_w.append(selection_white)
    selection_oth = np.mean(df_cv.loc[df_cv['race-binary'] == '0.0']['predictions'].astype(float))
    selection_nw.append(selection_oth)

    selection_ratio = selection_oth/selection_white
    s_ratios.append(selection_ratio)

    #if selection_ratio < 4/5:
    if len(vars_to_drop) < df_cv.shape[1]-1:

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


# let's plot some results
xlist = range(0, df_cv.shape[1]-3)

# plotting selection rates
overall = plt.plot(xlist, overall_selection, 'k:', label = 'Overall')
white = plt.plot(xlist, selection_w, 'b', label = 'White')
nonwhite = plt.plot(xlist, selection_nw, 'g', label = 'Nonwhite')
plt.axvline(x=12, color='r')
plt.legend(loc="upper right")
plt.ylabel('Selection Rate')
plt.xlabel('# of Variables Dropped')
plt.show()
plt.close()

#plotting accuracy
overall = plt.plot(xlist, accuracies, 'k:', label = 'Overall')
white = plt.plot(xlist, accuracies_w, 'b', label = 'White')
nonwhite = plt.plot(xlist, accuracies_nw, 'g', label = 'Nonwhite')
plt.axvline(x=12, color='r')
plt.legend(loc="upper right")
plt.ylabel('Accuracy')
plt.xlabel('# of Variables Dropped')
plt.show()
plt.close()

# plotting false positives
overall = plt.plot(xlist, falsepos, 'k:', label = 'Overall')
white = plt.plot(xlist, falsepos_w, 'b', label = 'White')
nonwhite = plt.plot(xlist, falsepos_nw, 'g', label = 'Nonwhite')
plt.axvline(x=12, color='r')
plt.legend(loc="upper right")
plt.ylabel('False Positive Rate')
plt.xlabel('# of Variables Dropped')
plt.show()
plt.close()

# plotting false negatives
overall = plt.plot(xlist, falseneg, 'k:', label = 'Overall')
white = plt.plot(xlist, falseneg_w, 'b', label = 'White')
nonwhite = plt.plot(xlist, falseneg_nw, 'g', label = 'Nonwhite')
plt.axvline(x=12, color='r')
plt.legend(loc="upper right")
plt.ylabel('False Negative Rate')
plt.xlabel('# of Variables Dropped')
plt.show()
plt.close()

# plotting ppv

overall = plt.plot(xlist, ppv, 'k:', label = 'Overall')
white = plt.plot(xlist, ppv_w, 'b', label = 'White')
nonwhite = plt.plot(xlist, ppv_nw, 'g', label = 'Nonwhite')
plt.axvline(x=12, color='r')
plt.legend(loc="upper right")
plt.ylabel('Positive Predictive Value')
plt.xlabel('# of Variables Dropped')
plt.show()
plt.close()


# plotting npv
overall = plt.plot(xlist, npv, 'k:', label = 'Overall')
white = plt.plot(xlist, npv_w, 'b', label = 'White')
nonwhite = plt.plot(xlist, npv_nw, 'g', label = 'Nonwhite')
plt.axvline(x=12, color='r')
plt.legend(loc="upper right")
plt.ylabel('Negative Predictive Value')
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
