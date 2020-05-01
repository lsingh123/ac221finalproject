
# # 2-NB Models

# In[1]:


import csv
import pandas as pd
import sklearn.preprocessing

file = 'cleaned_acs_2.csv'

with open(file, 'r') as fi:
    reader = csv.reader(fi)
    headers = next(reader)
    for i in range(len(headers)):
        print(i, headers[i])     


# ## Prep: Reading the data in

# In[2]:


race = 38
race_b = 39
income = 35
sex = 28
with open(file, 'r') as fi:
    reader = csv.reader(fi)
    h = next(reader)
    headers = []
    for i in range(len(h)):
      if i != race and i != race_b and i != income:
        headers.append(h[i])
    
    white = []
    nonwhite = []
    income_w = []
    income_n = []
    
    for line in reader:
        
        # remove id, income, race, and race_binary
        data = []
        for i in range(len(line)):
            if i != race and i != race_b and i != income:
                data.append(int(line[i]))


        # put in the correct racial category
        if int(line[race_b]) == 1:
            white.append(data)
            income_w.append(int(line[income]))
        else:
            nonwhite.append(data)
            income_n.append(int(line[income]))

df_white = pd.DataFrame(white, columns=headers)
print(df_white.head())
enc = sklearn.preprocessing.OrdinalEncoder()
enc.fit(df_white)
df_white = pd.DataFrame(enc.transform(df_white), index = None, columns = headers, dtype = str)

# ## Making the Models

# I am going to make the models using k-fold cross validation to try to improve the accuracy. Let's start with white people.

# In[3]:


import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import KFold

# returns indexes into fields and labels
def run_kfold(fields, labels):
    kf = KFold(n_splits=5)
    best = [], []
    best_accuracy = 0

    # train_index and test_index index into fields and labels
    for train_index, test_index in kf.split(fields):
        train_fields = fields.iloc[train_index].reset_index(drop = True)
        train_labels = [labels[i] for i in train_index]
        test_fields = fields.iloc[test_index].reset_index(drop = True)
        test_labels = [labels[i] for i in test_index]

        clf = CategoricalNB()
        clf.fit(train_fields, train_labels)

        try:
            res = clf.predict(test_fields).tolist()
        except IndexError:
            continue
        
        accuracy = []
        for i in range(len(res)):
            if res[i] == test_labels[i]:
                accuracy.append(1)
            else:
                accuracy.append(0)
        accuracy = [1 if res[i] == test_labels[i] else 0 for i in range(len(res))]
        acc = sum(accuracy)/len(accuracy)

        if (acc > best_accuracy):
            best = train_index, test_index
            best_accuracy = acc

        print("accuracy rate: ", acc)
    return best

#training_w, testing_w = run_kfold(df_white, income_w)


# And now for nonwhite people

# In[34]:

df_nwhite = pd.DataFrame(nonwhite, columns=headers)
print(df_nwhite.head())
enc = sklearn.preprocessing.OrdinalEncoder()
enc.fit(df_nwhite)
df_nwhite = pd.DataFrame(enc.transform(df_nwhite), index = None, columns = headers, dtype = str)
#training_n, testing_n = run_kfold(df_nwhite, income_n)


# ## Accuracy

# ### Differential Accuracy by Race Binary

# Let's examine the false positive and false negative rates respectively. **White people**:

# In[35]:


#training and testing index into fields and labels 
#so max(training) < len(fields) 
def run_model(training, testing, fields, labels):
    train_fields = fields.iloc[training].reset_index(drop = True)
    train_labels = [labels[i] for i in training]
    test_fields = fields.iloc[testing].reset_index(drop = True)
    test_labels = [labels[i] for i in testing]

    clf = CategoricalNB()
    clf.fit(train_fields, train_labels)

    res = clf.predict(test_fields).tolist()

    accuracy = []
    for i in range(len(res)):
        if res[i] == 1 and test_labels[i] == 0:
            accuracy.append(1)
        elif res[i] == 0 and test_labels[i] == 1:
            accuracy.append(-1)
        else:
            accuracy.append(0)

    fp = sum([1 if accuracy[i] == 1 else 0 for i in range(len(accuracy))])/len(accuracy)
    fn = sum([1 if accuracy[i] == -1 else 0 for i in range(len(accuracy))])/len(accuracy)
    acc = sum([1 if accuracy[i] == 0 else 0 for i in range(len(accuracy))])/len(accuracy)
    print("false positive rate: %4f" % fp)
    print("false negative rate: %4f" % fn)
    print("accuracy: %4f" % acc)
    return res, acc, fp, fn

print("Results of running the model for white people:")
res_w, acc_w, fp_w, fn_w = run_model(training_w, testing_w, df_white, income_w)


# And now for **nonwhite people:**

# In[36]:


print("Results of running the model for nonwhite people:")
res_n, acc_n, fp_n, fn_n = run_model(training_n, testing_n, df_nwhite, income_n)


# **Overall Accuracy**

# In[37]:


acc = (acc_n*len(res_n)+acc_w*len(res_w))/(len(res_n)+len(res_w))
fp = (fp_n*len(res_n)+fp_w*len(res_w))/(len(res_n)+len(res_w))
fn = (fn_n*len(res_n)+fn_w*len(res_w))/(len(res_n)+len(res_w))
print("false positive rate: %4f" % fp)
print("false negative rate: %4f" % fn)
print("accuracy: %4f" % acc)


# ### Differential Accuracy by Sex

# Let's examine differential accuracy for different subgroups. Let's initially break it down by sex.

# In[38]:


def calc_accuracy(res, labels):
    
    accuracy = []
    for i in range(len(res)):
        if res[i] == 1 and labels[i] == 0:
            accuracy.append(1)
        elif res[i] == 0 and labels[i] == 1:
            accuracy.append(-1)
        else:
            accuracy.append(0)
    return accuracy


def sex_accuracy(testing, fields, labels, res):
    test_fields = [fields[i] for i in testing]
    test_labels = [labels[i] for i in testing]
        
    male = []
    female = []
    
    for i in range(len(testing)):
        if fields[testing[i]][sex] == 1:
            male.append(i)
        if fields[testing[i]][sex] == 2:
            female.append(i)
            
    accuracy = calc_accuracy([res[i] for i in male], [test_labels[i] for i in male])
    fp = sum([1 if accuracy[i] == 1 else 0 for i in range(len(accuracy))])/len(accuracy)
    fn = sum([1 if accuracy[i] == -1 else 0 for i in range(len(accuracy))])/len(accuracy)
    print("Accuracy for male:")
    print("    false positive rate: %4f" % fp)
    print("    false negative rate: %4f" % fn)
    print("    accuracy: %4f" % (1-(fp + fn)))
    
    accuracy = calc_accuracy([res[i] for i in female], [test_labels[i] for i in female])
    fpf = sum([1 if accuracy[i] == 1 else 0 for i in range(len(accuracy))])/len(accuracy)
    fnf = sum([1 if accuracy[i] == -1 else 0 for i in range(len(accuracy))])/len(accuracy)
    print("Accuracy for female:")
    print("    false positive rate: %4f" % fpf)
    print("    false negative rate: %4f" % fnf)
    print("    accuracy: %4f" % (1-(fpf + fnf)))
    


# **Accuracy broken down by sex for nonwhite people**

# In[39]:


sex_accuracy(testing_n, df_nwhite, income_n, res_n)


# **Accuracy broken down by sex for white people**

# In[40]:


sex_accuracy(testing_w, white, income_w, res_w)


# ### Differential Accuracy By Race Categorical

# This is a little tricker. I first need some way to lookup up the race of each entry. Since I've removed unique identifiers, I'm going to have to go back and collect them. Let's make a list of just the racial categories.

# In[63]:


race_list = []
with open(file, 'r') as fi:
    reader = csv.reader(fi)
    headers = next(reader)
    
    for line in reader:
        if (int(line[race_b]) != 1):
            race_list.append(int(line[race]))
#f_nwhite['race'] = race_list

# **Accuracy by specific racial group for nonwhite people**

# In[61]:


import pandas as pd
def race_accuracy(testing, labels, res):
    test_labels = [labels[i] for i in testing]
    data = {"race":[], "false positive":[], "false negative":[], "accuracy":[], 'count':[]}
    
    for r in range(max(race_list)+1):

        # indeces within res for the current group
        current = []
        for i in range(len(testing)):
            if (race_list[testing[i]] == r):
                current.append(i)

        # calculate accuracy
        accuracy = calc_accuracy([res[i] for i in current], [test_labels[i] for i in current])
        try:
            fp = sum([1 if accuracy[i] == 1 else 0 for i in range(len(accuracy))])/len(accuracy)
            fn = sum([1 if accuracy[i] == -1 else 0 for i in range(len(accuracy))])/len(accuracy)
            print("Results for race %d:" % r)
            print("    %d ENTRIES TOTAL" % len(current))
            print("    false positive rate: %4f" % fp)
            print("    false negative rate: %4f" % fn)
            print("    accuracy: %4f" % (1-(fp + fn)))
            data["race"].append(r)
            data["false positive"].append(fp)
            data["false negative"].append(fn)
            data["accuracy"].append((1-(fp + fn)))
            data["count"].append(len(current))
        except ZeroDivisionError:
            print("No results for race %d" % r)
    
    return pd.DataFrame(data)

df = race_accuracy(testing_n, income_n, res_n)
with open("2nb_categorical.csv", "w") as fo:
    writer = csv.writer(fo)
    writer.writerow(['race', 'false positive', 'false negative', 'accuracy'])
    for index, row in df.iterrows():
        writer.writerow([row['race'], row['false positive'], row['false negative'], row['accuracy'], row['count']])


# In[65]:


print(len(testing_w))


# In[64]:


print(df['count'])


# # n-NB Models

# In[46]:


# race_fields[i] = indeces for race i
race_indeces = [[] for i in range(max(race_list)+1)]
race_labels = [[] for i in range(max(race_list)+1)]

# iterate over each row in the data
for i in range(len(nonwhite)):
    race_indeces[race_list[i]].append(i)
    race_labels[race_list[i]].append(income_n[i])
for race_i in race_indeces:
    print(len(race_i)/5)


# In[59]:


def run_race(race, fields, labels):
        
    print("TRAINING")
    print("-----------------------")
    print()
    
    training, testing = run_kfold(fields, labels)
    
    print()
    print("RUNNING THE BEST MODEL")
    print("-----------------------")
    print()
    
    res, acc, fp, fn = run_model(training, testing, fields, labels)
    
    '''print()
    print("ACCURACY BY SEX")
    print("-----------------------")
    print()
    
    sex_accuracy(testing, fields, labels, res)'''
    return res, acc, fp, fn 


# In[60]:


fp_overall = 0
fn_overall = 0
accuracy_overall = 0
total = 0
with open("nNB.csv", "w") as fo:
    writer = csv.writer(fo)
    writer.writerow(['race', 'false positive', 'false negative', 'accuracy'])
    
    for i in range(len(race_indeces)):
        if len(race_indeces[i]) > 350:
            print()
            print("RACE %d " % i)
            print()
            fields = df_nwhite.iloc[race_indeces[i]].reset_index(drop = True)
            res, acc, fp, fn = run_race(i, fields, race_labels[i])
            writer.writerow([i, fp, fn, acc])
            fp_overall += fp*len(res)
            fn_overall += fn*len(res)
            accuracy_overall += acc*len(res)
            total += len(res)
print()
print("false positive rate: %4f" % (fp_overall/total))
print("false negative rate: %4f" % (fn_overall/total))
print("accuracy: %4f" % (accuracy_overall/total))


# In[ ]:

# write testing and training indeces to files for both white and nonwhite 

with open("test.csv", "w") as fo_test, open("train.csv", "w") as fo_train:
    writer_test = csv.writer(fo_test)
    writer_train = csv.writer(fo_train)
    headers = df_nwhite.columns.tolist()
    headers.append("race")
    headers.append("income")
    writer_test.writerow(headers)
    writer_test.writerow(headers)
    for i in range(len(race_indeces)):
        if len(race_indeces[i]) > 0:
            fields = df_nwhite.iloc[race_indeces[i]].reset_index(drop = True)
            training, testing = run_kfold(fields, race_labels[i])
            training = training.tolist()
            testing = testing.tolist()
            for index in testing:
                writer_test.writerow(fields.iloc[index].tolist() + [i] + [race_labels[i][index]])
            for index in training:
                writer_train.writerow(fields.iloc[index].tolist() + [i] + [race_labels[i][index]])
    training, testing = run_kfold(df_white, income_w)
    training = training.tolist()
    testing = testing.tolist()
    for index in testing:
        writer_test.writerow(df_white.iloc[index].tolist() + [1] + [income_w[index]])
    for index in training:
        writer_train.writerow(df_white.iloc[index].tolist() + [1] + [income_w[index]])


# In[]:
import csv
import pandas as pd 

with open("test.csv", "r") as fi:
    reader = csv.reader(fi)
    headers = next(reader)
    next(reader)
    headers.append("group")
    lines = []
    
    for line in reader:
        lines.append(line + ["test"])
        


# In[]:

with open("train.csv", "r") as fi:
    reader = csv.reader(fi)
    h = next(reader)
    
    for line in reader:
        lines.append(line + ["train"])
        
data = pd.DataFrame(lines, columns=headers)
print(data.head(15))


# In[]:
print(data.columns)

# In[]:

# results for nnb
from sklearn.naive_bayes import CategoricalNB
import numpy as np

 
results = ["-1"] * data.shape[0]
for race in data['race'].unique():
    train = data[(data['race'] == race) & (data['group'] == 'train')]
    test = data[(data['race'] == race) & (data['group'] == 'test')]

    clf = CategoricalNB()
    clf.fit(train.drop(columns=['race', 'income', 'group']), train['income'])
    
    res = clf.predict(test.drop(columns=['race', 'income', 'group'])).tolist()
    
    res_index = 0
    test_indeces = np.where((data['race'] == race) & (data['group'] == 'test'))[0].tolist()
    for test_index in test_indeces:
        results[test_index] = res[res_index]
        res_index += 1
        
data['nnb_res'] = results
print(results[0:15])

# In[]:

# 2NB 

results = ["-1"]*data.shape[0]

# handle white people
train = data[(data['race'] == '1') & (data['group'] == 'train')]
test = data[(data['race'] == '1') & (data['group'] == 'test')]
    
clf = CategoricalNB()
clf.fit(train.drop(columns=['race', 'income', 'group', 'nnb_res']), train['income'])
    
res = clf.predict(test.drop(columns=['race', 'income', 'group', 'nnb_res'])).tolist()
    
res_index = 0
test_indeces = np.where((data['race'] == '1') & (data['group'] == 'test'))[0].tolist()
for test_index in test_indeces:
   results[test_index] = res[res_index]
   res_index += 1

# handle non white people
train = data[(data['race'] != '1') & (data['group'] == 'train')]
test = data[(data['race'] != '1') & (data['group'] == 'test')]
    
clf = CategoricalNB()
clf.fit(train.drop(columns=['race', 'income', 'group', 'nnb_res']), train['income'])
    
res = clf.predict(test.drop(columns=['race', 'income', 'group', 'nnb_res'])).tolist()
    
res_index = 0
test_indeces = np.where((data['race'] != '1') & (data['group'] == 'test'))[0].tolist()
for test_index in test_indeces:
   results[test_index] = res[res_index]
   res_index += 1

data['2nb_res'] = results

# In[]:

# basic accuracy counts

print(data[data['nnb_res'] == data['2nb_res']].shape)
print(data[data['nnb_res'] == data['income']].shape)
print(data[data['income'] == data['2nb_res']].shape)

# In[]:
print(data.columns)

# In[]:

tested = data[data["group"] == "test"][['race', 'income', 'nnb_res', '2nb_res']]
tested = tested.values.tolist()

with open("results.csv", "w") as fo:
    writer = csv.writer(fo)
    writer.writerow(['race', 'income', 'nnb_res', '2nb_res'])
    
    for row in tested:
        writer.writerow(row)