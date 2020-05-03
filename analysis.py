#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 21:30:27 2020

@author: lavanyasingh
"""

# In[]:

import csv
import pandas as pd

lines = []
with open("hirevue.csv", "r") as fi:
    reader = csv.reader(fi)
    headers = next(reader)[1:]
    
    for line in reader:
        lines.append(line[1:])

res = pd.DataFrame(lines, columns=headers)
print(res.head(5))

# In[]:

nnb = []
twonb = []
with open("results.csv", "r") as fi:
    reader = csv.reader(fi)
    
    print(next(reader))
    
    for line in reader:
        nnb.append(line[2])
        twonb.append(line[3])

res["nnb"] = nnb
res["twonb"] = twonb
print(res.head(5))


res["race"] = res["race"].apply(pd.to_numeric)
res_old = res.copy()
print(res_old.race.unique())

res["race"] = [3 if i == 4 or i == 5 else i for i in res["race"]]
res["race"] = [i-2 if i > 3 else i for i in res["race"]]

print(res.race.unique())

# In[]:

from sklearn.metrics import confusion_matrix

models = ['basic', 'multirace', 'hirevue', 'nnb', 'twonb']
acc = pd.DataFrame(models, columns=['model'])
ppv = []
npv = []
for model in models:
    conf = confusion_matrix(res["income"], res[model])
    ppv.append((conf[0][1])/(conf[1][1]+conf[0][1]))
    npv.append(conf[1][0]/(conf[1][0]+conf[0][0]))
acc["ppv"] = ppv
acc["npv"] = npv

print(acc)

# In[]:

fp = []
fn = []
accuracy = []
for model in models:
    pos = res[(res["income"] == '0') & (res[model] == '1')]
    fp.append(pos.shape[0]/res.shape[0])
    neg = res[(res["income"] == '1') & (res[model] == '0')]
    fn.append(neg.shape[0]/res.shape[0])
    diffs = res[res["income"] == res[model]]
    accuracy.append(diffs.shape[0]/res.shape[0])
acc["accuracy"] = accuracy
print(acc)

# In[]:

races_ppv = [[] for i in res["race"].unique()]
races_npv = [[] for i in res["race"].unique()]
races_acc = [[] for i in res["race"].unique()]
for model in models:
    for race in res["race"].unique():
        data = res[res["race"] == race]
        conf = confusion_matrix(data["income"], data[model])
        races_ppv[int(race) - 1].append((conf[0][1])/(conf[1][1]+conf[0][1]))
        races_npv[int(race) - 1].append(conf[1][0]/(conf[1][0]+conf[0][0]))
        
        diffs = data[data["income"] == data[model]]
        races_acc[int(race)-1].append(diffs.shape[0]/data.shape[0])
        
for race in res["race"].unique():
    acc["ppv_"+str(race)] = races_ppv[race-1]
    acc["npv_"+str(race)] = races_npv[race-1]
    acc["accuracy_"+str(race)] = races_acc[race-1]
print(acc)


# In[]:

nonwhite_ppv = []
nonwhite_npv = []
nonwhite_acc = []
for model in models:
    data = res[res["race"] != '1']
    conf = confusion_matrix(data["income"], data[model])
    nonwhite_ppv.append((conf[0][1])/(conf[1][1]+conf[0][1]))
    nonwhite_npv.append(conf[1][0]/(conf[1][0]+conf[0][0]))
    diffs = data[data["income"] == data[model]]
    nonwhite_acc.append(diffs.shape[0]/data.shape[0])
acc["nonwhite_ppv"] = nonwhite_ppv
acc["nonwhite_npv"] = nonwhite_npv
acc["nonwhite_accuracy"] = nonwhite_acc
print(acc)


# In[]:

# differential ppv

race_map = {"1":"white", "2":"black", "3":"indigenous", "4":"asian", "5":"pacific islander", "6":"other", "7":"mixed"}
tograph = acc[["ppv_"+str(i) for i in res.race.unique()] + ["model"]]
tograph = tograph.rename(columns={"ppv_"+str(key):value for key,value in race_map.items()})
tograph = tograph.drop(0)
tograph.at[1, "model"] = "basic"
ax = tograph.set_index("model").plot.bar()
ax.get_legend().remove()
ax.set_ylabel("ppv")

# In[]:

# differential npv

tograph = acc[["npv_"+str(i) for i in res.race.unique()] + ["model"]]
tograph = tograph.rename(columns={"npv_"+str(key):value for key,value in race_map.items()})
tograph = tograph.drop(0)
tograph.at[1, "model"] = "basic"
ax = tograph.set_index("model").plot.bar()
ax.get_legend().remove()
ax.set_ylabel("npv")

# In[]:

# differential accuracy

tograph = acc[["accuracy_"+str(i) for i in res.race.unique()] + ["model"]]
tograph = tograph.rename(columns={"accuracy_"+str(key):value for key,value in race_map.items()})
tograph = tograph.drop(0)
tograph.at[1, "model"] = "basic"
ax = tograph.set_index("model").plot.bar()
ax.legend(bbox_to_anchor=(1.1, 1.05))
ax.set_ylabel("accuracy")

# In[]:

# accuracy for nonwhite, white, and overall 

tograph = acc[["nonwhite_accuracy", "accuracy_1", "accuracy", "model"]]
tograph = tograph.rename(columns={"nonwhite_accuracy":"nonwhite", "accuracy_1":"white", "accuracy":"overall"})
tograph = tograph.drop(0)
tograph.at[1, "model"] = "basic"
ax = tograph.set_index("model").plot.bar()
ax.legend(bbox_to_anchor=(1.1, 1.05))
ax.set_ylabel("accuracy")

# In[]:

# ppv for nonwhite, white, and overall

tograph = acc[["nonwhite_ppv", "ppv_1", "ppv", "model"]]
tograph = tograph.rename(columns={"nonwhite_ppv":"nonwhite", "ppv_1":"white", "ppv":"overall"})
tograph = tograph.drop(0)
tograph.at[1, "model"] = "basic"
ax = tograph.set_index("model").plot.bar()
ax.get_legend().remove()
ax.set_ylabel("ppv")

# In[]:

# npv for nonwhite, white, and overall

tograph = acc[["nonwhite_npv", "npv_1", "npv", "model"]]
tograph = tograph.rename(columns={"nonwhite_npv":"nonwhite", "npv_1":"white", "npv":"overall"})
tograph = tograph.drop(0)
tograph.at[1, "model"] = "basic"
ax = tograph.set_index("model").plot.bar()
ax.get_legend().remove()
ax.set_ylabel("npv")

# In[]:

from scipy.stats import levene

# brown forsythe to see if variance in diff accuracy across models is significantly diff

for model in models[2:]:
    accuracies = []
    basic = []
    for race in res["race"].unique():
        accuracies.append(acc[acc["model"] == model]["accuracy_"+str(race)].tolist()[0])
        basic.append(acc[acc["model"] == "multirace"]["accuracy_"+str(race)].tolist()[0])
    print(model, levene(accuracies, basic, center="median"))

# same for nnb versus hirevue 
    
nnb_acc = []
hirevue_acc = []
for race in res["race"].unique():
    nnb_acc.append(acc[acc["model"] == "nnb"]["accuracy_"+str(race)].tolist()[0])
    hirevue_acc.append(acc[acc["model"] == "hirevue"]["accuracy_"+str(race)].tolist()[0])
print(levene(nnb_acc, hirevue_acc, center="median"))

# In[]:

# same for ppv
for model in models[2:]:
    ppv = []
    basic = []
    for race in res["race"].unique():
        ppv.append(acc[acc["model"] == model]["ppv_"+str(race)].tolist()[0])
        basic.append(acc[acc["model"] == "multirace"]["ppv_"+str(race)].tolist()[0])
    print(model, levene(accuracies, basic, center="median"))

# In[]:

# and npv
for model in models[2:]:
    npv = []
    basic = []
    for race in res["race"].unique():
        npv.append(acc[acc["model"] == model]["npv_"+str(race)].tolist()[0])
        basic.append(acc[acc["model"] == "multirace"]["npv_"+str(race)].tolist()[0])
    print(model, levene(accuracies, basic, center="median"))

# In[]:

from statistics import variance 

# table of variances in npv, accuracy, and ppv
    
npv_var = []
ppv_var = []
accuracy_var = []
for model in models[1:]:
    temp1 = variance(acc[acc["model"] == model]["ppv_"+str(race)].tolist()[0] for race in res["race"].unique())
    ppv_var.append(temp1)
    
    temp1 = variance(acc[acc["model"] == model]["npv_"+str(race)].tolist()[0] for race in res["race"].unique())
    npv_var.append(temp1)
    
    temp1 = variance(acc[acc["model"] == model]["accuracy_"+str(race)].tolist()[0] for race in res["race"].unique())
    accuracy_var.append(temp1)
variance = pd.DataFrame([npv_var, ppv_var, accuracy_var], columns=models[1:])
variance["measure"] = ["npv", "ppv", "accuracy"]
variance = variance[["measure"]+variance.columns.tolist()[:-1]]
print(variance.head())
variance.to_html('variance.html')
    