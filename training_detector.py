#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier, StackingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb


# # Load dataset

# In[2]:


X = pd.read_feather('dataset/train.feather')
y = X.iloc[:, -1]
X = X.iloc[:, :-1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.25, stratify=y)


# # Train and test Machine Learning models

# In[3]:


clf1 = ExtraTreesClassifier(n_jobs=-1, n_estimators=1000, criterion='entropy')
clf2 = RandomForestClassifier(
    n_jobs=-1, n_estimators=1000, criterion='entropy')

param = {'max_depth': 0}
clf3 = xgb.XGBClassifier(param, 1000, objective='multi:softmax')


# In[4]:

print("training ExtraTreesClassifier")
clf1.fit(X_train, y_train)
print("training RandomForestClassifier")
clf2.fit(X_train, y_train)
print("training XGBoost")
clf3.fit(X_train, y_train)


# In[ ]:


print(clf1.score(X_test, y_test), clf2.score(
    X_test, y_test), clf3.score(X_test, y_test))


# In[ ]:


# Save trained model


with open('models/ExtraTrees.pickle', 'wb') as handle:
    pickle.dump(clf1, handle)

with open('models/RandomForest.pickle', 'wb') as handle:
    pickle.dump(clf2, handle)

with open('models/XGBoost.pickle', 'wb') as handle:
    pickle.dump(clf3, handle)


# In[ ]:


# test saved model

with open('models/ExtraTrees.pickle', 'rb') as handle:
    clf_et = pickle.load(handle)

with open('models/RandomForest.pickle', 'rb') as handle:
    clf_rf = pickle.load(handle)

with open('models/XGBoost.pickle', 'rb') as handle:
    clf_xgb = pickle.load(handle)

print(clf_et.score(X_test, y_test), clf_rf.score(
    X_test, y_test), clf_xgb.score(X_test, y_test))
