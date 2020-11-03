#!/usr/bin/env python
# coding: utf-8

# # librarys

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import math
import random
import warnings
warnings.simplefilter('ignore')
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from skrebate import ReliefF
from sklearn.model_selection import cross_val_score
from numpy import array
from mlxtend.feature_selection import ColumnSelector
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from IPython.display import Image  
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
import pydotplus
from sklearn.externals.six import StringIO  
from sklearn.svm import SVC
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
get_ipython().run_line_magic('matplotlib', 'inline')


# # functions

# In[2]:


def evaluate(model, x_test, y_test):
    y_pred = model.predict(x_test)
    #     print('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_pred)))
#     print('Kappa: {:.3f}'.format(cohen_kappa_score(y_test, y_pred)))
# #     print('F1 socre: {:.3f}'.format(f1_score(y_test, y_pred)))
# #     print('Recall socre: {:.3f}'.format(recall_score(y_test, y_pred)))
#     print(classification_report(y_test, y_pred))
    return(accuracy_score(y_test, y_pred))

def report(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_pred)))
    print('Kappa: {:.3f}'.format(cohen_kappa_score(y_test, y_pred)))
    print('Recall socre: {:.3f}'.format(recall_score(y_test, y_pred,average='micro')))
    print('F1 socre: {:.3f}'.format(f1_score(y_test, y_pred, average='micro')))
#     print('Balanced accuracy socre: {:.3f}'.format(balanced_accuracy_score(y_test, y_pred)))
    print(classification_report(y_test, y_pred))

def density_plot(locations, data, element):
    # Iterate through the 6 locations
    for location in locations:
        # Subset to the locations
        subset = data[data['lv'] == location]

        # Draw the density plot
#         sns.distplot(subset[element], hist = False, kde = True,

#                      label = location)
        sns.kdeplot(subset[element], bw = 0.1, label = location)
    # Plot formatting
    plt.legend(prop={'size': 16}, title = 'locations')
    plt.title('Density Plot')
    plt.xlabel(element)
    plt.ylabel('Density')
    

def reliefF(data, label):
    x, y = data.drop(label, axis = 1).values, data[label].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 233, stratify = y)
    fs = ReliefF(n_jobs=-1, n_neighbors=len(x_train))
    fs.fit(x_train,y_train)
    relief_result = pd.DataFrame(columns = ['feature', 'score'])
    relief_result['score'] = fs.feature_importances_
    relief_result['feature'] =data.columns[1:len(data.columns)]
    relief_result=relief_result.sort_values(by=['score'], ascending=False)
    return relief_result


# # data import

# In[3]:


data = pd.read_csv("C:/Users/fzkon/Documents/GitHub/Rice_authenticity_ICP_new\grand.csv")


# # Relief feature selection

# In[4]:


relief_result = reliefF(data, 'lv')


# In[5]:


relief_result.head()


# In[7]:


relief_result.to_csv("C:/Users/fzkon/Documents/GitHub/Rice_authenticity_ICP_new/updated_results/relief_result_python.csv", index=False)


# # hyperparameter optimization for rf

# In[6]:


relief_result = pd.read_csv("C:/Users/fzkon/Documents/GitHub/Rice_authenticity_ICP_new/updated_results/relief_result_python.csv")
relief_result = relief_result.sort_values(by=['score'], ascending=False)


# In[7]:


x = data.loc[:, data.columns != 'lv']
y = data.loc[:, data.columns == 'lv']


# In[8]:


data['lv'].describe()


# In[9]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 233, stratify = y)


# In[12]:


data['lv'].value_counts()


# In[14]:


len(y_test)


# In[15]:


len(y_train)


# In[16]:


y_train['lv'].value_counts()


# In[10]:


y_test['lv'].value_counts()


# In[13]:


# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [int(x) for x in np.linspace(start=1, stop = 101, num = 5)],
#     'max_features': ['auto', 'sqrt'],
#     'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 500, num = 5)]
#     }
# rf = RandomForestClassifier(n_jobs=-1)
# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
#                           cv = 10, n_jobs = -1)
# grid_search.fit(x_train, y_train)


# In[22]:


param_grid = {
    'bootstrap': [True],
    'max_depth': [int(x) for x in np.linspace(start=1, stop = 101, num = 5)],
    'max_features': ['auto', 'sqrt'],
    'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 500, num = 5)]
    }
rf = RandomForestClassifier(n_jobs=-1)
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 10, n_jobs = -1)

summary = pd.DataFrame(columns=['number of features', 'training accuracy', 'testing accuracy', 'best params'])

training_accuracy =[]
testing_accuracy = []
features = []
best_params = []

for i in range(1, x.shape[1]+1):
    print("i am in the cycle of", i)
    mask = relief_result['feature'][0:i]
    x_temp = x[mask]
    x_train, x_test, y_train, y_test = train_test_split(x_temp, y, test_size = 0.2, random_state = 233, stratify = y)
    grid_search.fit(x_train, y_train)
    best_grid = grid_search.best_estimator_
    grid_accuracy = evaluate(best_grid, x_test, y_test)
    training_accuracy.append(grid_search.best_score_)
    testing_accuracy.append(grid_accuracy)
    features.append(list(mask))
    best_params.append(grid_search.best_params_)
    if i > 1:
        if training_accuracy[i-2]>=grid_search.best_score_:
            print("i am the best!")
            break
#     if grid_search.best_score_ == 1:
#         print("i have reached 100% accuracy")
#         print(grid_search.best_params_)
#         break
    


# In[23]:


# do not delete me!!!!
summary['number of features'] = features
summary['training accuracy'] = training_accuracy
summary['testing accuracy'] = testing_accuracy
summary['best params'] = best_params
print(summary)


# In[24]:


summary.to_csv("C:/Users/fzkon/Documents/GitHub/Rice_authenticity_ICP_new/updated_results/rf_result.csv", 
               index=True)


# In[26]:


mask = relief_result['feature'][0:4]
x_temp = x[mask]
x_train, x_test, y_train, y_test = train_test_split(x_temp, y, test_size = 0.2, random_state = 233, stratify = y)


# In[27]:


rf_best = RandomForestClassifier(n_jobs=-1, max_depth=26, max_features='auto', n_estimators=500, bootstrap=True)


# In[28]:


rf_best.fit(x_train, y_train)


# In[103]:


# rf_prob = pd.DataFrame(rf_best.predict_proba(x_test))


# In[104]:


# rf_prob.to_csv("/Users/analytical/Documents/GitHub/Rice_authenticity_ICP_new/rf_prob.csv")


# In[29]:


report(rf_best, x_test, y_test)


# # hyperparameter optimization for svm

# In[30]:


from sklearn import svm


# In[31]:


param_grid = [{'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
                     'C': [1, 10, 100, 1000]},
              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
              {'kernel':['poly'], 'degree':[0, 1, 2, 3, 4, 5, 6]}
             ]
svm = svm.SVC()
grid_search = GridSearchCV(estimator = svm, param_grid = param_grid, 
                          cv = 10, n_jobs = -1)

summary = pd.DataFrame(columns=['number of features', 'training accuracy', 'testing accuracy', 'best params'])

training_accuracy =[]
testing_accuracy = []
features = []
best_params = []

for i in range(1, x.shape[1]+1):
    print("i am in the cycle of", i)
    mask = relief_result['feature'][0:i]
    x_temp = x[mask]
    x_train, x_test, y_train, y_test = train_test_split(x_temp, y, test_size = 0.2, random_state = 233, stratify = y)
    grid_search.fit(x_train, y_train)
    best_grid = grid_search.best_estimator_
    grid_accuracy = evaluate(best_grid, x_test, y_test)
    training_accuracy.append(grid_search.best_score_)
    testing_accuracy.append(grid_accuracy)
    features.append(list(mask))
    best_params.append(grid_search.best_params_)
    if i > 1:
        if training_accuracy[i-2]>=grid_search.best_score_:
            print("i am the best!")
            break
#     if grid_search.best_score_ == 1:
#         print("i have reached 100% accuracy")
#         print(grid_search.best_params_)
#         break
    


# In[32]:


summary['number of features'] = features
summary['training accuracy'] = training_accuracy
summary['testing accuracy'] = testing_accuracy
summary['best params'] = best_params
print(summary)


# In[33]:


summary.to_csv("C:/Users/fzkon/Documents/GitHub/Rice_authenticity_ICP_new/updated_results/svm_result.csv", 
               index=True)


# In[17]:


mask = relief_result['feature'][0:4]
x_temp = x[mask]
x_train, x_test, y_train, y_test = train_test_split(x_temp, y, test_size = 0.2, random_state = 233, stratify = y)


# In[35]:



svm = SVC(kernel='rbf', gamma=0.1, C=1,probability=True)

svm.fit(x_train, y_train)


# In[100]:


# svm_prob  = svm.predict_proba(x_test)


# In[36]:


report(svm, x_test, y_test)


# In[107]:


# probs = svm.predict_proba(x_test)


# ## Starting of density plot

# In[51]:


locations = ['JS', 'SY', 'WC', 'PJ-1', 'PJ-2', 'GG']


# Al

# In[79]:


density_plot(locations, data, elements[0])

data.groupby('lv')[elements[0]].describe()


# Rb

# In[80]:


density_plot(locations, data, elements[1])

data.groupby('lv')[elements[1]].describe()


# B

# In[81]:


density_plot(locations, data, elements[2])

data.groupby('lv')[elements[2]].describe()


# Na

# In[82]:


density_plot(locations, data, elements[3])

data.groupby('lv')[elements[3]].describe()


# Sr

# In[83]:


density_plot(locations, data, elements[4])

data.groupby('lv')[elements[4]].describe()


# # construction of decision tree 

# In[86]:


data['lv'].value_counts()


# In[84]:


relief_result = pd.read_csv("/Users/analytical/Desktop/GFSC/Projects/Rice Authentication Project/Rice-authentication-ICP-MS/relief_result_python.csv")

relief_result = relief_result.sort_values(by=['score'], ascending=False)

elements = list(relief_result['feature'][0:5])

x_tree = x[elements]

x_names = x_tree.columns
x_names = x_names.tolist()

y_names = ['GG', 'JS', 'PJ-1', 'PJ-2', 'SY', 'WC']

dtc = DecisionTreeClassifier(random_state=296)
sample_split_range = list(range(1, 50))
param_grid = dict(min_samples_split=sample_split_range)
grid = GridSearchCV(dtc, param_grid, cv = 20, scoring='accuracy')
grid.fit(x_tree, y)
tree = grid.best_estimator_


# Extract single tree

from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(tree, out_file='tree.dot', 
                feature_names = x_names,
                class_names = y_names,
                rounded = True, proportion = False, 
                precision = 3, filled = True, max_depth=7)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')

