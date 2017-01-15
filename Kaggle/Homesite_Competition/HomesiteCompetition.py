#This is an example for kaggle competition
#https://www.kaggle.com/c/homesite-quote-conversion

#Parameter grid search with xgboost
#feature engineering is not so useful and the LB is so overfitted/underfitted
#so it is good to trust your CV

#go xgboost, go mxnet, go DMLC! http://dmlc.ml

#Credit to Shize's R code and the python re-implementation

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

from sklearn.cross_validation import *
from sklearn.grid_search import GridSearchCV

train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

#0 for rows and 1 for columns
train = train.drop('QuoteNumber', axis=1)
test = test.drop('QuoteNumber', axis=1)

# Lets play with some dates
train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
train = train.drop('Original_Quote_Date', axis=1)

test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
test = test.drop('Original_Quote_Date', axis=1)

train['Year'] = train['Date'].apply(lambda x: int(str(x)[:4]))
train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))
train['weekday'] = train['Date'].dt.dayofweek

test['Year'] = test['Date'].apply(lambda x: int(str(x)[:4]))
test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))
test['weekday'] = test['Date'].dt.dayofweek

train = train.drop('Date', axis=1)
test = test.drop('Date', axis=1)
print(train.head())

#缺失特征值处理的例子
#查看哪些数据是NaN,然后对其进行简单填充 fill -999 to NAs
#更为具体的例子可以见，kaggle中的这个notebook
#https://www.kaggle.com/apryor6/santander-product-recommendation/detailed-cleaning-visualization-python

#if train.isnull().any().any() == True:
#    print(train.ix[:,train.isnull().any()])
train = train.fillna(-999)
test = test.fillna(-999)

features = list(train.columns[1:])  #la colonne 0 est le quote_conversionflag
print(features)

#现实数据中很多特征并不是数值类型，而是类别类型，
#虽然决策树天然擅长处理类别类型的特征，但是还是需要我们把原始的字符串值转为类别编号。
for f in train.columns:
    if train[f].dtype=='object':
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))
print(train.head())

xgb_model = xgb.XGBClassifier()

#brute force scan for all parameters, here are the tricks
#usually max_depth is 6,7,8
#learning rate is around 0.05, but small changes may make big diff
#tuning min_child_weight subsample colsample_bytree can have
#much fun of fighting against overfit
#n_estimators is how many round of boosting
#finally, ensemble xgboost with multiple seeds may reduce variance
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.05], #so called `eta` value
              'max_depth': [6],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [5], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [1337]}


clf = GridSearchCV(xgb_model, parameters, n_jobs=5,
                   cv=StratifiedKFold(train['QuoteConversion_Flag'], n_folds=5, shuffle=True),
                   scoring='roc_auc',
                   verbose=2, refit=True)

clf.fit(train[features], train["QuoteConversion_Flag"])

#trust your CV!
best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
print('Raw AUC score:', score)
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

test_probs = clf.predict_proba(test[features])[:,1]

sample = pd.read_csv('./input/sample_submission.csv')
sample.QuoteConversion_Flag = test_probs
sample.to_csv("xgboost_best_parameter_submission.csv", index=False)

