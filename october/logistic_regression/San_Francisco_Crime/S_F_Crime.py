import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import numpy as np

# 先了解自己的数据
train = pd.read_csv('sf_data/train.csv', parse_dates=['Dates'])
test = pd.read_csv('sf_data/test.csv', parse_dates=['Dates'])
#print(train.head())
#print(test.head())

# 拿出所有的地址信息来构成array
all_addr = np.array(train.Address.tolist() + test.Address.tolist())
#print(list(all_addr))

# Convert a collection of text documents to a matrix of token counts
stop_words = ['dr', 'wy', 'bl', 'av', 'st', 'ct', 'ln', 'block', 'of']
vectorizer = CountVectorizer(max_features=300, stop_words=stop_words)
features = vectorizer.fit_transform(all_addr).toarray()
features[0,:]

X = features[:train.shape[0]]
y = train.Category

# 分成80%的训练集和20%的验证集
# 它的用途是在随机划分训练集和测试集时候，划分的结果并不是那么随机
# 也即，确定下来random_state是某个值后，重复调用这个函数，划分结果是确定的。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

log_model = LogisticRegression().fit(X=X_train, y=y_train)
results = log_model.predict_proba(X_test)

np.round(results[1], 3)   # 将数组中的元素按指定的精度进行四舍五入

# http://scikit-learn.org/stable/modules/model_evaluation.html
log_loss_score = log_loss(y_test, results)
print('log loss score: {0}'.format(round(log_loss_score, 3)))

# 开始做预测
log_model = LogisticRegression().fit(X=features[:train.shape[0]], y=train.Category)
results = log_model.predict_proba(features[train.shape[0]:])
results

"""
# 整理提交的结果
submission = pd.DataFrame(results)
submission.columns = sorted(train.Category.unique())
submission.set_index(test.Id)
submission.index.name="Id"
submission.to_csv('py_submission_logreg_addr_300.csv')
"""