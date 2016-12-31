import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)

#import seaborn as sns
#sns.set_context('notebook')
#sns.set_style('white')

data = np.loadtxt('linear_regression_data1.txt', delimiter=',')
X = np.c_[np.ones(data.shape[0]),data[:,0]]  # it seem declear type here
y = np.c_[data[:,1]]        # np.r_按row来组合array，np.c_按colunm来组合array
#print(X)
#print(y)

plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1) # s，点的大小
plt.xlim(4,24) # 轴的极限范围,图显示的更内容更有效
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s');

# 计算损失函数
def computeCost(X, y, theta=[[0],[0]]):
    m = y.size
    J = 0
    h = X.dot(theta)   #矩阵乘法
    J = 1.0/(2*m)*(np.sum(np.square(h-y)))
    return J

# 梯度下降
def gradientDescent(X, y, theta=[[0],[0]], alpha=0.01, num_iters=1500):
    m = y.size
    J_history = np.zeros(num_iters)

    for iter in np.arange(num_iters):
        h = X.dot(theta)
        theta = theta - alpha*(1.0/m)*(X.T.dot(h-y))
        J_history[iter] = computeCost(X, y, theta)
    return(theta, J_history)

theta , Cost_J = gradientDescent(X, y)
# 画出每一次迭代和损失函数变化
#print('theta: ',theta.ravel())
#print(Cost_J)
"""
plt.plot(Cost_J)       # plot y using x as index array 0..N-1
plt.ylabel('Cost J')
plt.xlabel('Iterations')
"""

xx = np.arange(5,23)
yy = theta[0]+theta[1]*xx

# 画出我们自己写的线性回归梯度下降收敛的情况
plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
plt.plot(xx,yy, label='Linear regression (Gradient descent)')

# 和Scikit-learn中的线性回归对比一下
regr = LinearRegression()
regr.fit(X[:,1].reshape(-1,1), y.ravel())  #数据的标准化处理
plt.plot(xx, regr.intercept_+regr.coef_*xx, label='Linear regression (Scikit-learn GLM)')

plt.xlim(4,24)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.legend(loc=4);

plt.show()