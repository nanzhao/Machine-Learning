import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import minimize

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)

#import seaborn as sns
#sns.set_context('notebook')
#sns.set_style('white')

def loaddata(file, delimeter): # numpy load 数据
    data = np.loadtxt(file, delimiter=delimeter)
    print('Dimensions: ',data.shape)
    print(data[1:6,:])
    return(data)

def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
    # 获得正负样本的下标(即哪些是正样本，哪些是负样本)
    neg = data[:,2] == 0       #选择正样本的条件
    pos = data[:,2] == 1       #选择负样本的条件

    if axes == None:
        axes = plt.gca()

    #分轴来绘制散点图
    axes.scatter(data[pos][:,0], data[pos][:,1], marker='+', c='k', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[neg][:,0], data[neg][:,1], c='y', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon= True, fancybox = True) # matploblib 绘制散点图

data = loaddata('data1.txt', ',')

X = np.c_[np.ones((data.shape[0],1)), data[:,0:2]] # ones 创建数组
y = np.c_[data[:,2]]

plotData(data, 'Exam 1 score', 'Exam 2 score', 'Pass', 'Fail')

#定义sigmoid函数
def sigmoid(z):
    return(1 / (1 + np.exp(-z)))

#定义损失函数
def costFunction(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))

    J = -1.0*(1.0/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y))

    if np.isnan(J[0]):
        return(np.inf)
    return J[0]

#求解梯度
def gradient(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1,1)))

    grad =(1.0/m)*X.T.dot(h-y)

    return(grad.flatten())

initial_theta = np.zeros(X.shape[1])  #zeros 创建0矩阵
cost = costFunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)
#print('Cost: \n', cost)
#print('Grad: \n', grad)

res = minimize(costFunction, initial_theta, args=(X,y), jac=gradient, options={'maxiter':400})
#print(res)

def predict(theta, X, threshold=0.5):
    p = sigmoid(X.dot(theta.T)) >= threshold
    return(p.astype('int'))

sigmoid(np.array([1, 45, 85]).dot(res.x.T))

#绘制决策边界曲线
plt.scatter(45, 85, s=60, c='r', marker='v', label='(45, 85)') #绘制一个散点
plotData(data, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not admitted')
x1_min, x1_max = X[:,1].min(), X[:,1].max(),
x2_min, x2_max = X[:,2].min(), X[:,2].max(),

#函数接收两个一维数组，并产生两个二维矩阵
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0],1)), xx1.ravel(), xx2.ravel()].dot(res.x))
h = h.reshape(xx1.shape)
plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b') # h是由xx1,xx2构成的函数，轮廓图选h=0.5的直线

plt.show()


