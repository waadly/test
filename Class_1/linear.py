import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 单变量线性回归
path = 'D:\\Practice Program\\python\\Coursera\\Coursera\\吴恩达ml\\machine-learning-ex1\\ex1\\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

data.insert(0,'Ones',1) # 在第0行，第一列的位置插入一列

data.plot(kind='scatter',x='Population',y='Profit',figsize=(8,5))
plt.show()

def costFunction(X, y, th):
    inner = np.power((np.dot(X, th.T) - y), 2)
    J = np.sum(inner) / (2*len(X))
    return J


# 对于数据，profit为y，其余为X
cols = data.shape[1]
X = data.iloc[:, 0:2]# X = data.iloc[:,0:cols-1] # 使用cols进行计算，适用于多列情况
y = data.iloc[:, 2:3] #取最后一列# y = data.iloc[:,cols-1:cols]

th = np.zeros((1,2))
# ii = (np.dot(X, th.T) - y) * X
# print(ii)
J = costFunction(X, y, th)
print(X.shape,th.shape,y.shape)

# 梯度下降
def gradientDescent(X, y, th, al, iters):
    """reuturn theta, cost"""
    cost = np.zeros(iters) # 迭代次数  
    m = X.shape[0]  # 样本数量m

    for i in range(iters):
        # 利用向量化一步求解
        temp1 = np.dot(X, th.T) - y
        temp2 = np.dot(temp1.T, X)
        th = th - (al / m) * temp2

        cost[i] = costFunction(X, y, th)

    return th, cost

al = 0.01
iters = 1000

theta1,cost = gradientDescent(X, y, th, al, iters)
print(theta1)


x = np.linspace(data.Population.min(), data.Population.max(), 100)  # 横坐标
f = theta1[0, 0] + (theta1[0, 1] * x)  # 纵坐标，利润

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data['Population'], data.Profit, label='Traning Data')
ax.legend(loc=2)  # 2表示在左上角
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(np.arange(iters), cost, 'r')  
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training iters')
plt.show()


# 多变量线性回归

path2 = 'D:\\Practice Program\\python\\Coursera\\Coursera\\吴恩达ml\\machine-learning-ex1\\ex1\\ex1data2.txt'
data2 = pd.read_csv(path2, header=None, names=['Size', 'Badrooms','Price'])


# 特征归一化
data2 = (data2 - data2.mean()) / data2.std()
print(data2.shape)

data2.insert(0,'Ones',1)

cols = data2.shape[1]
X = data2.iloc[:,0:cols-1]
y = data2.iloc[:,cols-1:cols]

theta2 = np.zeros((1,3))

g2,cost2 = gradientDescent(X,y,theta2,al,iters)

print(g2)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training iters')
plt.show()

