import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

# 使用sklearn
path = 'D:\\Practice Program\\python\\Coursera\\Coursera\\吴恩达ml\\machine-learning-ex1\\ex1\\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

data.insert(0,'Ones',1) # 在0的位置插入一列

# data.plot(kind='scatter',x='Population',y='Profit',figsize=(8,5))
# plt.show()

# 对于数据，profit为y，其余为X
cols = data.shape[1]
X = data.iloc[:, 0:2]# X = data.iloc[:,0:cols-1] # 使用cols进行计算，适用于多列情况
y = data.iloc[:, 2:3] #取最后一列# y = data.iloc[:,cols-1:cols]

model = linear_model.LinearRegression()
model.fit(X, y)

x = np.linspace(data.Population.min(), data.Population.max(), 100)  # 横坐标
f = model.predict(X).flatten  # 纵坐标，利润

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data['Population'], data.Profit, label='Traning Data')
ax.legend(loc=2)  # 2表示在左上角
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()
