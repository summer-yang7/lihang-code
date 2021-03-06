"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# %matplotlib inline

# data
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = [
        'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1
    # print(data)
    return data[:, :2], data[:, -1]


X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

plt.scatter(X[:50,0],X[:50,1], label='0')
plt.scatter(X[50:,0],X[50:,1], label='1')
plt.legend()
# plt.show()


from sklearn.svm import SVC
clf = SVC()
clf.fit(X_train, y_train)

clf.score(X_test, y_test)

from sklearn.svm import SVC

# 加载数据
X = [[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]]
y = [1, 1, 1, -1, -1]

# 训练SVM模型
clf = SVC(kernel='linear', C=10000)
clf.fit(X, y)

print("w = ", clf.coef_)
print("b = ", clf.intercept_)
print("support vectors =", clf.support_vectors_)

import matplotlib.pyplot as plt
import  numpy as np 

# 绘制数据点
color_seq = ['red' if v == 1 else 'blue' for v in y]
plt.scatter([i[0] for i in X], [i[1] for i in X], c=color_seq)
# 得到x轴的所有点
xaxis = np.linspace(0, 3.5)
w = clf.coef_[0]
# 计算斜率
a = -w[0] / w[1]
# 得到分离超平面
y_sep = a * xaxis - (clf.intercept_[0]) / w[1]
# 下边界超平面
b = clf.support_vectors_[0]
yy_down = a * xaxis + (b[1] - a * b[0])
# 上边界超平面
b = clf.support_vectors_[-1]
yy_up = a * xaxis + (b[1] - a * b[0])
# 绘制超平面
plt.plot(xaxis, y_sep, 'k-')
plt.plot(xaxis, yy_down, 'k--')
plt.plot(xaxis, yy_up, 'k--')
# 绘制支持向量
plt.xlabel("$x^{(1)}$")
plt.xlabel("$x^{(2)}$")
plt.scatter(clf.support_vectors_[:,0],
            clf.support_vectors_[:,1],
            s=150,
            facecolors="none",
            edgecolors='k')   

plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-4, 5, 1)
y = np.array((x >= -2) & (x <= 2), dtype='int')

plt.scatter(x[y==0], [0]*len(x[y==0]))
plt.scatter(x[y==1], [0]*len(x[y==1]))
plt.show()

def gaussian(x, l):
    # 此处直接将超参数 γ 设定为 1.0；
    # 此处 x 表示一维的样本，也就是一个具体的值，l 相应的也是一个具体的数，因为 l 和 x 一样，从特征空间中选定；
    gamma = 1.0
    # 此处因为 x 和 l 都只是一个数，不需要再计算模，可以直接平方；
    return np.exp(-gamma * (x-l)**2)

# 设定地标 l1、l2 为 -1和1
l1, l2 = -1, 1
x_new = np.empty((len(x), 2))


for i, data in enumerate(x):
    x_new[i, 0] = gaussian(data, l1)
    x_new[i, 1] = gaussian(data, l2)

plt.scatter(x_new[y==0, 0], x_new[y==0, 1])
plt.scatter(x_new[y==1, 0], x_new[y==1, 1])
plt.show()

print("ycx")