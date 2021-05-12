#写一个二项逻辑回归，与sklearn作比较
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import math
from sklearn.linear_model import LogisticRegression

breast_cancer = load_breast_cancer()
x = breast_cancer.data
y = breast_cancer.target


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=2021)


class logistic_regression:
    def __init__(self,Lambda=1,penalty=None):
        self.W = None
        self.max_iter = 100 #最大迭代次数
        self.et = 0.00001 #早停
        self.Lambda = Lambda
        self.penalty = penalty


    def sigmoid(self,z):
        #直接计算有溢出，用tanh代替一下试试
        # t = (1/(1+np.exp(-z)))
        t = 0.5 * (np.tanh(0.5*z)+1)
        return t

    #loss function
    def loss(self,y_true,y_pred):
        #cross entropy
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.sum(-y_true*np.log(y_pred+1e-10)-(1-y_true)*np.log(1-y_pred+1e-10)) / float(len(y_true))

    def optimize(self,W,x,y_pred,y_true):
        eta = 0.1
        #推导公式的时候遇到一个问题，交叉熵log的底是多少？
        #虽然感觉很可能是以2为底，但是毕竟计算时我用了np.log，默认底为e，机器学习里面应该都是默认为e
        #偏导W = x(y` - y)
        #偏导b = (y` - y)
        dW =  x.T*(y_pred - y_true) / float(len(x))
        #加入l1和l2正则项，和之前的线性回归正则化一样
        if self.penalty == 'l2':
            dW = dW + self.Lambda * self.W
        elif self.penalty == 'l1':
            dW = dW + self.Lambda * np.sign(self.W)

        W = W - eta*dW

        return W


    def fit(self,x,y):
        #z = Wx+b
        #y = sigmoid(z)
        b = np.ones((len(x),1))
        x = np.column_stack((b,x))
        self.W = np.mat(np.ones((x.shape[1],1)))
        x = np.mat(x)
        y = np.mat(y.reshape(-1,1))

        for i in range(self.max_iter):
            z = x*self.W
            y_pred = self.sigmoid(z)
            self.W = self.optimize(self.W,x,y_pred,y)

        return self

    def predict(self,x):
        b = np.ones((len(x),1))
        x = np.column_stack((b,x))
        x = np.mat(x)
        y_pred = self.sigmoid(x*self.W)
        y_pred = (y_pred>0.5) * 1
        y_pred = np.array(y_pred).reshape(-1,)
        return y_pred 


# std=x.std(axis=0)
# mean=x.mean(axis=0)
# X_norm = (x - mean) / std
# x_train,x_test,y_train,y_test = train_test_split(X_norm,y,test_size=0.3,random_state=2021)

#my_clf
my_clf = logistic_regression(penalty='l2')
my_clf.fit(x_train,y_train)
my_result = my_clf.predict(x_test)
my_score = accuracy_score(y_test,my_result)
print("my_clf score:{:.4f}".format(my_score))   



#sklearn
clf = LogisticRegression()
clf.fit(x_train,y_train)
result = clf.predict(x_test)
score = accuracy_score(y_test,result)
print("sklearn_clf score:{:.4f}".format(score))

#用tanh可以解决sigmoid溢出问题
#手写效果比不上sklearn，速度还比较慢，增大迭代次数能逐渐提升效果
#考虑到正则化，加入l2正则化，再次尝试
#有l2正则化之后可以以比较小的迭代次数收敛,但最终结果还是低于sklearn 0.9006 对比0.9415