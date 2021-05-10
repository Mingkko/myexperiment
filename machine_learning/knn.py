#写一个距离度量为欧氏距离，权重为uniform的knn，与sklearn的作比较，数据使用鸢尾花
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
x = iris.data
y = iris.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)



#定义欧氏距离
def cal_dis(x,y):
    distance = 0
    for i in range(len(x)):
        distance += (np.abs(x[i]-y[i]))**2
    
    return np.sqrt(distance)


class my_node:
    def __init__(self,v,d,tag):
        self.v = v
        self.d = d
        self.tag = tag

class my_neighbor:
    def __init__(self,k_neighbor=5):
        self.k_neighbor=k_neighbor
        self.node_list = []

    def fit(self,x,y):
        for i in range(len(x)):
            node = my_node(x[i],0,y[i])
            self.node_list.append(node)
        return self
    
    def predict(self,x):
        result = np.zeros(len(x),dtype=np.int)
        for j in range(len(x)):
            for i in range(len(self.node_list)):
                node = self.node_list[i]
                node.d = cal_dis(node.v,x[j])
            target_node_list = sorted(self.node_list, key=lambda b:b.d)[:self.k_neighbor]
            my_dict = {}
            for node in target_node_list:
                if node.tag not in my_dict:
                    my_dict[node.tag] = 1
                else:
                    my_dict[node.tag] += 1
            target = sorted(my_dict.items(),key=lambda b:b[1],reverse=True)
            result[j] = target[0][0]
        return result

#my_kneighbor
my_clf = my_neighbor(5)
my_clf.fit(x_train,y_train)
my_result = my_clf.predict(x_test)
my_score = accuracy_score(y_test,my_result)
print("my_kneighbor_score:{:.4f}".format(my_score))

#sklearn
clf = KNeighborsClassifier(5,p=2)
clf.fit(x_train,y_train)
result = clf.predict(x_test)
score = accuracy_score(y_test,result)
print("sklearn_score:{:.4f}".format(score))

#实验结果表明效果是一样的，不过目前数据量比较小，所以也并没有实现kd tree 和 ball tree
        
