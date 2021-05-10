#写一个朴素贝叶斯分类器，与sklearn作比较，数据用鸢尾花数据。
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import math
from sklearn.naive_bayes import GaussianNB

iris = load_iris()
x = iris.data
y = iris.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


class my_node:
    def __init__(self,v,tag):
        self.v = v
        self.tag = tag
        self.t = []

#鸢尾花数据的特征列是连续值，假设符合正态分布，所以写一个高斯分布的朴素贝叶斯分类器

#计算条件概率时用属于yk类别的正态分布概率密度函数值
class naive_bayes:
    def __init__(self):
        self.y_dict = {}
        self.node_list = []
    
    @staticmethod
    def mean(x):
        return sum(x) / float(len(x))

    #计算标准差
    def cal_std(self,x):
        #就不用numpy来算了
        avg = self.mean(x)
        std = math.sqrt(sum(math.pow((i-avg),2) for i in x)/len(x))
        return std+1e-10 #var smoothing
    
    #计算概率密度函数
    def gaussian_probability_density(self,x,avg,std):
        exp = math.exp(-(math.pow((x-avg),2)) / (2*math.pow(std,2)))
        return (1 / (math.sqrt(2*math.pi)*std)) * exp
    
    def fit(self,x,y):
        for i in range(len(x)):
            node = my_node(x[i],y[i])
            self.node_list.append(node)
        
        #计算先验概率
        y_dict = {}
        for node in self.node_list:
            if node.tag not in y_dict:
                y_dict[node.tag] = 1
            else:
                y_dict[node.tag] +=1
        for k,v in y_dict.items():
            y_dict[k] = v / float(len(y))
        
        self.y_dict = y_dict


        #计算均值和标准差
        temp_dict = {}
        for tag in y_dict.keys():
            tag_node_list_v = [node.v for node in self.node_list if node.tag == tag]
            temp = [(self.mean(i),self.cal_std(i)) for i in zip(*tag_node_list_v)]
            temp_dict[tag] = temp

        for node in self.node_list:
            node.t = temp_dict[node.tag]

        
        return self
    
    def predict(self,x):
        result = np.zeros(len(x),dtype=np.int)
        for j in range(len(x)):
            prob_list = []
            for tag, prob in self.y_dict.items():
                tag_node_list_avg_std = [node.t for node in self.node_list if node.tag == tag][0]
                t_prob = np.log(prob)
                for i, t in zip(x[j],tag_node_list_avg_std):
                    condition_prob = self.gaussian_probability_density(i, t[0], t[1])                   
                    t_prob += np.log(condition_prob)
                prob_list.append((tag,t_prob))
            temp = sorted(prob_list,key=lambda b:b[1],reverse=True)
            result[j] = temp[0][0]
        return result


#my_naive_bayes
my_clf = naive_bayes()
my_clf.fit(x_train,y_train)
my_result = my_clf.predict(x_test)
my_score = accuracy_score(y_test,my_result)
print("my_clf socre:{:.4f}".format(my_score))


#sklearn
clf = GaussianNB()
clf.fit(x_train,y_train)
result = clf.predict(x_test)
score = accuracy_score(y_test,result)
print("sklearn_clf score:{:.4f}".format(score))

#实验结果表明，两者正确率一致
