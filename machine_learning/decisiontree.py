#手写一个决策树做分类任务，与sklearn做比较
import numpy as np
from numpy.lib.function_base import sort_complex
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine

wine = load_wine()
x = wine.data
y = wine.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2021)



# a1 = 0
# a2 = 0.0143

# b1 = [0,0.1071,0.4067,0.4303,0.0559]
# b2 = [0.0317,0.0969,0.3824,0.3707,0.1041]

# y2 = 0
# y1 = []
# for i in range(5):
#     t = b1[i]*b2[i]+b1[i]*a2+b2[i]*a1
#     y2 += t
#     y1.append(t)

# z = []
# for i in range(5):
#     t = y1[i]/y2
#     z.append(t)

# print(z)


# exit(0)




#数据集的特征列都是连续值，相比于离散值处理起来要复杂一点，使用二分法来寻找最优划分点，最终分类有3类

class my_node:
    def __init__(self):
        self.v = None
        self.depth = None
        self.split_feature_index=None
        self.split_feature_point=None
        self.left=None
        self.right=None
        self.tag = None




#简单的决策树
class my_dt:
    def __init__(self,max_depth=20,criterion='gini'):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree=None

    def cal_ent(self,y):
        labels = list(set(y))
        ent_y = np.zeros(len(labels))
        for i in range(len(labels)):
            ent_y[i] = list(y).count(labels[i])
            ent_y[i] = ent_y[i] / float(len(y))
        ent_d = -sum(p*np.log2(p) for p in ent_y)

        return end_d
    
    def cal_gini(self,y):
        labels = list(set(y))
        gini_y = np.zeros(len(labels))
        for i in range(len(labels)):
            gini_y[i] = list(y).count(labels[i])
            gini_y[i] = gini_y[i] / float(len(y))
        gini_y = 1-sum(p**2 for p in gini_y)

        return gini_y

    def cal_gain(self,feature,y):
        feature.sort()
        mid = len(feature) // 2
        #二分
        i = 0
        result = []
        while i <= mid and (i+1)<len(feature):
            split_point = (feature[i] + feature[i+1]) / 2
            d_ = (i+1) / len(feature)
            dplus_ = (len(feature) - (i+1)) / len(feature)

            y_ = [py for d,py in zip(feature,y) if d < split_point]
            yplus_ = [py for d,py in zip(feature,y) if d > split_point]

            if self.criterion == 'entropy':
                end_d = self.cal_ent(y)

                gain_a_t = end_d - (d_* self.cal_ent(y_) + dplus_*self.cal_ent(yplus_))
                result.append(gain_a_t)

            if self.criterion == 'gini':
                gini_d = d_* self.cal_gini(y_) + dplus_* self.cal_gini(yplus_)
                result.append(gini_d)
            
            i+=1

        if self.criterion == 'entropy':
            r_index = result.index(max(result))
            r_split_point = (feature[r_index] + feature[r_index+1] )/2

            return r_split_point, max(result)

        if self.criterion == 'gini':
            r_index = result.index(min(result))
            r_split_point = (feature[r_index] + feature[r_index+1] )/2

            return r_split_point, min(result)
        

    def find_split_feature(self,x,y):
        select_index = []
        select_gain = []

        for i in range(len(x[0])):
            split_point,feature_gain = self.cal_gain(x[:,i],y)
            select_gain.append(feature_gain)
            select_index.append((i,split_point))
        r_gain = max(select_gain)
        r_index = select_gain.index(r_gain)

        return select_index[r_index]

        
    def partition(self,x,y):
        
        feature_index_point = self.find_split_feature(x,y)
        split_index = feature_index_point[0]
        split_point = feature_index_point[1]

        if len(x[0]) !=1:
            data_ = [(np.delete(x_,split_index),y_) for x_,y_ in zip(x,y) if x_[split_index]<= split_point]
            dataplus_ = [(np.delete(x_,split_index),y_) for x_,y_ in zip(x,y) if x_[split_index]> split_point]
        else:
            data_ = [(x_,y_) for x_,y_ in zip(x,y) if x_[split_index]<= split_point]
            dataplus_ = [(x_,y_) for x_,y_ in zip(x,y) if x_[split_index]> split_point]

        return data_, dataplus_, split_index, split_point

    def build_tree(self,x,y,depth,max_depth):
        node = my_node()
        node.v = [(x_,y_) for x_,y_ in zip(x,y)]
        node.depth =depth

        labels = list(set(y))
        if len(labels) ==0:
            return None
        y_ = np.zeros(len(labels))
        for i in range(len(labels)):
            y_[i] = list(y).count(labels[i])
        node.tag = list(y_).index(max(y_))
        if len(labels) == 1 or len(x[0])==1:
            pass
        else:
            data_, dataplus_, split_index, split_point = self.partition(x,y)
            node.split_feature_index = split_index
            node.split_feature_point = split_point

        if depth == max_depth or len(labels)==1 or len(x[0])==1:
            return node
        else:
            data_x = np.array([t[0] for t in data_])
            data_y = np.array([t[1] for t in data_])
            node.left = self.build_tree(data_x,data_y,depth+1,max_depth)

            dataplus_x = np.array([t[0] for t in dataplus_])
            dataplus_y = np.array([t[1] for t in dataplus_])
            node.right = self.build_tree(dataplus_x,dataplus_y,depth+1,max_depth)
        
            return node



    def fit(self,x,y):
        tree = self.build_tree(x,y,0,self.max_depth)

        self.tree = tree
        return self
    


    def predict(self,x):
        result = []
        for x_ in x:
            node = self.tree

            while node:
                if node.left and node.right:
                    if x_[node.split_feature_index] <= node.split_feature_point:
                        node = node.left
                    else:
                        node = node.right
                else:
                    result.append(node.tag)
                    break

        return np.array(result)


#my_decision
my_de = my_dt()
my_de.fit(x_train,y_train)
my_result = my_de.predict(x_test)
my_score = accuracy_score(y_test,my_result)
print("my_decisiontree score:{:.4f}".format(my_score))

#sklearn
det = DecisionTreeClassifier(max_depth=20)
det.fit(x_train,y_train)
result = det.predict(x_test)
score = accuracy_score(y_test,result)
print("sklearn score:{:.4f}".format(score))


#实验结果为 my：sklearn = 0.1667：0.2778 都非常低，这是为什么。