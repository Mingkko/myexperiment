#手写一个简单kmeans，与sklearn作比较
# from matplotlib import pyplot  as plt
from functools import update_wrapper
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

#处理一下数据
digits = load_digits()
data = scale(digits.data)
pca = PCA(n_components=2).fit(data)
points = PCA(n_components=2).fit_transform(data)
# print(points)
#变成二维数据点


class my_kmeans:
    def __init__(self,k):
        self.k = k
        self.cluster_centers=None
    
    #随机初始化簇中心
    def init_center(self):
        return np.random.random(self.k*2).reshape((self.k,2))
    
    #定义欧式距离
    def distance(self,a,b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]+b[1])**2)

    
    #划分样本到簇
    def partition(self,points,centroids):
        new_index = np.zeros(len(points))
        for j,point in enumerate(points):
            index = 0
            min_dis = np.inf
            for i in range(len(centroids)):
                d = self.distance(point,centroids[i])
                if d<min_dis:
                    min_dis=d
                    index=i
            new_index[j] = index
        new_index = new_index.astype(np.int)
        return new_index
                

    #计算簇中心
    def update_center(self,points,centroids_index):
        new_centroids = np.zeros((self.k,2))
        for i in range(self.k):
            new_centroids[i] = points[centroids_index == i].mean(axis=0)
        return new_centroids
    

    def fit(self,points):
        self.cluster_centers = self.init_center()
        indeces = self.partition(points,self.cluster_centers)

        for i in range(100):
            self.cluster_centers = self.update_center(points,indeces)
            indeces = self.partition(points,self.cluster_centers)

        return self


#my_kmearns
m_kmeans = my_kmeans(10)
m_kmeans.fit(points)
print(m_kmeans.cluster_centers)

print("=======")
#sklearn
kmeans = KMeans(n_clusters=10)
kmeans.fit(points)
print(kmeans.cluster_centers_)