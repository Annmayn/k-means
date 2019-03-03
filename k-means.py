# -*- coding: utf-8 -*-
import numpy as np
import clusterProcessing
import pandas as pd
import matplotlib.pyplot as plt

arr=np.array([[1,1],[1,2],[1,3],[4,1],[5,1],[0,0],[6,1],[5,5],[5,6],[6,5]])
k = 3


def runKMeans(arr, k, num_iter=1000, eps=10e-7):
    e=10
    centroids = np.random.choice(len(arr), size=k, replace=False)
    centroids = arr[centroids]
    while (num_iter>0 and e>eps):
        clusterNumber = clusterProcessing.updateCluster(arr, centroids)
        centroids_old = centroids
        centroids = clusterProcessing.updateCentroids(arr, clusterNumber, centroids)
        e = abs(np.mean(centroids) - np.mean(centroids_old))
    return clusterNumber
        
clusterNumber = runKMeans(arr, k, num_iter=10000)
df = pd.DataFrame()
df['x']=arr[:,0]
df['y']=arr[:,1]
df['cluster']=clusterNumber
plt.scatter(df['x'], df['y'], c=df['cluster'])