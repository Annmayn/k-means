# -*- coding: utf-8 -*-
import numpy as np
def updateCluster(arr, centroids):
    clusterUpdatedList=[]
    for point in arr:
        dist=np.sqrt(np.sum((centroids-point)**2, axis=1))
        clusterUpdatedList.append(np.argmin(dist))
    return np.array(clusterUpdatedList)

def updateCentroids(arr, clusterNumber, centroids):
    for i in range(len(centroids)):
        cluster = arr[clusterNumber==i]
        mean_val = np.mean(cluster, axis=0)
        centroids[i] = mean_val
    return centroids