# -*- coding: utf-8 -*-
import numpy as np

#returns the new cluster organization in the order of the points in original input array
def updateCluster(arr, centroids):
    #keep a list that stores the cluster organization
    clusterUpdatedList=[]
    #for every point, find the nearest centroid
    # and assign that clusters "clusterID" to that point
    for point in arr:
        dist=np.sqrt(np.sum((centroids-point)**2, axis=1))
        clusterUpdatedList.append(np.argmin(dist))
    return np.array(clusterUpdatedList)


#updates the centroids based on the cluster
def updateCentroids(arr, clusterNumber, centroids):
    #do this for every centroid
    for i in range(len(centroids)):
        #get all points belonging to the cluster 'i'
        cluster = arr[clusterNumber==i]
        #mean_val of the form (x,y)
        mean_val = np.mean(cluster, axis=0)
        #update centroid
        centroids[i] = mean_val
    return centroids