# -*- coding: utf-8 -*-
import numpy as np
import clusterProcessing
import pandas as pd
import matplotlib.pyplot as plt

#fixed input for now
arr=np.array([[1,1],[1,2],[1,3],[4,1],[5,1],[0,0],[6,1],[5,5],[5,6],[6,5]])
#fixed cluster number for now
k = 3

#responsible for calling all other processing functions
#default no_of_iteration = 1000
#default error_difference =10e-7
def runKMeans(arr, k, num_iter=1000, eps=10e-7):
    #get k random centroids
    centroids = np.random.choice(len(arr), size=k, replace=False)
    centroids = arr[centroids]
    #initialize error_difference to some big value
    e=10
    #run until termination condition
    while (num_iter>0 and e>eps):
        #update the cluster ID for all points
        clusterNumber = clusterProcessing.updateCluster(arr, centroids)
        centroids_old = centroids
        #update centroids based on new cluster
        centroids = clusterProcessing.updateCentroids(arr, clusterNumber, centroids)
        #calculate error, np.mean witt axis=None gives single-valued float
        e = abs(np.mean(centroids) - np.mean(centroids_old))
    return clusterNumber
        
#run the main function with provided parameters
clusterNumber = runKMeans(arr, k, num_iter=10000)

#pandas dataframe for easy plotting and viewing
df = pd.DataFrame()
df['x']=arr[:,0]
df['y']=arr[:,1]
df['cluster']=clusterNumber

#scatter plot of x vs y with different color (provided by clusterID)
plt.scatter(df['x'], df['y'], c=df['cluster'])