import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.cluster.vq import kmeans2

def run_kmeans_minibatch(data, k):
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(data)

    centroids = kmeans.cluster_centers_

    return centroids

def run_kmeans2(data, k):
    centroids = kmeans2(data, k)
    return centroids

def run_kmeans(data, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(data)

    centroids = kmeans.cluster_centers_

    return centroids
