import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.cluster.vq import kmeans2


def run_kmeans_minibatch(data, k):
    # Create and fit the MiniBatchKMeans model
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)

    # Get the centroids
    centroids = kmeans.cluster_centers_

    return centroids


def run_kmeans2(data, k):
    centroids, _ = kmeans2(data, k)
    return centroids


def run_kmeans(data, k):
    # Create and fit the model
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)

    # Get the centroids
    centroids = kmeans.cluster_centers_

    return centroids
