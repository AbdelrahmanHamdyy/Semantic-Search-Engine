import pandas as pd
from sklearn.cluster import MiniBatchKMeans


def run_kmeans(data, k):
    # Create and fit the MiniBatchKMeans model
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)

    # Get the centroids
    centroids = kmeans.cluster_centers_

    return centroids
