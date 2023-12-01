import pandas as pd
from sklearn.cluster import KMeans


def run_kmeans(data, k):
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)

    # Get centroids
    centroids = kmeans.cluster_centers_

    # Each vector is assigned to a cluster
    # labels = kmeans.labels_

    # print("Centroids:", centroids)
    # print("Labels:", labels)
    # print("Length:", len(labels))

    # Now, 'centroids' contains the coordinates of the centroids,
    # and 'labels' contains the cluster assignments for each vector.

    return centroids
