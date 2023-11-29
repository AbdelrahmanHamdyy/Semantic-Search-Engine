import pandas as pd
from sklearn.cluster import KMeans


def run_kmeans(data):
    # Specify the number of centroids (clusters)
    num_centroids = 30

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=num_centroids, random_state=42)
    kmeans.fit(data)

    # Get centroids
    centroids = kmeans.cluster_centers_

    # Each vector is assigned to a cluster
    labels = kmeans.labels_

    print(centroids)
    print(len(labels))

    # Now, 'centroids' contains the coordinates of the centroids,
    # and 'labels' contains the cluster assignments for each vector.
