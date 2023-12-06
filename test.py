from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Generate some random data for demonstration
np.random.seed(42)
data_points = np.random.randn(1000, 2)  # 1000 data points in 2D space

# Specify the number of clusters (you can change this)
num_clusters = 3

# Create and fit the KMeans model with cosine similarity
class CosineKMeans(KMeans):
    def fit(self, X, y=None, sample_weight=None):
        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(X)
        return super().fit(similarity_matrix, sample_weight=sample_weight)

# Create and fit the CosineKMeans model
cosine_kmeans = CosineKMeans(n_clusters=num_clusters, random_state=42)
cosine_kmeans.fit(data_points)

# Get cluster assignments and cluster centers
cluster_assignments = cosine_kmeans.labels_
cluster_centers = cosine_kmeans.cluster_centers_

# Visualize the data and cluster centers
for i in range(num_clusters):
    cluster_data = data_points[cluster_assignments == i]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {i}', alpha=0.5)
    plt.scatter(cluster_centers[i, 0], cluster_centers[i, 1], c=f'C{i}', marker='x', s=200, label=f'Centroid {i}')

plt.title('Cosine Similarity K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
