import numpy as np

def pca(X, num_components):
    # Standardize the data
    X_mean = np.mean(X, axis=0)
    X_std = X - X_mean
    covariance_matrix = np.cov(X_std, rowvar=False)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort eigenvectors based on eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_indices = sorted_indices[:num_components]
    top_eigenvectors = eigenvectors[:, top_indices]

    # Project the data onto the top eigenvectors
    principal_components = np.dot(X_std, top_eigenvectors)

    return principal_components

# Example usage:
# Assuming 'data' is your dataset, and you want to reduce it to 2 principal components
data = np.random.rand(10, 5)  # 100 samples with 5 features each
num_components = 2
result = pca(data, num_components)
print(data)
print("Original data shape:", data.shape)
print("Reduced data shape:", result.shape)
