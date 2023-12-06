import numpy as np


def generate_random_line(input_dim):
    # Generate random coefficients for a hyperplane (line in 2D, plane in 3D, etc.)
    coefficients = np.random.randn(input_dim)
    # Normalize to unit vector
    return coefficients / np.linalg.norm(coefficients)


def above_or_below_line(vector, line):
    # Check if the vector is above (output 1) or below (output 0) the line
    return int(np.dot(vector, line) > 0)


# Example usage
input_dim = 3  # Adjust the dimensionality as needed
random_line = generate_random_line(input_dim)
print("Random Line Coefficients:", random_line)

vector_to_check = np.random.randn(input_dim)
result = above_or_below_line(vector_to_check, random_line)
print("Vector is above the line (1) or below (0):", result)
