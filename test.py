import numpy as np

# Generate 32 unique sets
two_d_array_of_sets = np.empty((5,), dtype=object)
two_d_array_of_sets[:] = [set() for _ in range(5)]

print(two_d_array_of_sets)