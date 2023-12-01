import numpy as np

class LSH:
    def __init__(self, num_hashes, data_dim):
        num_hyper_planes=5 #number of random hyper planes to be generated
    vectorizer = TfidfVectorizer(ngram_range=(ngram_min,ngram_max),max_features=max_features,min_df=min_df)
    train_vectors= vectorizer.fit_transform(train_data.text).toarray()
    #generating random hyper planes
    random_hyper_planes=[]
    for i in range(num_hyper_planes):
        random_hyper_planes.append(np.random.normal(0,1,train_vectors.shape[1]))
    planes=np.array(random_hyper_planes).reshape(train_vectors.shape[1],num_hyper_planes)
    dist_train= np.dot(train_vectors,planes)

    def generate_hash_function(self):
        # Generating a random hyperplane
        return np.random.randn(self.data_dim)

    def hash_data_point(self, data_point, hash_function):
        # Hash a data point using a hash function
        return int(np.dot(data_point, hash_function) > 0)

    def hash_and_store(self, data_point):
        # Hash a data point using all hash functions and store in corresponding buckets
        for i, hash_function in enumerate(self.hash_functions):
            hash_code = self.hash_data_point(data_point, hash_function)
            if hash_code not in self.hash_tables[i]:
                self.hash_tables[i][hash_code] = []
            self.hash_tables[i][hash_code].append(data_point)

    def query(self, query_point):
        # Convert NumPy array to tuple for hashability
        query_point_tuple = tuple(query_point)

        # Query for similar points to the input query point
        similar_points = set()
        for i, hash_function in enumerate(self.hash_functions):
            hash_code = self.hash_data_point(query_point_tuple, hash_function)
            if hash_code in self.hash_tables[i]:
                # Convert NumPy arrays to tuples before updating the set
                similar_points.update(map(tuple, self.hash_tables[i][hash_code]))

        return similar_points

# Example Usage:
# Assuming each data point is a 3D vector
data_dimension = 3
# Number of hash functions
num_hashes = 5

# Create LSH object
lsh = LSH(num_hashes, data_dimension)

# Generate some random data points
data_points = np.random.rand(10, data_dimension)

# Hash and store data points
for point in data_points:
    lsh.hash_and_store(point)

# Query for similar points to a random query point
query_point = np.random.rand(data_dimension)
similar_points = lsh.query(query_point)

print("Query Point:", query_point)
print("Similar Points:", similar_points)
