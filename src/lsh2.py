import numpy as np
from collections import defaultdict
from storage import storage

class LSH:
    # hash_size: the length of the resulting binary hash code
    def __init__(self, hash_size, input_dim, num_hashtables=1):

        self.hash_size = hash_size
        self.input_dim = input_dim
        self.num_hashtables = num_hashtables
        self.storage_config={ 'dict': None }
        self._init_uniform_planes()
        self._init_hashtables()
    
    # initialize uniform planes used to generate binary hash codes
    def _init_uniform_planes(self):

        self.uniform_planes = [self._generate_uniform_planes()
                            for _ in range(self.num_hashtables)]

    def _generate_uniform_planes(self):
        return np.random.random((self.hash_size, self.input_dim))
    
    # initialize hash tables, each hash table is a dictionary
    def _init_hashtables(self):
        self.hash_tables = [storage(self.storage_config, i)
                                for i in range(self.num_hashtables)]
    # hash input_point and store it in the corresponding hash table
    def _hash(self, planes, input_point):
        input_point = np.array(input_point)  # for faster dot product
        projections = np.dot(planes, input_point)
        return "".join(['1' if i > 0 else '0' for i in projections.tolist()])

    def _as_np_array(self, json_or_tuple):
        """ Takes either a JSON-serialized data structure or a tuple that has
        the original input points stored, and returns the original input point
        in numpy array format.
        """
        if isinstance(json_or_tuple, str):
            # JSON-serialized in the case of Redis
            try:
                # Return the point stored as list, without the extra data
                tuples = json.loads(json_or_tuple)[0]
            except TypeError:
                print("The value stored is not JSON-serilizable")
                raise
        else:
            # If extra_data exists, `tuples` is the entire
            # (point:tuple, extra_data). Otherwise (i.e., extra_data=None),
            # return the point stored as a tuple
            tuples = json_or_tuple
    def insert_records(self, data):
        
        for input_point in data:
            for i, table in enumerate(self.hash_tables):
                h = self._hash(self.uniform_planes[i], input_point)
                table.append_val(h, input_point)

    def retrive(self, query_vector, num_results=5):
        candidates = set()
        d_func = LSH.euclidean_dist_square
        candidates_list = []
        for i, table in enumerate(self.hash_tables):
            binary_hash = self._hash(self.uniform_planes[i], query_vector)
            candidates.update(map(tuple, table.get_list(binary_hash)))
        # candidates = set(candidates_list)

        # rank candidates by distance function
        candidates = [(ix, d_func(query_vector, ix))
                      for ix in candidates]
        candidates = sorted(candidates, key=lambda x: x[1])

        return candidates[:num_results] if num_results else candidates

    @staticmethod
    def euclidean_dist_square(x, y):
        diff = np.array(x) - y
        return np.dot(diff, diff)

# create 6-bit hashes for input data of 8 dimensions:
lsh = LSH(6, 8,2)

# index vector
lsh.insert_records([[2,3,4,5,6,7,8,9],[10,12,99,1,5,31,2,3],[10,11,94,1,4,31,2,3]])

# query a data point
top_n = 1
nn = lsh.retrive([1,2,3,4,5,6,7,7], num_results=top_n)
print(nn)

# unpack vector, extra data and vectorial distance
top_n = 3
nn = lsh.retrive([10,12,99,1,5,30,1,1], num_results=top_n)
for (vec,distance) in nn:
    print(vec, distance)