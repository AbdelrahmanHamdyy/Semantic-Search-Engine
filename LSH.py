from datasketch import MinHash, MinHashLSH
import numpy as np
import faiss
import numpy as np
from collections import defaultdict
from storage import storage
import pickle
import heapq

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
        return np.random.randn(self.hash_size, self.input_dim)
    
    # initialize hash tables, each hash table is a dictionary
    def _init_hashtables(self):
        self.hash_tables = [storage(self.storage_config, i)
                                for i in range(self.num_hashtables)]
    # hash input_point and store it in the corresponding hash table
    def _hash(self, planes, input_point):
        input_point = np.array(input_point)  # for faster dot product
        projections = np.dot(planes, input_point.T)
        return "".join(['1' if i > 0 else '0' for i in projections])


    def insert_records(self, data):
        for input_point in data:
            for i, table in enumerate(self.hash_tables):
                h = self._hash(self.uniform_planes[i], input_point["embed"])
                table.append_val(h, input_point)
        # for i, table in enumerate(self.hash_tables):
        # # Precompute hash values for all input points for the current table
        #     hash_values = [self._hash(self.uniform_planes[i], point["embed"]) for point in data]
        #     for j, input_point in enumerate(data):
        #         h = hash_values[j]  # Use the precomputed hash value
        #         # Append the record to the table
        #         table.append_val(h, input_point)

        # for i, table in enumerate(self.hash_tables):
        #     print("-----------------")
        #     print(table.keys())
        # for i, table in enumerate(self.hash_tables):
        #     hashes = [self._hash(self.uniform_planes[i], point["embed"]) for point in data]
        #     table.append_vals(hashes, data)

        # self.save_index(self.hash_tables)


    def save_index(self,index, path='indexLSH.pkl'):
        with open(path, 'wb') as file:
            pickle.dump(index, file)

    def load_index(self,path='indexLSH.pkl'):
        with open(path, 'rb') as file:
            return pickle.load(file)
    def retrive(self, query_vector, num_results=5):
        # hash_tables = self.load_index()
        candidates = set()
        d_func = LSH._cal_score
        for i, table in enumerate(self.hash_tables):
            binary_hash = self._hash(self.uniform_planes[i], query_vector)
            tableList= table.get_list(binary_hash)
            for ele in tableList:
                embed_tuple = tuple(ele['embed'])
                frozenset_dict = frozenset([('id', ele['id']), ('embed', embed_tuple)])
                candidates.add(frozenset_dict)
        # rank candidates by distance function
        candidates = [(ix, d_func(query_vector, dict(ix)["embed"]))
                    for ix in candidates]
        # candidates = sorted(candidates, key=lambda x: x[1])
        # result = [dict(candidate[0])["id"] for candidate in candidates[:num_results]]
        result = heapq.nsmallest(num_results, candidates, key=lambda x: x[1])
        result = [dict(candidate[0])["id"] for candidate in result]

        # result = []
        # for i in range(num_results):
        #     result.append(dict(candidates[i][0])["id"])
        # candidates = [(ix, d_func(query_vector, ix["embed"])) for ix in candidates]
        # candidates = sorted(candidates, key=lambda x: x[1])
        # result = [candidate[0]["id"] for candidate in candidates[:num_results]]

        return result

    @staticmethod
    def euclidean_dist_square(x, y):
        diff = x[0] - y
        return np.dot(diff, diff)
    @staticmethod
    def _cal_score(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

def lsh_faiss():
    data = np.random.random((10000, 70))
    lsh = MinHashLSH(threshold=0.5, num_perm=128)

    M = []
    for i, v in enumerate(data):
        m = MinHash(num_perm=128)
        m.update(v)
        M.append(m)
        lsh.insert("m" + str(i), m)

    # Create LSH index
    query_m = MinHash(num_perm=128)
    q = np.random.rand(1, 70)
    query_m.update(q)

    print(data)
    print("Query:", q)

    result = lsh.query(query_m)
    print("Approximate neighbours with Jaccard similarity > 0.5", result)
