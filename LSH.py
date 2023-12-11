from datasketch import MinHash, MinHashLSH
import numpy as np
import numpy as np
from collections import defaultdict
import pickle
import heapq
import struct
import h5py

class LSH:
    # hash_size: the length of the resulting binary hash code
    def __init__(self,file_path="saved_db.csv", new_db=True):
        self.data_size = 0
        self.hash_size = 1
        self.input_dim = 70
        self.data_file_path = file_path
        self.num_hashtables = 1
        self.index_file_path = "indexLSH.bin"
        self.hashes_file_path = "hashesLSH.bin"
        self.vectors_file_path = "vectors.bin"
        self.hashes=None
    # initialize uniform planes used to generate binary hash codes
    def _init_uniform_planes(self):
        self.uniform_planes = [self._generate_uniform_planes()
                            for _ in range(self.num_hashtables)]

    def _generate_uniform_planes(self):
        return np.random.randn(self.hash_size, self.input_dim)

    def set_number_of_clusters(self):
        if self.data_size == 10000:
            self.hash_size=10
            self.num_hashtables=9
        elif self.data_size == 100000:
            self.hash_size=15
            self.num_hashtables=15
        elif self.data_size == 1000000:
            self.hash_size=18
            self.num_hashtables=20
        elif self.data_size == 5000000:
            self.hash_size=20
            self.num_hashtables=20
        # elif self.data_size == 10000000:
        #     self.n_clusters = 1024
        # elif self.data_size == 20000000:
        #     self.n_clusters = 2048
        self._init_uniform_planes()
        self._init_hashtables()

    # initialize hash tables, each hash table is a dictionary
    def _init_hashtables(self):
        # self.hash_tables = dict() #[dict() for _ in range(self.num_hashtables)]
        self.hash_tables  = np.empty((2**self.hash_size,), dtype=object)
        self.hash_tables [:] = [set() for _ in range(2**self.hash_size)]
    # hash input_point and store it in the corresponding hash table
    def _hash(self, planes, input_point):
        input_point = np.array(input_point)  # for faster dot product
        projections = np.dot(planes, input_point.T)
        return int("".join(['1' if i > 0 else '0' for i in projections]), 2)

    def save_vectors(self, indexes):
        with open(self.vectors_file_path, 'wb') as file:
            for indx,vector in enumerate(indexes):
                id_size = 'i'
                vec_size = 'f' * self.input_dim
                binary_data = struct.pack(
                    id_size + vec_size, indx, *vector)
                file.write(binary_data)

    
    def insert_records(self, data):
        self.data_size += len(data)
        self.save_vectors(data)
        self.set_number_of_clusters()

        chunk_size = struct.calcsize('i') + (struct.calcsize('f') * self.input_dim)
        with open(self.vectors_file_path, 'rb') as file:
            # Reading records after the jump
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                id, *vector = struct.unpack('I' + 'f' * self.input_dim, chunk)
                # input_point = {"id": id, "embed": vector}
                for i, _ in enumerate(self.uniform_planes):
                    h = self._hash(self.uniform_planes[i], vector)
                    # input_point["embed"]=tuple(input_point["embed"])
                    self.hash_tables[h].add((id,tuple(vector)))
        index=[]
        count=0
        self.hashes  = np.empty((2**self.hash_size,), dtype=object)
        self.hashes [:] = [[] for _ in range(2**self.hash_size)]
        for ele in range(2**self.hash_size):
            # hashes={}
            # for ele in table.keys():
            temp=self.hash_tables[ele]
            self.hashes[ele]=[len(temp),count]
            count+=len(temp)
            my_set_of_frozensets=temp
            index.extend([(frozenset_item) for frozenset_item in my_set_of_frozensets])
            # index.extend([(frozenset_item) for frozenset_item in my_set_of_frozensets])
            # self.hashes.append(hashes)
        self.save_index(index)
        self.save_hashes(self.hashes)
        self.save_uniform_planes()

    def save_uniform_planes(self):
        with open("uniform_planes.bin", 'wb') as file:
            for vector in self.uniform_planes:
                for ele in vector:
                    vec_size = 'f' * self.input_dim
                    binary_data = struct.pack(
                        vec_size, *ele)
                    file.write(binary_data)

    def save_index(self, index):
        with open(self.index_file_path, 'wb') as file:
            for vector in index:
                id_size = 'i'
                vec_size = 'f' * self.input_dim
                binary_data = struct.pack(
                    id_size + vec_size, vector[0], *vector[1])
                file.write(binary_data)

    def save_hashes(self, hashes):
        with open(self.hashes_file_path, 'wb') as file:
            for ele in range(2**self.hash_size):
                vec_size = 'i'
                count_size = 'i'
                prev_count_size = 'i'
                # id_ = 'i'
                binary_data = struct.pack(
                    vec_size + count_size + prev_count_size, ele, hashes[ele][0], hashes[ele][1])
                file.write(binary_data)
    
    def load_hashes(self):
        dtype = np.dtype([('key_hash', 'i'), ('count', 'i'),('prev', 'i')])

        return np.memmap(
            self.hashes_file_path, dtype=dtype, mode='r')
    def load_index(self):
        dtype = np.dtype([('id', 'i'), ('vector', 'f',self.input_dim)])

        return np.memmap(
            self.index_file_path, dtype=dtype, mode='r')
    def load_uniform_planes(self):
        dtype = np.dtype([('vector', 'f',self.input_dim)])
        return np.memmap(
            "uniform_planes.bin", dtype=dtype, mode='r')

    def retrive(self, query_vector, num_results=5):
        nearest_vectors = set()
        d_func = LSH._cal_score

        for i in range(self.num_hashtables):
            binary_hash = self._hash(
                self.load_uniform_planes()['vector'][i * self.hash_size:(i + 1) * self.hash_size],
                query_vector
            )

            hash_tables = self.load_hashes()
            tableList = hash_tables[hash_tables['key_hash'] == binary_hash]

            if len(tableList) == 0:
                continue

            tableList = tableList[0]
            start_idx = tableList[2]
            end_idx = start_idx + tableList[1]

            index = self.load_index()
            similarities = [d_func(index[i][1], query_vector[0]) for i in range(start_idx, end_idx)]
            nearest_vectors.update(zip(similarities, index[start_idx:end_idx]['id']))

        # Sort the nearest_vectors and get the top-k results
        result_ids = [vector[1] for vector in heapq.nlargest(
            num_results, nearest_vectors, key=lambda x: x[0])]

        return result_ids

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
