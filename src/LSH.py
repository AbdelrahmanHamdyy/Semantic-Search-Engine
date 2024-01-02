import numpy as np
import numpy as np
import heapq
import struct

class LSH:
    # hash_size: the length of the resulting binary hash code
    def __init__(self,file_path="1M/saved_db.csv", new_db=True):
        self.data_size = 0
        self.hash_size = 1
        self.input_dim = 70
        self.data_file_path = file_path
        self.num_hashtables = 1
        self.index_file_path = "1M/indexLSH.bin"
        self.hashes_file_path = "1M/hashesLSH.bin"
        self.vectors_file_path = "1M/vectors.bin"
        self.uniform_planes_file_path = "1M/uniform_planes.bin"
        self.hashes=None

    # initialize uniform planes used to generate binary hash codes
    def _init_uniform_planes(self):
        self.uniform_planes = [self._generate_uniform_planes()
                            for _ in range(self.num_hashtables)]

    def _generate_uniform_planes(self):
        return np.random.randn(self.hash_size, self.input_dim)

    def set_number_of_clusters(self):
        if self.data_size == 10000:
            self.hash_size=8
            self.num_hashtables=3
        elif self.data_size == 100000:
            self.hash_size=15
            self.num_hashtables=15
        elif self.data_size == 1000000:
            self.hash_size=18
            self.num_hashtables=10
        elif self.data_size == 5000000:
            self.hash_size=20
            self.num_hashtables=20
        self._init_uniform_planes()
        self._init_hashtables()

    # initialize hash tables, each hash table is a dictionary
    def _init_hashtables(self):
        # self.hash_tables = dict() #[dict() for _ in range(self.num_hashtables)]
        self.hash_tables  = np.empty((self.num_hashtables,2**self.hash_size,), dtype=object)
        self.hash_tables [:,:] = [set() for _ in range(2**self.hash_size)]
        
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
        del data
        self.set_number_of_clusters()

        chunk_size = struct.calcsize('i') + (struct.calcsize('f') * self.input_dim)
        with open(self.vectors_file_path, 'rb') as file:
            # Reading records after the jump
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                id, *vector = struct.unpack('I' + 'f' * self.input_dim, chunk)
                for i, _ in enumerate(self.uniform_planes):
                    h = self._hash(self.uniform_planes[i], vector)
                    self.hash_tables[i][h].add((id,tuple(vector)))
        self.save_hashes()
        self.save_uniform_planes()

    def save_uniform_planes(self):
        with open( self.uniform_planes_file_path , 'wb') as file:
            for vector in self.uniform_planes:
                for ele in vector:
                    vec_size = 'f' * self.input_dim
                    binary_data = struct.pack(
                        vec_size, *ele)
                    file.write(binary_data)
     
    def save_hashes(self):
        count=0
        with open(self.index_file_path, 'wb') as file_vector, open(self.hashes_file_path, 'wb') as file:
            for i in range(self.num_hashtables):
                hash_size = 'i'
                count_size = 'i'
                prev_count_size = 'i'
                binary_data_hashes = bytearray()
                
                for ele in range(2**self.hash_size):
                    temp = self.hash_tables[i][ele]
                    for vec in temp:
                        id_size = 'i'
                        vec_size = 'f' * self.input_dim
                        binary_data_vector = struct.pack(id_size + vec_size, vec[0], *vec[1])
                        file_vector.write(binary_data_vector)

                    binary_data_hashes += struct.pack(hash_size + count_size + prev_count_size, ele, len(temp), count)
                    count += len(temp)
                file.write(binary_data_hashes)

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
            self.uniform_planes_file_path , dtype=dtype, mode='r')

    def retrive(self, query_vector, num_results=5):
        nearest_vectors = set()
        d_func = LSH._cal_score

        for i in range(self.num_hashtables):
            binary_hash = self._hash(
                self.load_uniform_planes()['vector'][i * self.hash_size:(i + 1) * self.hash_size],
                query_vector
            )

            hash_tables = self.load_hashes()[i* 2** self.hash_size:(i+1)*2**self.hash_size]
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
