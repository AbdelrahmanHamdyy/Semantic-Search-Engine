from datasketch import MinHash, MinHashLSH
import numpy as np
import faiss
import numpy as np
from collections import defaultdict
from storage import storage
import pickle
import heapq
import struct

class LSH:
    # hash_size: the length of the resulting binary hash code
    def __init__(self, hash_size, input_dim, num_hashtables=1):
        self.hash_size = hash_size
        self.input_dim = input_dim
        self.num_hashtables = num_hashtables
        self.storage_config={ 'dict': None }
        self._init_uniform_planes()
        self._init_hashtables()
        self.index_file_path = "indexLSH.bin"
        self.hashes_file_path = "hashesLSH.bin"
        self.vectors_file_path = "vectors.bin"
        self.hashes=[]
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

    def save_vectors(self, index):
        with open(self.vectors_file_path, 'wb') as file:
            for vector in index:
                id_size = 'i'
                vec_size = 'f' * len(vector["embed"])
                binary_data = struct.pack(
                    id_size + vec_size, vector["id"], *vector["embed"])
                file.write(binary_data)
    
    def insert_records(self, data):
        self.save_vectors(data)
        chunk_size = struct.calcsize('i') + (struct.calcsize('f') * self.input_dim)
        with open(self.vectors_file_path, 'rb') as file:
            # Reading records after the jump
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                id, *vector = struct.unpack('I' + 'f' * self.input_dim, chunk)
                input_point = {"id": id, "embed": vector}
                for i, table in enumerate(self.hash_tables):
                    h = self._hash(self.uniform_planes[i], input_point["embed"])
                    input_point["embed"]=tuple(input_point["embed"])
                    table.append_val(h, frozenset(input_point.items()))
        index=[]
        count=0
        for table in self.hash_tables:
            hashes={}
            for ele in table.keys():
                hashes[ele]=(len(table.get_list(ele)),count)
                count+=len(table.get_list(ele))
                my_set_of_frozensets=table.get_list(ele)
                index.extend([dict(frozenset_item) for frozenset_item in my_set_of_frozensets])
            self.hashes.append(hashes)
        self.save_index(index)
        # self.save_hashes(hashes)

    def save_index(self, index):
        with open(self.index_file_path, 'wb') as file:
            for vector in index:
                id_size = 'i'
                vec_size = 'f' * len(vector["embed"])
                binary_data = struct.pack(
                    id_size + vec_size, vector["id"], *vector["embed"])
                file.write(binary_data)

    def save_hashes(self, hashes):
        with open(self.hashes_file_path, 'wb') as file:
            for i,hash_ in enumerate(hashes):
                for ele in hash_:
                    vec_size = 'i'
                    count_size = 'i'
                    prev_count_size = 'i'
                    id_ = 'i'
                    binary_data = struct.pack(
                        vec_size + count_size + prev_count_size+id_, ele[0], ele[1], ele[2],i)
                    file.write(binary_data)

    def retrive(self, query_vector, num_results=5):
        # hash_tables = self.load_index()
        candidates = set()
        d_func = LSH._cal_score
        for i, table in enumerate(self.hashes):
            binary_hash = self._hash(self.uniform_planes[i], query_vector)
            tableList= table.get(binary_hash, [])
            if(tableList==[]):
                continue
            count=0
            chunk_size = struct.calcsize('i') + (struct.calcsize('f') * self.input_dim)
            with open(self.index_file_path, 'rb') as file:
                file.seek(tableList[1] * chunk_size)
                # Reading records after the jump
                while count != tableList[0]:
                    chunk = file.read(chunk_size)
                    id, *vector = struct.unpack('I' + 'f' * self.input_dim, chunk)
                    frozenset_dict = frozenset([('id', id), ('embed', tuple(vector))])
                    candidates.add(frozenset_dict)
                    count += 1
        # rank candidates by distance function
        candidates = [(dict(ix)["id"], d_func(query_vector, dict(ix)["embed"]))
                    for ix in candidates]
        candidates = sorted(candidates, key=lambda x: x[1])
        result = [candidate[0] for candidate in candidates[-num_results:]]
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
