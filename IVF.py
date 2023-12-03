import numpy as np
import bisect
import faiss
import struct
from main import *
from PQ import *
from kmeans import *
from evaluation import *
from worst_case_implementation import *
from sklearn.metrics.pairwise import cosine_similarity


N = 100000  # Size of the data
D = 70  # Vector Dimension
K = 5  # TOP_K
RUNS = 10  # Number of Runs

# 100k
CLUSTERS = 32  # Number of clusters
P = K  # Probing count

# 1M
# CLUSTERS = 500
# P = 100


class Node:
    def __init__(self, id, data):
        self.id = id
        self.data = data


class IVF:
    def __init__(self, data_size, n_clusters, n_probe, dim):
        self.data_size = data_size
        self.n_clusters = n_clusters
        self.n_probe = n_probe
        self.dim = dim
        self.index_file_path = "index.bin"
        self.centroids_file_path = "centroids.bin"
        self.centroids = None
        self.vectors = None

    def euclidean_distance(self, vec1, vec2):
        return np.linalg.norm(vec1 - vec2)

    def generate_ivf(self, quantize=False, pq: PQ = None):
        inverted_index = {i: [] for i in range(len(self.centroids))}
        centroids_dict = {i: [self.centroids[i], 0, 0]
                          for i in range(len(self.centroids))}

        # Assign each vector to the nearest centroid
        similarities = cosine_similarity(self.vectors, self.centroids)
        assigned_centroids = np.argmax(similarities, axis=1)

        for vector_id, centroid_idx in enumerate(assigned_centroids):
            centroids_dict[centroid_idx][1] += 1
            vec = pq.get_compressed_data(
                self.vectors[vector_id]) if quantize else self.vectors[vector_id]
            inverted_index[centroid_idx].append(Node(vector_id, vec))

        # Sort centroids_dict and inverted_index
        centroids_dict = dict(sorted(centroids_dict.items()))
        inverted_index = dict(sorted(inverted_index.items()))

        sum = 0
        for _, val in centroids_dict.items():
            val[2] = sum
            sum += val[1]

        return centroids_dict, inverted_index

    def save_index(self, index):
        with open(self.index_file_path, 'wb') as file:
            for vectors in index:
                for node in vectors:
                    id_size = 'i'
                    vec_size = 'f' * len(node.data)

                    binary_data = struct.pack(
                        id_size + vec_size, node.id, *node.data)

                    file.write(binary_data)

    def save_centroids(self, centroids):
        with open(self.centroids_file_path, 'wb') as file:
            for centroid in centroids:
                vec_size = 'f' * len(centroid[0])
                count_size = 'i'
                prev_count_size = 'i'

                binary_data = struct.pack(
                    vec_size + count_size + prev_count_size, *centroid[0], centroid[1], centroid[2])

                file.write(binary_data)

    def load_centroids(self):
        vec_size = struct.calcsize('f') * self.dim
        count_size = struct.calcsize('i')
        prev_count_size = struct.calcsize('i')
        chunk_size = vec_size + count_size + prev_count_size

        centroids = []
        with open(self.centroids_file_path, "rb") as file:
            while chunk := file.read(chunk_size):
                vec_size = 'f' * self.dim
                count_size = 'i'
                prev_count_size = 'i'
                # Unpacking the binary data
                *values, x, y = struct.unpack(vec_size +
                                              count_size + prev_count_size, chunk)
                centroids.append([values, x, y])

        return centroids

    def build_index(self, quantize=False, pq: PQ = None):
        self.centroids = run_kmeans_minibatch(self.vectors, k=self.n_clusters)
        # self.centroids = run_kmeans2(self.vectors, k=self.n_clusters)
        # self.centroids = run_kmeans(self.vectors, k=self.n_clusters)

        centroids_dict, index = self.generate_ivf(quantize, pq)

        centroids_dict = list(centroids_dict.values())
        index = list(index.values())

        self.save_centroids(centroids_dict)
        self.save_index(index)

    def pq_distance(self, query, target, subvector_size):
        # Split the vectors into subvectors
        query_subvectors = np.split(query, len(query) // subvector_size)
        target_subvectors = np.split(target, len(target) // subvector_size)

        # Calculate the Product Quantization Distance (PQD)
        pqd = sum(np.linalg.norm(q - t)
                  for q, t in zip(query_subvectors, target_subvectors))

        return pqd

    def search(self, query, k, dim=D, pq: PQ = None):
        counts, prev_counts, centroid_vectors = [], [], []
        for centroid_obj in self.centroids:
            centroid_vectors.append(centroid_obj[0])
            counts.append(centroid_obj[1])
            prev_counts.append(centroid_obj[2])

        # Find the nearest centroid to the query
        similarities = cosine_similarity(query, centroid_vectors)
        nearest_centroid_indices = np.argsort(similarities)[0][-self.n_probe:]

        if pq is not None:
            query = pq.get_compressed_data(query)

        # Search in each of the nearest centroids
        nearest_vectors = []
        for centroid_idx in nearest_centroid_indices:
            count = 0
            distances = []
            ids = []
            chunk_size = struct.calcsize('i') + (struct.calcsize('f') * dim)
            with open('index.bin', 'rb') as file:
                file.seek(prev_counts[centroid_idx] * chunk_size)

                # Reading records after the jump
                while count != counts[centroid_idx]:
                    chunk = file.read(chunk_size)
                    id, *vector = struct.unpack('I' + 'f' * dim, chunk)

                    if pq is not None:
                        distances.append(self.pq_distance(
                            np.array(query), np.array(vector), dim))
                    else:
                        distances.append(
                            self.euclidean_distance(vector, query))
                    ids.append(id)

                    count += 1

            sorted_distances = np.argsort(distances)[:k]
            for idx in sorted_distances:
                bisect.insort(nearest_vectors, (distances[idx], ids[idx]))

        return [vector[1] for vector in nearest_vectors[:k]]

    def run_queries(self, np_rows, top_k, num_runs, algo, dim=D, index=[], pq: PQ = None):
        results = []
        for _ in range(num_runs):
            query = np.random.random((1, D))
            db_ids = []

            tic = time.time()
            if algo == "faiss":
                _, I = index.search(query, top_k)
                db_ids = I[0]
            elif algo == "ivf":
                db_ids = self.search(query, top_k)
            else:  # IVF_PQ
                db_ids = self.search(
                    query, top_k, pq=pq, dim=dim)
            toc = time.time()
            run_time = toc - tic

            tic = time.time()
            actual_ids = np.argsort(np_rows.dot(query.T).T / (np.linalg.norm(
                np_rows, axis=1) * np.linalg.norm(query)), axis=1).squeeze().tolist()[::-1]
            toc = time.time()
            np_run_time = toc - tic

            results.append(Result(run_time, top_k, db_ids, actual_ids))
        return results

    def generate_vectors(self):
        db = VecDBWorst()
        records_np = np.random.random((self.data_size, self.dim))
        records_dict = [{"id": i, "embed": list(row)}
                        for i, row in enumerate(records_np)]
        db.insert_records(records_dict)

    def run_ivf_faiss(self, top_k=K, num_runs=RUNS):
        data = np.random.random((self.data_size, self.dim))
        nlist = self.n_clusters
        quantizer = faiss.IndexFlatL2(self.dim)
        index = faiss.IndexIVFFlat(quantizer, self.dim, nlist)
        index.train(data)
        index.add(data)
        index.nprobe = self.n_probe

        results = self.run_queries(data, top_k=top_k, num_runs=num_runs,
                                   algo="faiss", index=index)
        print(eval(results))

    def run_ivf(self, option, top_k=K, num_runs=RUNS):
        if option == "build":
            self.generate_vectors()
            self.vectors = read_data()
            self.build_index()
        else:
            self.vectors = read_data()
            self.centroids = self.load_centroids()
            res = self.run_queries(self.vectors, top_k=top_k, num_runs=num_runs,
                                   algo="ivf")
            print(eval(res))

    def run_ivf_pq(self, top_k=K, num_runs=RUNS):
        self.generate_vectors()
        self.vectors = read_data()
        new_dim = 35
        pq = PQ(20, new_dim, self.dim)
        pq.train(self.vectors)
        self.build_index(self.vectors, True, pq=pq)

        # Search
        self.centroids = self.load_centroids()
        res = self.run_queries(self.vectors, top_k=top_k, num_runs=num_runs,
                               algo="ivf_pq", dim=new_dim, pq=pq)
        print(eval(res))

    def run_ivf_pq_faiss(self, top_k=K, num_runs=RUNS):
        self.generate_vectors()
        data = read_data()

        m = 14
        nbits = 5

        # Train the IVF with PQ index
        index = faiss.IndexIVFPQ(faiss.IndexFlatL2(
            self.dim), self.dim, self.n_clusters, m, nbits)
        index.train(data)
        index.add(data)
        results = self.run_queries(data, top_k=top_k, num_runs=num_runs,
                                   algo="faiss", index=index)
        print(eval(results))

    def run_hnsw_faiss(self, dim, top_k=K, num_runs=RUNS, data=None):
        if data is None:
            data = np.random.random(
                (self.data_size, self.dim)).astype('float32')
        n_clusters = 64
        index = faiss.IndexHNSWFlat(dim, n_clusters, faiss.METRIC_L2)
        index.add(data)
        results = self.run_queries(data, top_k=top_k, num_runs=num_runs,
                                   algo="faiss", index=index)
        print(eval(results))


if __name__ == '__main__':
    ivf = IVF(N, CLUSTERS, P, D)
    ivf.run_ivf("build")
    ivf.run_ivf("search")
    # ivf.run_ivf_faiss()
    # ivf.run_ivf_pq()
    # ivf.run_ivf_pq_faiss()
    # ivf.run_hnsw_faiss()
