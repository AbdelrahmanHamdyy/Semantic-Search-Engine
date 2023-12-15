import numpy as np
import pandas as pd
import heapq
import struct
from PQ import *
from kmeans import *
from worst_case_implementation import *


N = 1000000  # Size of the data
D = 70  # Vector Dimension
K = 5  # TOP_K
RUNS = 10  # Number of Runs

CLUSTERS = 32  # Number of clusters
P = 19  # Probing count


class Node:
    def __init__(self, id, data):
        self.id = id
        self.data = data


class VecDB:
    def __init__(self, file_path="saved_db.csv", new_db=True):
        self.index_file_path = "index.bin"
        self.centroids_file_path = "centroids.bin"
        self.n_clusters = CLUSTERS
        self.n_probe = P
        self.data_size = 0
        self.data_file_path = file_path
        if file_path == "saved_db_10k.csv":
            self.data_size = 10000
        elif file_path == "saved_db_100k.csv":
            self.data_size = 100000
        elif file_path == "saved_db_1m.csv":
            self.data_size = 1000000
        elif file_path == "saved_db_5m.csv":
            self.data_size = 5000000
        elif file_path == "saved_db_10m.csv":
            self.data_size = 10000000
        elif file_path == "saved_db_15m.csv":
            self.data_size = 15000000
        elif file_path == "saved_db_20m.csv":
            self.data_size = 20000000
        self.set_number_of_clusters()

        self.centroids = None
        self.vectors = []
        self.index_np = None
        self.centroids_np = None
        self.labels = None
        self.iterations = 0

    def calc_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def save_vectors(self, rows):
        if self.data_size == 10000:
            with open(self.data_file_path, "w") as fout:
                for row in rows:
                    id, embed = row["id"], row["embed"]
                    row_str = f"{id}," + ",".join([str(e) for e in embed])
                    fout.write(f"{row_str}\n")
        else:
            np.savetxt(self.data_file_path, rows, delimiter=',')
            if self.data_size >= 10000000:
                # print("Generating centroids (>10m)")
                self.centroids, _ = run_kmeans2(
                    rows[:1000000], k=self.n_clusters)

    def set_number_of_clusters(self):
        if self.data_size == 10000:
            self.n_clusters = 16
            self.n_probe = 10
            self.data_file_path = "saved_db_10k.csv"
            self.index_file_path = "index_10k.bin"
            self.centroids_file_path = "centroids_10k.bin"
        elif self.data_size == 100000:
            self.n_clusters = 64
            self.n_probe = 10
            self.data_file_path = "saved_db_100k/saved_db_100k.csv"
            self.index_file_path = "saved_db_100k/index_100k.bin"
            self.centroids_file_path = "saved_db_100k/centroids_100k.bin"
        elif self.data_size == 1000000:
            self.n_clusters = 200
            self.n_probe = 14
            self.data_file_path = "saved_db_1m/saved_db_1m.csv"
            self.index_file_path = "saved_db_1m/index_1m.bin"
            self.centroids_file_path = "saved_db_1m/centroids_1m.bin"
        elif self.data_size == 5000000:
            self.n_clusters = 750
            self.n_probe = 32
            self.data_file_path = "saved_db_5m/saved_db_5m.csv"
            self.index_file_path = "saved_db_5m/index_5m.bin"
            self.centroids_file_path = "saved_db_5m/centroids_5m.bin"
        elif self.data_size == 10000000:
            self.n_clusters = 1024
            self.n_probe = 30
            self.data_file_path = "saved_db_10m/saved_db_10m.csv"
            self.index_file_path = "saved_db_10m/index_10m.bin"
            self.centroids_file_path = "saved_db_10m/centroids_10m.bin"
        elif self.data_size == 15000000:
            self.n_clusters = 1536
            self.n_probe = 55
            self.data_file_path = "saved_db_15m/saved_db_15m.csv"
            self.index_file_path = "saved_db_15m/index_15m.bin"
            self.centroids_file_path = "saved_db_15m/centroids_15m.bin"
        elif self.data_size == 20000000:
            self.n_clusters = 2048
            self.n_probe = 77
            self.data_file_path = "saved_db_20m/saved_db_20m.csv"
            self.index_file_path = "saved_db_20m/index_20m.bin"
            self.centroids_file_path = "saved_db_20m/centroids_20m.bin"

    def insert_records(self, rows):
        if self.data_size == 0:
            self.data_size += len(rows)
        # print("Data Size:", self.data_size)
        self.set_number_of_clusters()
        # print("Number of Clusters:", self.n_clusters)
        self.save_vectors(rows)
        # print("Vectors saved")
        self.build_index()
        # print("Index built")

    def read_data(self):
        if self.data_size == 10000:
            df = pd.read_csv(self.data_file_path, header=None)

            self.vectors = df.iloc[:, 1:].to_numpy()
        else:
            self.vectors = np.loadtxt(self.data_file_path, delimiter=',')

    def train_ivf(self):
        similarities = []
        norms = np.linalg.norm(self.centroids, axis=1)
        for vector in self.vectors:
            similarities.append(np.dot(self.centroids, vector) / (
                norms * np.linalg.norm(vector)))
        return np.argmax(similarities, axis=1)

    def populate_index(self, assigned_centroids, offset=0):
        for vector_id, centroid_idx in enumerate(assigned_centroids):
            self.centroids_np['count'][centroid_idx] += 1
            self.index_np[centroid_idx].append(
                Node(vector_id + offset, self.vectors[vector_id]))

        sum_count = np.cumsum(self.centroids_np['count'])
        self.centroids_np['prev_count'] = np.roll(sum_count, shift=1)
        self.centroids_np['prev_count'][0] = 0

    def generate_ivf(self, flag=True, offset=0):
        if flag:
            self.index_np = np.empty(self.n_clusters, dtype=Node)
            self.index_np[:] = [[] for _ in range(self.n_clusters)]

            dtype = [('vector', np.float64, D), ('count', np.int32),
                     ('prev_count', np.int32)]
            self.centroids_np = np.zeros(self.n_clusters, dtype=dtype)

            self.centroids_np['vector'] = self.centroids
            self.centroids_np['count'] = 0
            self.centroids_np['prev_count'] = 0

        assigned_centroids = self.train_ivf()

        self.populate_index(assigned_centroids, offset)

    def save_index(self):
        with open(self.index_file_path, 'wb') as file:
            for vectors in self.index_np:
                for node in vectors:
                    id_size = 'i'
                    vec_size = 'f' * len(node.data)

                    binary_data = struct.pack(
                        id_size + vec_size, node.id, *node.data)

                    file.write(binary_data)

    def save_centroids(self):
        with open(self.centroids_file_path, 'wb') as file:
            for centroid in self.centroids_np:
                vec_size = 'f' * D
                count_size = 'i'
                prev_count_size = 'i'

                binary_data = struct.pack(
                    vec_size + count_size + prev_count_size, *centroid['vector'], centroid['count'], centroid['prev_count'])

                file.write(binary_data)

    def load_centroids(self):
        dtype = np.dtype([('values', 'f', D), ('x', 'i'), ('y', 'i')])

        return np.memmap(
            self.centroids_file_path, dtype=dtype, mode='r+')

    def handle_big_data(self, chunk_size):
        # Iterate through chunks
        for chunk_number, chunk in enumerate(pd.read_csv(self.data_file_path, chunksize=chunk_size, header=None)):
            # print(f"Processing Chunk {chunk_number + 1}")
            self.vectors = np.array(chunk)

            if chunk_number == 0:
                # self.centroids, _ = run_kmeans2(
                #     self.vectors, k=self.n_clusters)
                self.generate_ivf()
                # print("IVF Trained")
            elif chunk_number + 1 <= self.iterations:
                self.generate_ivf(False, offset=chunk_number * chunk_size)
                # print("Index and centroids dicts updated")
            else:
                break
        self.save_index()
        # print("Index file saved")
        self.save_centroids()
        # print("Centroids file saved")

    def handle_max_1m(self, stop_at=1000000):
        self.centroids, self.labels = run_kmeans2(
            self.vectors[:stop_at] if self.data_size > 1000000 else self.vectors, k=self.n_clusters)
        # print("Centroids set")
        self.generate_ivf()
        # print("IVF Trained")
        self.save_centroids()
        # print("Centroids saved")
        self.save_index()
        # print("Index saved")

    def build_index(self):
        if (self.data_size > 1000000):
            # print("Handling Big Data")
            chunk_size = 200000
            if self.data_size == 5000000:
                chunk_size = 1000000
            elif self.data_size == 10000000:
                chunk_size = 500000
            elif self.data_size == 15000000:
                chunk_size = 300000
            self.iterations = self.data_size // chunk_size
            # print("Chunk size:", chunk_size)
            # print("Number of Iterations:", self.iterations)
            self.handle_big_data(chunk_size)
        else:
            self.read_data()
            self.handle_max_1m()

    def retrive(self, query, k):
        centroids_array = self.load_centroids()
        prev_counts = centroids_array['y']
        counts = centroids_array['x']
        centroid_vectors = centroids_array['values']

        similarities = np.dot(centroid_vectors, query[0]) / (
            np.linalg.norm(centroid_vectors, axis=1) * np.linalg.norm(query[0]))

        nearest_centroid_indices = np.argsort(similarities)[-self.n_probe:]

        dtype = np.dtype([('id', 'i'), ('vector', 'f', D)])

        mmapped_array = np.memmap(self.index_file_path, dtype=dtype, mode='r')
        nearest_vectors = []

        for centroid_idx in nearest_centroid_indices:
            start_idx = prev_counts[centroid_idx]
            end_idx = start_idx + counts[centroid_idx]

            # Extract relevant records using slicing
            relevant_records = mmapped_array[start_idx:end_idx]

            # Calculate similarities for all records in one go
            similarities = np.dot(relevant_records['vector'], query[0]) / (
                np.linalg.norm(relevant_records['vector'], axis=1) * np.linalg.norm(query[0]))

            # Combine similarities with record IDs
            nearest_vectors.extend(zip(similarities, relevant_records['id']))

        # Sort the nearest_vectors and get the top-k results
        result_ids = [vector[1] for vector in heapq.nlargest(
            k, nearest_vectors, key=lambda x: x[0])]

        del mmapped_array
        del centroids_array

        return result_ids
