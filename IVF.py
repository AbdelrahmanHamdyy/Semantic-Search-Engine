import numpy as np
import pandas as pd
import csv
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
P = 10  # Probing count


class Node:
    def __init__(self, id, data):
        self.id = id
        self.data = data


class IVF:
    def __init__(self, file_path="saved_db.bin", new_db=True):
        self.data_size = 0
        self.n_clusters = CLUSTERS
        self.n_probe = P
        self.data_file_path = file_path
        self.index_file_path = "index.bin"
        self.centroids_file_path = "centroids.bin"
        self.centroids = None
        self.vectors = []
        self.inverted_index = {}
        self.centroids_dict = {}
        self.iterations = 0

    def calc_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def save_vectors(self, rows):
        with open(self.file_path, "a+") as fout:
            for row in rows:
                id, embed = row["id"], row["embed"]
                row_str = f"{id}," + ",".join([str(e) for e in embed])
                fout.write(f"{row_str}\n")

    def set_number_of_clusters(self):
        if self.data_size == 10000:
            self.n_clusters = 32
        elif self.data_size == 100000:
            self.n_clusters = 64
        elif self.data_size == 1000000:
            self.n_clusters = 128
        elif self.data_size == 5000000:
            self.n_clusters = 512
        elif self.data_size == 10000000:
            self.n_clusters = 1024
        elif self.data_size == 20000000:
            self.n_clusters = 2048

    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
        self.data_size += len(rows)
        self.set_number_of_clusters()
        self.save_vectors(rows)
        self.build_index()

    def read_data(self):
        df = pd.read_csv(self.data_file_path, header=None)

        self.vectors = df.iloc[:, 1:].to_numpy()

    def generate_ivf(self):
        self.inverted_index = {i: [] for i in range(self.n_clusters)}
        self.centroids_dict = {i: [self.centroids[i], 0, 0]
                               for i in range(self.n_clusters)}

        # Assign each vector to the nearest centroid
        similarities = []
        norms = np.linalg.norm(self.centroids, axis=1)
        for vector in self.vectors:
            similarities.append(np.dot(self.centroids, vector) / (
                norms * np.linalg.norm(vector)))
        assigned_centroids = np.argmax(similarities, axis=1)

        for vector_id, centroid_idx in enumerate(assigned_centroids):
            self.centroids_dict[centroid_idx][1] += 1
            vec = self.vectors[vector_id]
            self.inverted_index[centroid_idx].append(Node(vector_id, vec))

        sum = 0
        for _, val in self.centroids_dict.items():
            val[2] = sum
            sum += val[1]

    def save_index(self):
        with open(self.index_file_path, 'wb') as file:
            for vectors in list(self.inverted_index.values()):
                for node in vectors:
                    id_size = 'i'
                    vec_size = 'f' * len(node.data)

                    binary_data = struct.pack(
                        id_size + vec_size, node.id, *node.data)

                    file.write(binary_data)

    def save_centroids(self):
        with open(self.centroids_file_path, 'wb') as file:
            for centroid in list(self.centroids_dict.values()):
                vec_size = 'f' * len(centroid[0])
                count_size = 'i'
                prev_count_size = 'i'

                binary_data = struct.pack(
                    vec_size + count_size + prev_count_size, *centroid[0], centroid[1], centroid[2])

                file.write(binary_data)

    def load_centroids(self):
        dtype = np.dtype([('values', 'f', D), ('x', 'i'), ('y', 'i')])

        return np.memmap(
            self.centroids_file_path, dtype=dtype, mode='r')

    def handle_big_data(self):
        # Define the chunk size (adjust as needed)
        chunk_size = 1000000

        # Iterate through chunks
        for chunk_number, self.vectors in enumerate(pd.read_csv(self.data_file_path, chunksize=chunk_size)):
            print(f"Processing Chunk {chunk_number + 1}")
            # Perform operations on the chunk
            if chunk_number == 0:
                self.centroids = run_kmeans2(
                    self.vectors[:500000], k=self.n_clusters)

            # TODO: Build Index

            # If you want to stop after processing 5 million rows
            if chunk_number + 1 >= self.iterations:
                break

    def build_index(self):
        if (self.data_size > 10000000):
            self.iterations = self.data_size / 1000000
            self.handle_big_data()
        else:
            self.read_data()
            self.centroids = run_kmeans2(
                self.vectors[:100000] if self.data_size > 100000 else self.vectors, k=self.n_clusters)

        self.generate_ivf()

        self.save_centroids()
        self.save_index()

    def retrive(self, query, k):
        centroids_array = self.load_centroids()
        prev_counts = centroids_array['y']
        counts = centroids_array['x']
        centroid_vectors = centroids_array['values']

        similarities = np.dot(centroid_vectors, query[0]) / (
            np.linalg.norm(centroid_vectors, axis=1) * np.linalg.norm(query[0]))

        nearest_centroid_indices = np.argsort(similarities)[-self.n_probe:]

        filename = 'index.bin'
        dtype = np.dtype([('id', 'i'), ('vector', 'f', D)])

        mmapped_array = np.memmap(filename, dtype=dtype, mode='r')
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

        return result_ids
