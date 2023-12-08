import numpy as np
import bisect
import struct
from PQ import *
from kmeans import *
from worst_case_implementation import *
from dataclasses import dataclass


N = 1000000  # Size of the data
D = 70  # Vector Dimension
K = 5  # TOP_K
RUNS = 10  # Number of Runs

CLUSTERS = 64  # Number of clusters
P = 10  # Probing count


class Node:
    def __init__(self, id, data):
        self.id = id
        self.data = data


class IVF:
    def __init__(self, data_file_path, n_clusters, n_probe):
        self.data_size = 0
        self.n_clusters = n_clusters
        self.n_probe = n_probe
        self.data_file_path = data_file_path
        self.index_file_path = "index.bin"
        self.centroids_file_path = "centroids.bin"
        self.centroids = None
        self.vectors = []
        self.inverted_index = {}
        self.centroids_dict = {}

    def calc_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def save_vectors(self, records):
        with open(self.data_file_path, 'wb') as file:
            for record in records:
                id_size = 'i'
                vec_size = 'f' * len(record["embed"])
                binary_data = struct.pack(
                    id_size + vec_size, record["id"], *record["embed"])
                file.write(binary_data)

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
        self.vectors = list(self.vectors)
        chunk_size = struct.calcsize('i') + (struct.calcsize('f') * D)
        with open(self.data_file_path, 'rb') as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                _, *vector = struct.unpack('I' + 'f' * D, chunk)
                self.vectors.append(vector)
        self.vectors = np.array(self.vectors)

    def generate_ivf(self):
        for i in range(self.n_clusters):
            self.inverted_index[i] = []
            self.centroids_dict[i] = [self.centroids[i], 0, 0]

        # Assign each vector to the nearest centroid
        similarities = []
        for vector in self.vectors:
            vec_nearest = [self.calc_similarity(
                vector, centroid) for centroid in self.centroids]
            similarities.append(vec_nearest)
        assigned_centroids = np.argmax(similarities, axis=1)

        for vector_id, centroid_idx in enumerate(assigned_centroids):
            self.centroids_dict[centroid_idx][1] += 1
            vec = self.vectors[vector_id]
            self.inverted_index[centroid_idx].append(Node(vector_id, vec))

        # Sort centroids_dict and inverted_index
        self.centroids_dict = dict(sorted(self.centroids_dict.items()))
        self.inverted_index = dict(sorted(self.inverted_index.items()))

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
        vec_size = struct.calcsize('f') * D
        count_size = struct.calcsize('i')
        prev_count_size = struct.calcsize('i')
        chunk_size = vec_size + count_size + prev_count_size

        centroids = []
        with open(self.centroids_file_path, "rb") as file:
            while chunk := file.read(chunk_size):
                vec_size = 'f' * D
                count_size = 'i'
                prev_count_size = 'i'
                # Unpacking the binary data
                *values, x, y = struct.unpack(vec_size +
                                              count_size + prev_count_size, chunk)
                centroids.append([values, x, y])

        return centroids

    def build_index(self):
        self.read_data()
        self.centroids = run_kmeans2(
            self.vectors[:100000] if self.data_size > 100000 else self.vectors, k=self.n_clusters)

        self.generate_ivf()

        self.save_centroids()
        self.save_index()

    def retrive(self, query, k):
        centroids_list = self.load_centroids()
        centroid_vectors, counts, prev_counts = zip(*centroids_list)

        similarities = [self.calc_similarity(
            query[0], centroid) for centroid in centroid_vectors]

        nearest_centroid_indices = np.argsort(similarities)[-self.n_probe:]

        # Search in each of the nearest centroids
        nearest_vectors = []
        with open('index.bin', 'rb') as file:
            for centroid_idx in nearest_centroid_indices:
                count = 0
                chunk_size = struct.calcsize(
                    'i') + (struct.calcsize('f') * D)
                file.seek(prev_counts[centroid_idx] * chunk_size)

                # Reading records after the jump
                while count != counts[centroid_idx]:
                    chunk = file.read(chunk_size)
                    id, *vector = struct.unpack('i' + 'f' * D, chunk)

                    nearest_vectors.append(
                        (self.calc_similarity(vector, query[0]), id))

                    count += 1

        return [vector[1] for vector in sorted(nearest_vectors)[-k:]]
