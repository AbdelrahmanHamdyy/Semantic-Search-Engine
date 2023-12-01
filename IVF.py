import numpy as np
import pickle
import bisect
from main import *
from kmeans import *
from evaluation import *
from worst_case_implementation import *
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import struct


N = 1000000  # Size of the data
CLUSTERS = 30  # Number of clusters
P = 5  # Probing count
D = 70  # Vector Dimension

K = 5  # TOP_K
RUNS = 10  # Number of Runs


class Node:
    def __init__(self, id, data):
        self.id = id
        self.data = data


def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def generate_ivf(vectors, centroids):
    inverted_index = {i: [] for i in range(len(centroids))}
    centroids_dict = {i: [centroids[i], 0, 0] for i in range(len(centroids))}

    # Assign each vector to the nearest centroid
    similarities = cosine_similarity(vectors, centroids)
    assigned_centroids = np.argmax(similarities, axis=1)

    for vector_id, centroid_idx in enumerate(assigned_centroids):
        centroids_dict[centroid_idx][1] += 1
        inverted_index[centroid_idx].append(
            Node(vector_id, vectors[vector_id]))

    # Sort centroids_dict and inverted_index
    centroids_dict = dict(sorted(centroids_dict.items()))
    inverted_index = dict(sorted(inverted_index.items()))

    sum = 0
    for _, val in centroids_dict.items():
        val[2] = sum
        sum += val[1]

    return centroids_dict, inverted_index


def save_index(index, path='index.bin'):
    with open(path, 'wb') as file:
        for vectors in index:
            for node in vectors:
                id_size = 'i'
                vec_size = 'f' * len(node.data)

                binary_data = struct.pack(
                    id_size + vec_size, node.id, *node.data)

                file.write(binary_data)


def save_centroids(centroids, path='centroids.bin'):
    with open(path, 'wb') as file:
        for centroid in centroids:
            vec_size = 'f' * len(centroid[0])
            count_size = 'i'
            prev_count_size = 'i'

            binary_data = struct.pack(
                vec_size + count_size + prev_count_size, *centroid[0], centroid[1], centroid[2])

            file.write(binary_data)


def load_centroids():
    vec_size = struct.calcsize('f') * D
    count_size = struct.calcsize('i')
    prev_count_size = struct.calcsize('i')
    chunk_size = vec_size + count_size + prev_count_size

    centroids = []
    with open('centroids.bin', "rb") as file:
        while chunk := file.read(chunk_size):
            vec_size = 'f' * D
            count_size = 'i'
            prev_count_size = 'i'
            # Unpacking the binary data
            *values, x, y = struct.unpack(vec_size +
                                          count_size + prev_count_size, chunk)
            centroids.append([values, x, y])

    return centroids


def build_index(vectors, num_of_clusters):
    centroids = run_kmeans(vectors, k=num_of_clusters)
    centroids_dict, index = generate_ivf(vectors, centroids)

    centroids_dict = list(centroids_dict.values())
    index = list(index.values())

    save_centroids(centroids_dict, "centroids.bin")
    save_index(index, "index.bin")


def search(query, k, centroids, nprobe):
    counts, prev_counts, centroid_vectors = [], [], []
    for centroid_obj in centroids:
        centroid_vectors.append(centroid_obj[0])
        counts.append(centroid_obj[1])
        prev_counts.append(centroid_obj[2])

    # Find the nearest centroid to the query
    similarities = cosine_similarity(query, centroid_vectors)
    nearest_centroid_indices = np.argsort(similarities)[0][-nprobe:]

    # Search in each of the nearest centroids
    nearest_vectors = []
    for centroid_idx in nearest_centroid_indices:
        count = 0
        distances = []
        ids = []
        chunk_size = struct.calcsize('i') + (struct.calcsize('f') * D)
        with open('index.bin', 'rb') as file:
            file.seek(prev_counts[centroid_idx] * chunk_size)

            # Reading records after the jump
            while count != counts[centroid_idx]:
                chunk = file.read(chunk_size)
                id, *vector = struct.unpack('I' + 'f' * D, chunk)
                distances.append(euclidean_distance(vector, query))
                ids.append(id)

                count += 1

        # Select top k vectors in the current centroid
        sorted_distances = np.argsort(distances)[:k]
        for idx in sorted_distances:
            bisect.insort(nearest_vectors, (distances[idx], ids[idx]))

    return [vector[1] for vector in nearest_vectors[:k]]


def run_queries(np_rows, top_k, num_runs, algo, centroids=[], index=[]):
    results = []
    for _ in range(num_runs):
        query = np.random.random((1, D))
        db_ids = []

        tic = time.time()
        if algo == "faiss":
            _, I = index.search(query, top_k)
            db_ids = I[0]
        else:
            db_ids = search(query, top_k, centroids, nprobe=P)
        toc = time.time()
        run_time = toc - tic

        tic = time.time()
        actual_ids = np.argsort(np_rows.dot(query.T).T / (np.linalg.norm(
            np_rows, axis=1) * np.linalg.norm(query)), axis=1).squeeze().tolist()[::-1]
        toc = time.time()
        np_run_time = toc - tic

        results.append(Result(run_time, top_k, db_ids, actual_ids))
    return results


def ivf_faiss():
    data = np.random.random((N, D))
    nlist = CLUSTERS
    quantizer = faiss.IndexFlatL2(D)
    index = faiss.IndexIVFFlat(quantizer, D, nlist)
    index.train(data)
    index.add(data)
    index.nprobe = P

    results = run_queries(data, top_k=K, num_runs=RUNS,
                          algo="faiss", index=index)
    print(eval(results))


def generate_vectors():
    db = VecDBWorst()
    records_np = np.random.random((N, D))
    records_dict = [{"id": i, "embed": list(row)}
                    for i, row in enumerate(records_np)]
    db.insert_records(records_dict)


def ivf(option="build"):
    if option == "build":
        generate_vectors()
        vectors = read_data()
        build_index(vectors, num_of_clusters=CLUSTERS)
    else:
        vectors = read_data()
        centroids = load_centroids()
        res = run_queries(vectors, top_k=K, num_runs=RUNS,
                          algo="ivf", centroids=centroids)
        print(eval(res))


if __name__ == '__main__':
    ivf("build")
    ivf("search")
    # ivf_faiss()
