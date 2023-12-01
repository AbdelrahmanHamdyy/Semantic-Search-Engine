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

class Node:
    def __init__(self, id, data):
        self.id = id
        self.data = data


# Function to calculate Euclidean distance between two vectors
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def generate_ivf(vectors, centroids):
    inverted_index = {i: [] for i in range(len(centroids))}
    # Dictionary --> centroid_vector: sum
    centroids_dict = {i: [centroids[i], 0, 0] for i in range(len(centroids))}

    # Assign each vector to the nearest centroid
    similarities = cosine_similarity(vectors, centroids)
    assigned_centroids = np.argmax(similarities, axis=1)
    print(len(vectors))

    # Size of 10000
    # kol element feha --> [0, 5, 3, 20, 26...]
    for vector_id, centroid_idx in enumerate(assigned_centroids):
        centroids_dict[centroid_idx][1] += 1
        inverted_index[centroid_idx].append(
            Node(vector_id, vectors[vector_id]))

    # Sort centroids_dict and inverted_index
    centroids_dict = dict(sorted(centroids_dict.items()))
    inverted_index = dict(sorted(inverted_index.items()))

    sum = 0
    for key, val in centroids_dict.items():
        val[2] = sum
        sum += val[1]

    return centroids_dict, inverted_index


def save_index(data, path='index.bin'):
    with open(path, 'wb') as file:
        # pickle.dump(data, file)
        for ele in data:
            for node in ele:
                binary_data = struct.pack('I' + 'f' * len(node.data), node.id, *node.data)
                file.write(binary_data)

def save_centroids(data, path='centroids.bin'):
    with open(path, 'wb') as file:
        for ele in data:
            binary_data = struct.pack( 'f' * len(ele[0])+'I' +'I' ,  *ele[0],ele[1],ele[2])
            file.write(binary_data)

def load(path='index.bin'):
    with open(path, 'rb') as file:
        return pickle.load(file)


def build_index(vectors, num_of_clusters):
    centroids = run_kmeans(vectors, k=num_of_clusters)
    centroids_dict, inverted_index = generate_ivf(vectors, centroids)
    # Save to centroids.bin
    print(list(centroids_dict.values()))
    for v in list(inverted_index.values()):
        for node in v:
            print("VVV", node.id, node.data)
    save_centroids(list(centroids_dict.values()), "centroids.bin")
    save_index(list(inverted_index.values()), "index.bin")


def search(query, k, centroids, inverted_index, nprobe):
    # Find the nearest centroid to the query
    similarities = cosine_similarity(query, centroids)
    nearest_centroid_indices = np.argsort(similarities)[0][-nprobe:]
    # [4, 1, 7]

    # Search in each of the nearest centroids
    nearest_vectors = []
    for centroid_idx in nearest_centroid_indices:
        # 4
        record_size = 70  # Assuming each record is 70 bytes
        count = 0
        with open('your_file.txt', 'rb') as file:
            # Jumping 100 bytes
            file.seek(record_size * centroids[centroid_idx][2] * 8)

            # Reading records after the jump
            while count != centroids[centroid_idx][1]:
                # Reading a record (70 bytes)
                record = file.read(record_size * 8)
                print(record)
                # Process the record (you can print or do something with it)
                count += 1

        centroid = centroids[centroid_idx]

        # Find vectors in the current centroid
        centroid_vectors = inverted_index[centroid]

        # Calculate distances to the query
        distances = [euclidean_distance(vector.data, query)
                     for vector in centroid_vectors]

        # Select top k vectors in the current centroid
        sorted_distances = np.argsort(distances)[:k]
        for idx in sorted_distances:
            node = centroid_vectors[idx]
            bisect.insort(nearest_vectors, (distances[idx], node.id))

    # Return ids of these vectors
    result = [vector[1] for vector in nearest_vectors[:k]]

    return result


def run_queries(np_rows, top_k, num_runs, algo, centroids=[], index=[]):
    results = []
    for _ in range(num_runs):
        query = np.random.random((1, 70))
        db_ids = []

        tic = time.time()
        if algo == "faiss":
            D, I = index.search(query, top_k)
            db_ids = I[0]
        else:
            db_ids = search(query, top_k, centroids, index, nprobe=10)
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
    data = np.random.random((100000, 70))
    d = 70
    nlist = 30
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    index.train(data)
    index.add(data)
    index.nprobe = 10

    results = run_queries(data, top_k=5, num_runs=10,
                          algo="faiss", index=index)
    print(eval(results))


def generate_vectors():
    db = VecDBWorst()
    records_np = np.random.random((10, 7))
    records_dict = [{"id": i, "embed": list(row)}
                    for i, row in enumerate(records_np)]
    db.insert_records(records_dict)


def ivf(option="build"):
    if option == "build":
        generate_vectors()
        vectors = read_data()
        build_index(vectors, num_of_clusters=3)
    else:
        vectors = read_data()
        centroids = load("centroids.bin")
        res = run_queries(vectors, top_k=5, num_runs=10,
                          algo="ivf", centroids=centroids, index=index)
        print(eval(res))


if __name__ == '__main__':
    ivf("build")
    # ivf("search")
    # ivf_faiss()
