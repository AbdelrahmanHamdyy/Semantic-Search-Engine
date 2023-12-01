import numpy as np
import pickle
import bisect
from main import *
from kmeans import *
from evaluation import *
from worst_case_implementation import *
from sklearn.metrics.pairwise import cosine_similarity
import faiss


class Node:
    def __init__(self, id, data):
        self.id = id
        self.data = data


# Function to calculate Euclidean distance between two vectors
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def generate_ivf(vectors, centroids):
    inverted_index = {tuple(centroid): [] for centroid in centroids}

    # Assign each vector to the nearest centroid
    similarities = cosine_similarity(vectors, centroids)
    assigned_centroids = np.argmax(similarities, axis=1)

    for vector_id, centroid_idx in enumerate(assigned_centroids):
        inverted_index[tuple(centroids[centroid_idx])].append(
            Node(vector_id, vectors[vector_id]))

    return inverted_index


def save_index(index, path='index.bin'):
    with open(path, 'wb') as file:
        pickle.dump(index, file)


def load_index(path='index.bin'):
    with open(path, 'rb') as file:
        return pickle.load(file)


def build_index(vectors, num_of_clusters):
    centroids = run_kmeans(vectors, k=num_of_clusters)
    inverted_index = generate_ivf(vectors, centroids)
    save_index(inverted_index)


def search(query, k, centroids, inverted_index, nprobe):
    # Find the nearest centroid to the query
    similarities = cosine_similarity(query, centroids)
    nearest_centroid_indices = np.argsort(similarities)[0][-nprobe:]

    # Search in each of the nearest centroids
    nearest_vectors = []
    for centroid_idx in nearest_centroid_indices:
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
    records_np = np.random.random((100000, 70))
    records_dict = [{"id": i, "embed": list(row)}
                    for i, row in enumerate(records_np)]
    db.insert_records(records_dict)


def ivf(option="build"):
    if option == "build":
        generate_vectors()
        vectors = read_data()
        build_index(vectors, num_of_clusters=30)
    else:
        vectors = read_data()
        index = load_index()
        centroids = list(index.keys())
        res = run_queries(vectors, top_k=5, num_runs=10,
                          algo="ivf", centroids=centroids, index=index)
        print(eval(res))


if __name__ == '__main__':
    ivf("build")
    # ivf("search")
    # ivf_faiss()
