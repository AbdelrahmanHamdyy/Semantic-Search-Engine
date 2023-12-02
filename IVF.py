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


N = 10000  # Size of the data
CLUSTERS = 30  # Number of clusters
P = 10  # Probing count
D = 70  # Vector Dimension

K = 5  # TOP_K
RUNS = 10  # Number of Runs


class Node:
    def __init__(self, id, data):
        self.id = id
        self.data = data


def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def generate_ivf(vectors, centroids, quantize=False, pq: PQ = None):
    inverted_index = {i: [] for i in range(len(centroids))}
    centroids_dict = {i: [centroids[i], 0, 0] for i in range(len(centroids))}

    # Assign each vector to the nearest centroid
    similarities = cosine_similarity(vectors, centroids)
    assigned_centroids = np.argmax(similarities, axis=1)

    for vector_id, centroid_idx in enumerate(assigned_centroids):
        centroids_dict[centroid_idx][1] += 1
        vec = pq.get_compressed_data(
            vectors[vector_id]) if quantize else vectors[vector_id]
        inverted_index[centroid_idx].append(Node(vector_id, vec))

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


def build_index(vectors, num_of_clusters, quantize=False, pq: PQ = None):
    centroids = run_kmeans(vectors, k=num_of_clusters)
    centroids_dict, index = generate_ivf(vectors, centroids, quantize, pq)

    centroids_dict = list(centroids_dict.values())
    index = list(index.values())

    save_centroids(centroids_dict, "centroids.bin")
    save_index(index, "index.bin")


def pq_distance(query, target, subvector_size):
    # Split the vectors into subvectors
    query_subvectors = np.split(query, len(query) // subvector_size)
    target_subvectors = np.split(target, len(target) // subvector_size)

    # Calculate the Product Quantization Distance (PQD)
    pqd = sum(np.linalg.norm(q - t)
              for q, t in zip(query_subvectors, target_subvectors))

    return pqd


def search(query, k, centroids, nprobe, dim=D, pq: PQ = None):
    counts, prev_counts, centroid_vectors = [], [], []
    for centroid_obj in centroids:
        centroid_vectors.append(centroid_obj[0])
        counts.append(centroid_obj[1])
        prev_counts.append(centroid_obj[2])

    # Find the nearest centroid to the query
    similarities = cosine_similarity(query, centroid_vectors)
    nearest_centroid_indices = np.argsort(similarities)[0][-nprobe:]

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
                    distances.append(pq_distance(
                        np.array(query), np.array(vector), dim))
                else:
                    distances.append(euclidean_distance(vector, query))
                ids.append(id)

                count += 1

        sorted_distances = np.argsort(distances)[:k]
        for idx in sorted_distances:
            bisect.insort(nearest_vectors, (distances[idx], ids[idx]))

    return [vector[1] for vector in nearest_vectors[:k]]


def run_queries(np_rows, top_k, num_runs, algo, centroids=[], index=[], dim=D, pq: PQ = None):
    results = []
    for _ in range(num_runs):
        query = np.random.random((1, D))
        db_ids = []

        tic = time.time()
        if algo == "faiss":
            _, I = index.search(query, top_k)
            db_ids = I[0]
        elif algo == "ivf":
            db_ids = search(query, top_k, centroids, nprobe=P)
        else:  # IVF_PQ
            db_ids = search(query, top_k, centroids, nprobe=P, pq=pq, dim=dim)
        toc = time.time()
        run_time = toc - tic

        tic = time.time()
        actual_ids = np.argsort(np_rows.dot(query.T).T / (np.linalg.norm(
            np_rows, axis=1) * np.linalg.norm(query)), axis=1).squeeze().tolist()[::-1]
        toc = time.time()
        np_run_time = toc - tic

        results.append(Result(run_time, top_k, db_ids, actual_ids))
    return results


def generate_vectors():
    db = VecDBWorst()
    records_np = np.random.random((N, D))
    records_dict = [{"id": i, "embed": list(row)}
                    for i, row in enumerate(records_np)]
    db.insert_records(records_dict)


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


def ivf_pq():
    generate_vectors()
    vectors = read_data()
    new_dim = 35
    pq = PQ(20, new_dim, D)
    pq.train(vectors)
    build_index(vectors, CLUSTERS, True, pq=pq)

    # Search
    centroids = load_centroids()
    res = run_queries(vectors, top_k=K, num_runs=RUNS,
                      algo="ivf_pq", centroids=centroids, dim=new_dim, pq=pq)
    print(eval(res))


def ivf_pq_faiss():
    generate_vectors()
    data = read_data()

    m = 14
    nbits = 5

    # Train the IVF with PQ index
    index = faiss.IndexIVFPQ(faiss.IndexFlatL2(D), D, CLUSTERS, m, nbits)
    index.train(data)
    index.add(data)
    results = run_queries(data, top_k=K, num_runs=RUNS,
                          algo="faiss", index=index)
    print(eval(results))


if __name__ == '__main__':
    # ivf("build")
    # ivf("search")
    # ivf_faiss()
    # ivf_pq()
    ivf_pq_faiss()
