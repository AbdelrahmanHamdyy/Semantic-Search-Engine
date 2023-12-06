import faiss
import numpy as np
import time
from dataclasses import dataclass
from typing import List
from evaluation import eval


D = 70
K = 5
RUNS = 10


@dataclass
class Result:
    run_time: float
    top_k: int
    db_ids: List[int]
    actual_ids: List[int]


def run_queries(np_rows, top_k, num_runs, index):
    results = []
    for _ in range(num_runs):
        query = np.random.random((1, D))
        db_ids = []

        tic = time.time()
        _, I = index.search(query, top_k)
        db_ids = I[0]
        toc = time.time()
        run_time = toc - tic

        tic = time.time()
        actual_ids = np.argsort(np_rows.dot(query.T).T / (np.linalg.norm(
            np_rows, axis=1) * np.linalg.norm(query)), axis=1).squeeze().tolist()[::-1]
        toc = time.time()
        np_run_time = toc - tic

        results.append(Result(run_time, top_k, db_ids, actual_ids))
    return results


def run_ivf_faiss(top_k=K, num_runs=RUNS):
    data = np.random.random((100000, D))
    nlist = 64
    quantizer = faiss.IndexFlatL2(D)
    index = faiss.IndexIVFFlat(quantizer, D, nlist)
    index.train(data)
    index.add(data)
    index.nprobe = 10

    results = run_queries(data, top_k, num_runs, index)
    print(eval(results))


def run_ivf_pq_faiss(top_k=K, num_runs=RUNS):
    data = np.random.random((100000, D))
    m = 14
    nbits = 5
    n_clusters = 64

    # Train the IVF with PQ index
    index = faiss.IndexIVFPQ(faiss.IndexFlatL2(
        D), D, n_clusters, m, nbits)
    index.train(data)
    index.add(data)
    index.nprobe = 10
    results = run_queries(data, top_k, num_runs, index)
    print(eval(results))


def run_hnsw_faiss(dim=D, top_k=K, num_runs=RUNS, data=None):
    data = np.random.random((100000, D)).astype('float32')
    n_clusters = 64
    index = faiss.IndexHNSWFlat(dim, n_clusters, faiss.METRIC_L2)
    index.add(data)
    results = run_queries(data, top_k, num_runs, index)
    print(eval(results))


if __name__ == '__main__':
    run_ivf_pq_faiss()
