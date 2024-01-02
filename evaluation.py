import numpy as np
import os
from worst_case_implementation import VecDBWorst
from memory_profiler import memory_usage
import time
from dataclasses import dataclass
from typing import List
from IVF_PQ import IVF_PQ
from IVF import IVF
from PQ import PQ
DATA_PATH = "saved_db.csv"
results = []
DB_SEED = 50
QUERY_SEED = 20

@dataclass
class Result:
    run_time: float
    top_k: int
    db_ids: List[int]
    actual_ids: List[int]

def run_queries(db, query, top_k, actual_ids, num_runs):
    global results
    results = []
    for _ in range(num_runs):
        tic = time.time()
        db_ids = db.retrive(query, top_k)
        toc = time.time()
        run_time = toc - tic
        results.append(Result(run_time, top_k, db_ids, actual_ids))
    return results

def memory_usage_run_queries(args):
    global results
    # This part is added to calcauate the RAM usage
    mem_before = max(memory_usage())
    mem = memory_usage(proc=(run_queries, args, {}), interval=1e-3)
    return results, max(mem) - mem_before

def evaluate_result(results: List[Result]):
    # scores are negative. So getting 0 is the best score.
    scores = []
    run_time = []
    for res in results:
        run_time.append(res.run_time)
        # case for retireving number not equal to top_k, socre will be the lowest
        if len(set(res.db_ids)) != res.top_k or len(res.db_ids) != res.top_k:
            scores.append(-1 * len(res.actual_ids) * res.top_k)
            continue
        score = 0
        for id in res.db_ids:
            try:
                ind = res.actual_ids.index(id)
                if ind > res.top_k * 3:
                    score -= ind
            except:
                score -= len(res.actual_ids)
        scores.append(score)

    return sum(scores) / len(scores), sum(run_time) / len(run_time)

def get_actual_ids_first_k(actual_sorted_ids, k):
    return [id for id in actual_sorted_ids if id < k]

def evaluate(size, label, path="saved_db.csv"):
    rng = np.random.default_rng(DB_SEED)

    db = IVF(path)

    records_np = rng.random((10**7*2, 70), dtype=np.float32)
    db.insert_records(records_np[:size])
    
    rng_query = np.random.default_rng(QUERY_SEED)
    query = rng_query.random((1, 70), dtype=np.float32)
    
    actual_ids_20m = np.argsort(records_np.dot(query.T).T / (np.linalg.norm(
        records_np, axis=1) * np.linalg.norm(query)), axis=1).squeeze().tolist()[::-1]

    actual_ids = get_actual_ids_first_k(actual_ids_20m, size)
    res = run_queries(db, query, 5, actual_ids, 10)

    res, mem = memory_usage_run_queries((db, query, 5, actual_ids, 3))
    eval = evaluate_result(res)

    print(f"{label}\tscore\t{eval[0]}\ttime\t{eval[1]:.2f}\tRAM\t{mem:.2f} MB")


if __name__ == "__main__":
    evaluate(10000, "10k")
    evaluate(100000, "100k", "saved_db_100k.csv")
    evaluate(1000000, "1M", "saved_db_1m.csv")
    evaluate(5000000, "5M", "saved_db_5m.csv")
    evaluate(10000000, "10M", "saved_db_10m.csv")
    evaluate(15000000, "15M", "saved_db_15m.csv")
    evaluate(20000000, "20M", "saved_db_20m.csv")
