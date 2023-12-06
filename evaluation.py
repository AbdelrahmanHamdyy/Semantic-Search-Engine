import numpy as np
from worst_case_implementation import VecDBWorst
import time
from dataclasses import dataclass
from typing import List
from LSH import LSH
AVG_OVERX_ROWS = 10
import pandas as pd


@dataclass
class Result:
    run_time: float
    top_k: int
    db_ids: List[int]
    actual_ids: List[int]


def run_queries(db, np_rows, top_k, num_runs):
    results = []
    for _ in range(num_runs):
        print("ME")
        query = np.random.random((1, 70))

        tic = time.time()
        db_ids = db.retrive(query, top_k)
        toc = time.time()
        run_time = toc - tic
        print("Kasab")
        tic = time.time()
        actual_ids = np.argsort(np_rows.dot(query.T).T / (np.linalg.norm(np_rows, axis=1) * np.linalg.norm(query)), axis= 1).squeeze().tolist()[::-1]
        toc = time.time()
        np_run_time = toc - tic

        results.append(Result(run_time, top_k, db_ids, actual_ids))
    return results


def eval(results: List[Result]):
    print("eval")
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
    acc = sum(1 for num in scores if num == 0)
    return sum(scores) / len(scores), sum(run_time) / len(run_time), acc/len(scores)


if __name__ == "__main__":
    # db = LSH(17,70,17) # --> 1M  (0.0) 6.4   --> loop less
    # db = LSH(16,70,17) # --> 1M  (0.0) 5.7   --> loop less
    # db = LSH(18,70,20) # --> 1M  (0.0) 6.5   --> loop less
    # db = LSH(16,70,16) # --> 1M  (-1.6) 7.8
    # db = LSH(16,70,15) # --> 1M  (-1.6) 8.2
    # db = LSH(15,70,15) # --> 1M  (0.0)  12.6
    # db = LSH(6,70,15) # --> 1M  (0.0)  50  1.0
    # db = LSH(6,70,10) # --> 1M  (-7.2)  46.04  0.9
    # db = LSH(2,70,2)  # --> 10K  (0.0)   0.34  1.0
    # db = LSH(10,70,9)  # --> 10K  (0.0)   0.48  1.0 -->loop less
    # db = LSH(5,70,6) # --> 100K  (0.0)   3.4   1.0
    # db = LSH(6,70,6) # --> 100K  (0.0)  2.9  1.0
    # db = LSH(7,70,10) # --> 100K  (0.0)  3.3  1.0
    # db = LSH(13,70,14) # --> 100K  (0.0)  2.2  1.0
    # db = LSH(13,70,13) # --> 100K  (0.0)  1.5  1.0
    # db = LSH(14,70,14) # --> 100K  (0.0)  1.5  1.0
    # db = LSH(15,70,15) # --> 100K  (0.0)  1.5  1.0 --> loop less
    # db = LSH(15,70,14) # --> 100K  (-1.6)  0.6  1.0 
    # db = LSH(15,70,12) # --> 100K  (-1.6)  0.7  1.0

    db =  LSH(18,70,20)
    # db = VecDBWorst()
    records_np = np.random.random((1000000, 70))
    # df = pd.read_csv("./saved_db.csv")
    # records_np=df.values
    records_dict = [{"id": i, "embed": list(row)}
                    for i,row in enumerate(records_np)]
    _len = len(records_np)
    db.insert_records(records_dict)
    res = run_queries(db, records_np, 5, 10)
    print(eval(res))

    # records_np = np.concatenate([records_np, np.random.random((90000, 70))])
    # records_dict = [{"id": i + _len, "embed": list(row)} for i, row in enumerate(records_np[_len:])]
    # _len = len(records_np)
    # db.insert_records(records_dict)
    # res = run_queries(db, records_np, 5, 10)
    # print(eval(res))

    # records_np = np.concatenate([records_np, np.random.random((900000, 70))])
    # records_dict = [{"id": i + _len, "embed": list(row)} for i, row in enumerate(records_np[_len:])]
    # _len = len(records_np)
    # db.insert_records(records_dict)
    # res = run_queries(db, records_np, 5, 10)
    # eval(res)

    # records_np = np.concatenate([records_np, np.random.random((4000000, 70))])
    # records_dict = [{"id": i + _len, "embed": list(row)} for i, row in enumerate(records_np[_len:])]
    # _len = len(records_np)
    # db.insert_records(records_dict)
    # res = run_queries(db, records_np, 5, 10)
    # eval(res)

    # records_np = np.concatenate([records_np, np.random.random((5000000, 70))])
    # records_dict = [{"id": i + _len, "embed": list(row)} for i, row in enumerate(records_np[_len:])]
    # _len = len(records_np)
    # db.insert_records(records_dict)
    # res = run_queries(db, records_np, 5, 10)
    # eval(res)

    # records_np = np.concatenate([records_np, np.random.random((5000000, 70))])
    # records_dict = [{"id": i +  _len, "embed": list(row)} for i, row in enumerate(records_np[_len:])]
    # _len = len(records_np)
    # db.insert_records(records_dict)
    # res = run_queries(db, records_np, 5, 10)
    # eval(res)