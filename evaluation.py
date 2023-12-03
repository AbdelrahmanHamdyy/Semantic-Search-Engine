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
        actual_ids = np.argsort(np_rows.dot(query.T).T / (np.linalg.norm(
            np_rows, axis=1) * np.linalg.norm(query)), axis=1).squeeze().tolist()
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
    # db = LSH(15 or 25 ,70,1) # --> 1M  () 18sec
    # db = LSH(12 ,70,2) # --> 1M  (-160)
    # db = LSH(11 ,70,15) # --> 1M (-27.6)  15.1  0.5
    # db = LSH(13,70,15) # --> 1M  (-23.4)  10.2  0.5
    # db = LSH(9,70,12) # --> 1M   (-63)    15.1  0.6
    # db = LSH(12 ,70,15) # --> 1M (-8.7)   11.3  0.8
    # db = LSH(15 ,70,15) # --> 1M (-29.0)  7.1   0.5
    # db = LSH(15,70,20) # --> 1M  (-11.0)  7.6   0.7
    # db = LSH(9 ,70,20) # --> 1M  (-22.4)  17.4  0.5
    # db = LSH(12,70,15) # --> 1M  (-62.5)  15.79 0.5
    # db = LSH(12,70,20) # --> 1M  (-17.4)  12.5  0.6
    
    # db = LSH(6,70,15) # --> 1M  (0.0)  50  1.0
    # db = LSH(6,70,10) # --> 1M  (-7.2)  46.04  0.9
    # db = LSH(2,70,2)  # --> 10K  (0.0)   0.34  1.0
    # db = LSH(5,70,6) # --> 100K  (0.0)   3.7   1.0
    # db = LSH(6,70,6) # --> 100K  (0.0)  4.2   1.0

    db = LSH(7,70,10)
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