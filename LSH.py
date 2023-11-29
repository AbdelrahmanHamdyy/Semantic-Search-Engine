from datasketch import MinHash, MinHashLSH
import numpy as np

np.random.seed(42)
data = np.random.rand(100, 10)
lsh = MinHashLSH(threshold=0.5, num_perm=128)

M = []
for i, v in enumerate(data):
    m = MinHash(num_perm=128)
    m.update(v)
    M.append(m)
    lsh.insert("m" + str(i), m)


# Create LSH index
query_m = MinHash(num_perm=128)
q = np.random.rand(1, 10)
query_m.update(q)

print(data)
print("Query:", q)

result = lsh.query(query_m)
print("Approximate neighbours with Jaccard similarity > 0.5", result)
