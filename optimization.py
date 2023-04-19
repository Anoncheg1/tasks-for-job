import random
import numpy as np
import time

def FFD(s, bs):
    """heuristic - pack to containers
    :return [[8], [8], [7, 2], [7, 2], [6, 3], ...]"""
    bin_sizes=[]
    r = random.randint(0,len(bs)-1)
    remain = [bs[r]]
    bin_sizes.append(bs[r])
    sol = [[]]
    for item in sorted(s, reverse=True):
        for j,free in enumerate(remain):
            if free >= item:
                remain[j] -= item
                sol[j].append(item)
                break
        else:
            sol.append([item])
            r = random.randint(0,len(bs)-1)
            remain.append(bs[r]-item)
            bin_sizes.append(bs[r])
    return sol, bin_sizes

start_time = time.time()

weight = [8.78, 8.77, 8.77, 8.76, 8.74, 8.73, 8.72, 8.7, 8.63, 8.63, 8.62, 8.62, 8.62, 8.61, 8.6, 8.57, 8.57, 8.53, 8.53, 8.5, 8.49, 8.49, 8.47, 8.44, 8.42, 8.41, 8.39, 8.38, 8.37, 8.33, 8.31, 8.31, 8.28, 8.27, 8.27, 8.26, 8.2, 8.15, 8.12, 8.12, 7.85, 7.14, 7.13, 7.06, 6.31, 6.3, 6.03, 6.02, 5.81, 5.78, 5.76, 5.76, 5.75, 5.74, 5.64, 5.31, 5.31]
bins_s = [22.2, 27.6]
c = 12
mer = 99999999999999999
s,b = None, None
for _ in range(10000):
    sol, bs = FFD(weight,bins_s)
    if (len(sol) + sum(bs)) < mer:
        s = sol
        b = bs
        mer = len(sol) + sum(bs)

import collections
print(s)
print(collections.Counter(b))
print("time seconds:", round(time.time() - start_time))
