# Utilities
# Author: Zhe Liu (zl376@cornell.edu)
# Date: 2019-04-13

import pickle
import numpy as np



# Data I/O
def load_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def save_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
        
        

# Util
def lcs(a, b):
    '''
    Length of Longest Common Sequence between {a} and {b}
    '''
    M, N = len(a), len(b)
    dp = np.zeros(shape=(M+1, N+1))
    for i in range(M-2, -2, -1):
        for j in range(N-2, -2, -1):
            if a[i+1] == b[j+1]:
                dp[i, j] = 1 + dp[i+1, j+1]
            else:
                dp[i, j] = max(dp[i+1, j], dp[i, j+1])
    return dp[-1, -1]
