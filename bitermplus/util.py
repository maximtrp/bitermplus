import math
import numpy as np
from itertools import combinations, chain
from typing import List, Union
from scipy.sparse import csr


def biterms(m: csr.csr_matrix) -> List:
    B_d = []
    for a in m:
        b_i = [b for b in combinations(np.nonzero(a)[1], 2)]
        B_d.append(b_i)
    return B_d


def topic_summary(P_wz, X, V, M, verbose=True):
    res = {
        'coherence': [0] * len(P_wz),
        'top_words': [[None]] * len(P_wz)
    }
    for z, P_wzi in enumerate(P_wz):
        V_z = np.argsort(P_wzi)[:-(M + 1):-1]
        W_z = V[V_z]

        # calculate topic coherence score -> http://dirichlet.net/pdf/mimno11optimizing.pdf
        C_z = 0
        for m in range(1, M):
            for i in range(m):
                D_vmvl = np.in1d(np.nonzero(X[:, V_z[i]]), np.nonzero(X[:, V_z[m]])).sum(dtype=int) + 1
                D_vl = np.count_nonzero(X[:, V_z[i]])
                if D_vl != 0:
                    C_z += math.log(D_vmvl / D_vl)

        res['coherence'][z] = C_z
        res['top_words'][z] = W_z
        if verbose:
            print('Topic {} | Coherence={:0.2f} | Top words= {}'.format(z, C_z, ' '.join(W_z)))
    print('Average topic coherence for the top words is {}'.format(sum(res['coherence'])/len(res['coherence'])))
    return res
