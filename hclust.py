import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage


class hclust():
    def __init__(self, t, factors, metoda="ward"):
        self.x = t[factors].values              #valori tabel date csv
        self.h = linkage(self.x, method=metoda) #distante - inaltimi clusteri

    def calcPartition(self, k=None):        #impartire pe clustere si determinare nr partitiii
        jonctions = self.h.shape[0]
        n = jonctions + 1
        # print(self.h[:,0],"CEVA",self.h[:,1])
        if k is None:
            k_max = np.argmax(self.h[1:, 2] - self.h[:(jonctions - 1), 2])
            k = jonctions - k_max
        else:
            k_max = jonctions - k
        self.k = k
        self.threshold = (self.h[k_max, 2] + self.h[k_max + 1, 2]) / 2
        c = np.arange(n)
        for j in range(jonctions-k+1):
            k1 = self.h[j, 0]
            k2 = self.h[j, 1]
            # print(c,k1,k2)
            c[c == k1] = n + j
            c[c == k2] = n + j
        codes = pd.Categorical(c).codes
        return np.array( ["c"+str(i+1) for i in codes] )