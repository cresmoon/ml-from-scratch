import copy
import random

import numpy as np


class KMeans:
    def __init__(self, k):
        self.k = k
        self.c = []

    def _find_min_centroid(self, x):
        min_i = -1
        min_d = float("inf")
        for i in range(self.k):
            d = np.linalg.norm(x - self.c[i])  # l2-norm
            if d < min_d:
                min_i = i
                min_d = d
        return min_i

    def fit(self, train_x):
        k = self.k
        # init centroids, randomly from population
        self.c = np.array(random.sample(list(train_x), k), dtype="float")
        c_map = [-1] * len(train_x)
        # run k-means
        prev_c = None
        while (prev_c is None) or (np.linalg.norm(self.c - prev_c) > 10**(-9)):  # Frobenius norm
            prev_c = copy.deepcopy(self.c)
            # assign data to centroids
            for i, x in enumerate(train_x):
                min_c_idx = self._find_min_centroid(x)
                c_map[i] = min_c_idx
            # recalculate centroids
            c_size = [0] * k
            for i, x in enumerate(train_x):
                c_idx = c_map[i]
                self.c[c_idx] = (self.c[c_idx] * c_size[c_idx] + x) / (c_size[c_idx] + 1)
                c_size[c_idx] += 1

    def centroids(self):
        return self.c


def main():
    train_x = np.array([
        [0, 1], [1, 0], [1, 2], [2, 1],
        [9, 4], [10, 4], [11, 4]
    ])
    kmeans = KMeans(k=2)
    kmeans.fit(train_x)
    print("Input data:\n", train_x)
    print("Output centroids:\n", kmeans.centroids())


if __name__ == "__main__":
    main()
