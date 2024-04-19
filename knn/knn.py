import heapq

import numpy as np


class KNN:
    def __init__(self, k):
        self.k = k
        self.train_x = None
        self.train_y = None

    def fit(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

    def predict(self, test_x):
        pred_y = []
        for x in test_x:
            q = []
            for i, data_point in enumerate(self.train_x):
                data_label = self.train_y[i]
                dist_to_x = np.linalg.norm(x - data_point)
                if i < self.k:
                    heapq.heappush(q, (-dist_to_x, data_label))
                else:
                    heapq.heappushpop(q, (-dist_to_x, data_label))
            k_labels = [p[1] for p in q]
            if sum(k_labels) / len(k_labels) > 0.5:
                pred_y.append(1)
            else:
                pred_y.append(0)
        return pred_y


def main():
    train_x = np.array([
        [0., 0., 0.], [0., .5, 0.], [1., 1., .5], [1., 1., .6]
    ])
    train_y = np.array([0, 0, 0, 1])
    k = 3
    knn = KNN(k)
    knn.fit(train_x, train_y)
    test_x = np.array([
        [0., 0., 0.1],
        [1., 1., 1.]
    ])
    pred_y = knn.predict(test_x)
    print(f"k = {k}")
    print("Train data:")
    for i, x in enumerate(train_x):
        print(f"\t{x} - label: {train_y[i]}")
    print("Prediction:")
    for i, x in enumerate(test_x):
        print(f"\t{x} - label: {pred_y[i]}")


if __name__ == "__main__":
    main()
