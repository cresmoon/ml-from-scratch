# KNN

## Implementation notes

- A very simple implementation of kNN (k-nearest neighbors) algorithm using `heapq`; the prediction running time is `O(N*log(k))`.
- The python `heapq` by default is a min-heap, but in this case we want a max-heap; that is, whenever we see a new distance, we want to push it to the heap and pop the largest one. The trick is to negate the distance before pushing it to the heap.

## Usage

Example:

```python
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
```
