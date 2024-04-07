# K-means

## Implementation notes

- The centroid numpy array is initialized with explicit float `dtype`; otherwise numpy sometimes infers `dtype` from the data as integer, which is undesirable.
- Frobenius norm is used for stopping criteria.
- To track the centroid array at the previous iteration before updating, deepcopy needs to be used to copy the array.

## Usage

Example:

```python
train_x = np.array([
    [0, 1], [1, 0], [1, 2], [2, 1],
    [9, 4], [10, 4], [11, 4]
])
kmeans = KMeans(k=2)
kmeans.fit(train_x)
print("Input data:\n", train_x)
print("Output centroids:\n", kmeans.centroids())
```
