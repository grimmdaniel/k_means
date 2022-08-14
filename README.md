# k_means
Numpy implementation of K-means clustering algorithm.

Supports Pandas DataFrame and Numy ndarrays as input format.

## Parameters

- init(): initilaizing the K-means object with the number of clusters
  - **k: int**: number of clusters
- fit_predict(): compute cluster centers and predict cluster index for each sample
  - **data**: input data, can be a Pandas DataFrame or Numpy ndarray object
  - **dist_type: str**: type of distance measure to be used, can be **euclidean** (default), **manhattan**, **chebychev**
  - **max_iter: int**: number of maximum iterations, default is **20**
- predict(): predict the closest cluster each sample in X belongs to
  - **data**: input data, can be a Pandas DataFrame or Numpy ndarray object
  
## Attributes

- **cluster_centers_: numpy ndarray**: coordinates of cluster centers
- **inertia_: float**: sum of squared distances of samples to their closest cluster center


## Usage:

```python
from k_means.K_Means import K_Means

kmeans = K_Means(3)
labels, centroids, iterations = kmeans.fit_predict(data)

new_data = np.array([[-1,1],[0,-2],[2,0]])
kmeans.predict(new_data)
```

## Examples

![res1](https://user-images.githubusercontent.com/27343157/184531787-b10a1626-45b9-4a80-9605-20542747a978.png)

![res2](https://user-images.githubusercontent.com/27343157/184531815-94c5207d-35ac-42c5-b365-cea5d704544d.png)
