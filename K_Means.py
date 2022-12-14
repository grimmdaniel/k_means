import numpy as np
import pandas as pd
from typing import Tuple

class K_Means:

    def __init__(self, k: int) -> None:
        assert k > 0, f"Number of clusters has to be greater than 0, got: {k}"
        self.k = k
        self.fitted_centroids = None
        self.dist_type = None
        self.wcss = None

    def __calc_dist(self, xi: np.ndarray, xj: np.ndarray, dist_type: str) -> np.float64:
        if xi.shape != xj.shape:
            return np.float64(0)
        
        if dist_type == 'manhattan':
            return np.sum(np.abs(xi - xj))
        elif dist_type == 'euclidean':
            return np.sqrt(np.sum((xi - xj) ** 2))
        elif dist_type == 'chebychev':
            return np.max(np.abs(xi - xj))
        else:
            return np.float64(0)

    def __gen_random_centroids(self, n: int, min_coords: np.ndarray, max_coords: np.ndarray) -> np.ndarray:
        dimensionality: int = min_coords.shape[0]
        point_elements: np.ndarray = np.zeros(shape=(n,dimensionality))
        for k in range(dimensionality):
            point_elements[:,k] = [np.random.uniform(min_coords[k], max_coords[k]) for i in range(n)]
        return point_elements

    def __label_data(self, data: np.ndarray, centroids: np.ndarray, dist_type: str) -> np.ndarray:
        def closest_centroid(point: np.ndarray, centroids: np.ndarray, dist_type: str):
            return np.argmin([self.__calc_dist( point, centroid, dist_type) for centroid in centroids])
        return np.array([[closest_centroid(data[k,:], centroids, dist_type)] for k in range(data.shape[0])])

    def __update_centroids(self, n: int, data: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        for cluster in range(n):
            indices = np.where(np.any(labels == cluster, axis=1))
            points_in_current_cluster = data[indices]
            number_of_points_in_current_cluster: int = len(points_in_current_cluster)
            if number_of_points_in_current_cluster > 0:
                centroids[cluster] = np.sum(points_in_current_cluster, axis=0) / number_of_points_in_current_cluster
        return centroids
    
    def __calculate_wcss(self, data: np.ndarray, labels: np.ndarray, centroids: np.ndarray, dist_type: str) -> np.float64:
        sum = np.float64(0)
        for index in range(self.k):
            indices = np.where(np.any(labels == index, axis=1))
            points_in_current_cluster = data[indices]
            sum += np.sum((points_in_current_cluster - centroids[index]) ** 2)
        return sum

    def fit_predict(self, data: np.ndarray, dist_type: str = 'euclidean', max_iter: int = 20) -> Tuple[np.ndarray,np.ndarray,int]:
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        self.dist_type = dist_type
        old_labels: np.ndarray = np.empty((data.shape[0], 1))
        new_labels: np.ndarray = np.random.randint(self.k, size=(data.shape[0], 1)) 
        # initialize randomly the centroids
        centroids: np.ndarray = self.__gen_random_centroids(self.k, data.min(axis=0), data.max(axis=0))
        i: int = 0
        # REPEAT UNTIL i REACHES MAX_ITERATIONS OR UNTIL NEW LABELS AND OLD LABELS ARE EQUAL 
        while i < max_iter and not (old_labels==new_labels).all():
            old_labels = new_labels
            # get new_labels from data using the current centroid locations 
            new_labels = self.__label_data(data=data, centroids=centroids, dist_type=dist_type)     
            # update the centroid locations based on the new labels 
            centroids = self.__update_centroids(n=self.k, data=data, labels=new_labels, centroids=centroids)
            i += 1
        self.fitted_centroids = centroids
        self.wcss = self.__calculate_wcss(data, new_labels, centroids, dist_type)
        return (new_labels, centroids, i)

    def predict(self, data: np.ndarray) -> np.ndarray:
        if self.fitted_centroids is None:
            print('K-means object has to be fitted before use')
            return None
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        return self.__label_data(data, self.fitted_centroids,self.dist_type)

    @property
    def cluster_centers_(self):
        if self.fitted_centroids is None:
            print('K-means object has to be fitted before getting centroids')
            return None
        else:
            return self.fitted_centroids

    @property
    def inertia_(self):
        if self.wcss is None:
            print('K-means object has to be fitted before getting inertia')
            return None
        else:
            return self.wcss