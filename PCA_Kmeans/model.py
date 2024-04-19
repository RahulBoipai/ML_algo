import numpy as np
from tqdm import tqdm

class KMeans:
    def __init__(self, k: int, epsilon: float = 1e-6) -> None:
        self.num_clusters = k
        self.cluster_centers = None
        self.epsilon = epsilon
    
    def fit(self, X: np.ndarray, max_iter: int = 100) -> None:
        # Initialize cluster centers (need to be careful with the initialization,
        # otherwise you might see that none of the pixels are assigned to some
        # of the clusters, which will result in a division by zero error)
        
        pixel ,colors = X.shape
        i = np.random.choice(pixel, self.num_clusters, replace=False) # pick without replacement
        self.cluster_centers = X[i]


        for i in tqdm(range(max_iter)):
            # Assign each sample to the closest prototype
            sq_sum = ((X[:, np.newaxis, :] - self.cluster_centers)**2).sum(axis=2)
            distances = np.sqrt(sq_sum)
            nearest_clusters = np.argmin(distances, axis=1)

            # Update prototypes
            new_cluster_centers = np.zeros((self.num_clusters, colors))
            cluster = []
            for j in range(self.num_clusters):
                #find all the points in the cluster and its new centre
                cluster = X[nearest_clusters == j]
                if(cluster.size == 0):
                    #if cluster is empty then randomly assign new cluster
                    i = np.random.choice(pixel, 1) 
                    new_cluster_centers[j] = X[i]
                else:
                    new_cluster_centers[j] = cluster.mean(axis=0)
                    
            # Check for convergence
            if np.allclose(self.cluster_centers, new_cluster_centers, atol=self.epsilon):
                break
            else:
                self.cluster_centers =  new_cluster_centers  
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Predicts the index of the closest cluster center for each data point
        sq_sum = ((X[:, np.newaxis, :] - self.cluster_centers)**2).sum(axis=2)
        distances = np.sqrt(sq_sum)
        nearest_clusters = np.argmin(distances, axis=1)
        return nearest_clusters
    
    def fit_predict(self, X: np.ndarray, max_iter: int = 100) -> np.ndarray:
        self.fit(X, max_iter)
        return self.predict(X)
    
    def replace_with_cluster_centers(self, X: np.ndarray) -> np.ndarray:
        # Returns an ndarray of the same shape as X
        # Each row of the output is the cluster center closest to the corresponding row in X
        sq_sum = ((X[:, np.newaxis, :] - self.cluster_centers)**2).sum(axis=2)
        distances = np.sqrt(sq_sum)
        nearest_clusters = np.argmin(distances, axis=1)
        return self.cluster_centers[nearest_clusters]