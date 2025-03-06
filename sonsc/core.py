import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.preprocessing import StandardScaler
from .metrics import calculate_cohesion, calculate_separation
from .utils import euclidean_distance

class SONSC(BaseEstimator, ClusterMixin):
    """Self-Organizing Neural-Selective Clustering Algorithm
    
    Parameters
    ----------
    k_initial : int, default=2
        Initial number of clusters
    max_iterations : int, default=100
        Maximum number of iterations for each clustering attempt
    tolerance : float, default=1e-4
        Convergence tolerance
    random_state : int, default=None
        Random state for reproducibility
        
    Attributes
    ----------
    n_clusters_ : int
        The optimal number of clusters found
    labels_ : ndarray of shape (n_samples,)
        Labels of each point
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Cluster centers
    best_rci_ : float
        Best RCI score achieved
    """
    
    def __init__(self, k_initial=2, max_iterations=100, tolerance=1e-4, random_state=None):
        self.k_initial = k_initial
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        self.k_ = k_initial
        self.labels_ = None
        self.cluster_centers_ = None
        self.best_rci_ = -np.inf
        
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def calculate_cohesion(self, X, labels):
        C = 0
        for i in range(self.k_):
            cluster_points = X[labels == i]
            if len(cluster_points) > 1:
                center = self.cluster_centers_[i]
                similarities = np.array([
                    1 - self.euclidean_distance(point, center)
                    for point in cluster_points
                ])
                C += np.sum(similarities)
        return C
    
    def calculate_separation(self):
        S = 0
        for i in range(self.k_):
            for j in range(i+1, self.k_):
                S += 1 - self.euclidean_distance(
                    self.cluster_centers_[i],
                    self.cluster_centers_[j]
                )
        return S
    
    def fit(self, X, y=None):
        """Fit the SONSC clustering model
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances
        y : Ignored
            Not used, present here for API consistency by convention.
            
        Returns
        -------
        self : object
            Fitted estimator
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        while True:
            # Initialize centers
            idx = np.random.choice(X_scaled.shape[0], self.k_, replace=False)
            centers = X_scaled[idx]
            
            # Clustering iteration
            for _ in range(self.max_iterations):
                old_centers = centers.copy()
                
                # Assign points to clusters
                distances = np.array([
                    [self.euclidean_distance(x, c) for c in centers]
                    for x in X_scaled
                ])
                labels = np.argmin(distances, axis=1)
                
                # Update centers
                for i in range(self.k_):
                    cluster_points = X_scaled[labels == i]
                    if len(cluster_points) > 0:
                        centers[i] = cluster_points.mean(axis=0)
                
                # Check convergence
                if np.all(np.abs(old_centers - centers) < self.tolerance):
                    break
            
            # Calculate clustering quality
            self.cluster_centers_ = centers
            self.labels_ = labels
            
            C = self.calculate_cohesion(X_scaled, labels)
            S = self.calculate_separation()
            current_rci = (S - C) / (S + C) if (S + C) != 0 else -np.inf
            
            # Update best result if improved
            if current_rci > self.best_rci_:
                self.best_rci_ = current_rci
                self.best_labels_ = labels.copy()
                self.best_centers_ = centers.copy()
                self.k_ += 1
            else:
                self.k_ -= 1
                self.labels_ = self.best_labels_
                self.cluster_centers_ = self.best_centers_
                break
        
        # Transform centers back to original scale
        self.cluster_centers_ = self.scaler_.inverse_transform(self.cluster_centers_)
        return self
    
    def predict(self, X):
        """Predict the closest cluster for each sample in X
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict
            
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to
        """
        X_scaled = self.scaler_.transform(X)
        distances = np.array([
            [self.euclidean_distance(x, c) for c in self.scaler_.transform(self.cluster_centers_)]
            for x in X_scaled
        ])
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X, y=None):
        """Fit and predict in one step
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances
        y : Ignored
            Not used, present here for API consistency by convention.
            
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to
        """
        return self.fit(X).labels_ 