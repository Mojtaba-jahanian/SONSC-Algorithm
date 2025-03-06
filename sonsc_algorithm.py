import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class SONSC:
    def __init__(self, k_initial=2):
        self.k = k_initial
        self.centers = None
        self.labels = None
        self.best_rci = -np.inf
        
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def assign_clusters(self, X):
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.k))
        
        for i in range(n_samples):
            for j in range(self.k):
                distances[i, j] = self.euclidean_distance(X[i], self.centers[j])
                
        return np.argmin(distances, axis=1)
    
    def update_centers(self, X, labels):
        centers = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centers[i] = np.mean(cluster_points, axis=0)
        return centers
    
    def calculate_cohesion(self, X, labels):
        C = 0
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 1:
                center = self.centers[i]
                similarities = cosine_similarity(cluster_points, [center])
                C += np.sum(similarities)
        return C
    
    def calculate_separation(self, X):
        S = 0
        for i in range(self.k):
            for j in range(i+1, self.k):
                S += cosine_similarity([self.centers[i]], [self.centers[j]])[0][0]
        return S
    
    def calculate_rci(self, C, S):
        if (S + C) == 0:
            return -np.inf
        return (S - C) / (S + C)
    
    def fit(self, X, max_iterations=100, tolerance=1e-4):
        X = StandardScaler().fit_transform(X)
        best_k = self.k
        best_centers = None
        best_labels = None
        
        while True:
            idx = np.random.choice(X.shape[0], self.k, replace=False)
            self.centers = X[idx]
            
            prev_centers = None
            for _ in range(max_iterations):
                self.labels = self.assign_clusters(X)
                self.centers = self.update_centers(X, self.labels)
                
                if prev_centers is not None:
                    if np.all(np.abs(prev_centers - self.centers) < tolerance):
                        break
                prev_centers = self.centers.copy()
            
            C = self.calculate_cohesion(X, self.labels)
            S = self.calculate_separation(X)
            current_rci = self.calculate_rci(C, S)
            
            if current_rci > self.best_rci:
                self.best_rci = current_rci
                best_k = self.k
                best_centers = self.centers.copy()
                best_labels = self.labels.copy()
                self.k += 1
            else:
                self.k = best_k
                self.centers = best_centers
                self.labels = best_labels
                break
        
        return self
    
    def predict(self, X):
        X = StandardScaler().fit_transform(X)
        return self.assign_clusters(X)
    
    def plot_clusters(self, X):
        plt.figure(figsize=(10, 6))
        colors = plt.cm.rainbow(np.linspace(0, 1, self.k))
        
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       color=colors[i], label=f'Cluster {i+1}')
        
        plt.scatter(self.centers[:, 0], self.centers[:, 1], 
                   color='black', marker='x', s=200, label='Centers')
        plt.title('SONSC Clustering Results')
        plt.legend()
        plt.show()

# کد تست برای IRIS
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    
    # بارگذاری داده‌های IRIS
    iris = load_iris()
    X = iris.data
    
    # اجرای الگوریتم
    sonsc = SONSC(k_initial=2)
    sonsc.fit(X)
    
    # نمایش نتایج
    print(f"تعداد بهینه خوشه‌ها: {sonsc.k}")
    print(f"بهترین مقدار RCI: {sonsc.best_rci:.4f}")
    
    # نمایش خوشه‌ها با استفاده از دو ویژگی اول
    sonsc.plot_clusters(X[:, [0, 1]]) 