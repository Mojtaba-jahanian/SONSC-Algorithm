from sklearn.datasets import load_iris
from sonsc import SONSC
import matplotlib.pyplot as plt

def main():
    # Load data
    iris = load_iris()
    X = iris.data
    
    # Create and fit SONSC
    sonsc = SONSC(k_initial=2, random_state=42)
    sonsc.fit(X)
    
    # Print results
    print(f"Optimal number of clusters: {sonsc.k_}")
    print(f"Best RCI score: {sonsc.best_rci_:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.scatter(X[:, 0], X[:, 1], c=sonsc.labels_)
    plt.scatter(sonsc.cluster_centers_[:, 0], 
               sonsc.cluster_centers_[:, 1], 
               marker='x', s=200, linewidths=3, 
               color='r', label='Centroids')
    plt.title('SONSC Clustering on Iris Dataset')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main() 