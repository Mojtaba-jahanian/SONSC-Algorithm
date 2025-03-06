import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sonsc import SONSC
import seaborn as sns
import pandas as pd
from time import time
import os

def load_and_preprocess_mnist():
    """Load and preprocess MNIST dataset using scikit-learn"""
    print("Loading MNIST dataset...")
    # Load MNIST from OpenML
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    
    # Take a subset for faster processing (using 10000 samples)
    n_samples = 10000
    X = X[:n_samples]
    y = y[:n_samples]
    
    # Normalize to [0,1]
    X = X.astype('float32') / 255.0
    
    print(f"Dataset shape: {X.shape}")
    return X, y

def run_sonsc_mnist():
    """Run SONSC on MNIST dataset"""
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Load and preprocess data
    X, y_true = load_and_preprocess_mnist()
    
    # Initialize and fit SONSC
    print("\nFitting SONSC...")
    start_time = time()
    
    sonsc = SONSC(k_initial=2, random_state=42)
    sonsc.fit(X)
    
    execution_time = time() - start_time
    
    # Print results
    print("\nResults:")
    print(f"Optimal number of clusters: {sonsc.k_}")
    print(f"Best RCI score: {sonsc.best_rci_:.4f}")
    print(f"Execution time: {execution_time:.2f} seconds")
    
    # Visualize cluster representatives
    visualize_clusters(X, sonsc.labels_, sonsc.cluster_centers_)
    
    # Save results
    save_results(sonsc, X, y_true, execution_time)

def visualize_clusters(X, labels, centers):
    """Visualize cluster representatives"""
    n_clusters = len(np.unique(labels))
    
    # Plot cluster representatives
    plt.figure(figsize=(15, 5))
    plt.suptitle("MNIST Cluster Representatives", fontsize=14)
    
    for i in range(min(5, n_clusters)):
        plt.subplot(1, 5, i+1)
        
        # Get a random sample from the cluster
        cluster_samples = X[labels == i]
        if len(cluster_samples) > 0:
            sample = cluster_samples[0].reshape(28, 28)
            plt.imshow(sample, cmap='gray')
            plt.title(f'Cluster {i+1}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/mnist_clusters.png')
    plt.close()
    
    # Plot cluster sizes
    plt.figure(figsize=(10, 5))
    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values)
    plt.title('Cluster Sizes')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Samples')
    plt.savefig('results/cluster_sizes.png')
    plt.close()

def save_results(sonsc, X, y_true, execution_time):
    """Save detailed results"""
    results = {
        'n_clusters': sonsc.k_,
        'rci_score': sonsc.best_rci_,
        'execution_time': execution_time,
        'cluster_sizes': pd.Series(sonsc.labels_).value_counts().to_dict()
    }
    
    # Save to CSV
    pd.DataFrame([results]).to_csv('results/mnist_results.csv', index=False)
    
    # Create detailed report
    with open('results/mnist_report.txt', 'w') as f:
        f.write("SONSC Clustering Results on MNIST\n")
        f.write("================================\n\n")
        f.write(f"Dataset Shape: {X.shape}\n")
        f.write(f"Number of Clusters Found: {sonsc.k_}\n")
        f.write(f"RCI Score: {sonsc.best_rci_:.4f}\n")
        f.write(f"Execution Time: {execution_time:.2f} seconds\n\n")
        
        f.write("Cluster Sizes:\n")
        for cluster, size in sorted(results['cluster_sizes'].items()):
            f.write(f"Cluster {cluster}: {size} samples\n")

def plot_digit_samples(X, labels, n_clusters=5, samples_per_cluster=5):
    """Plot multiple samples from each cluster"""
    plt.figure(figsize=(15, 3*n_clusters))
    plt.suptitle("Sample Images from Each Cluster", fontsize=16)
    
    for i in range(min(n_clusters, len(np.unique(labels)))):
        cluster_samples = X[labels == i]
        for j in range(min(samples_per_cluster, len(cluster_samples))):
            plt.subplot(n_clusters, samples_per_cluster, i*samples_per_cluster + j + 1)
            plt.imshow(cluster_samples[j].reshape(28, 28), cmap='gray')
            plt.axis('off')
            if j == 0:
                plt.ylabel(f'Cluster {i}')
    
    plt.tight_layout()
    plt.savefig('results/cluster_samples.png')
    plt.close()

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("Starting MNIST clustering analysis...")
    run_sonsc_mnist()
    print("\nAnalysis complete. Check 'results' directory for outputs.") 