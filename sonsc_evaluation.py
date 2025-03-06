import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from sklearn.preprocessing import StandardScaler
from sonsc_algorithm import SONSC
from datetime import datetime
import os
from sklearn.datasets import load_iris
from tensorflow import keras

class ClusteringEvaluator:
    def __init__(self):
        self.metrics = {
            'Silhouette Score': silhouette_score,
            'Calinski-Harabasz Score': calinski_harabasz_score,
            'Davies-Bouldin Score': davies_bouldin_score
        }
        
        self.algorithms = {
            'SONSC': SONSC(k_initial=2),
            'K-Means': KMeans(n_init=10),
            'DBSCAN': DBSCAN(),
            'Hierarchical': AgglomerativeClustering()
        }
        
    def evaluate_algorithm(self, X, algorithm_name, algorithm):
        """Evaluate a single clustering algorithm"""
        # Measure execution time
        start_time = time()
        
        # Fit algorithm
        if algorithm_name == 'DBSCAN':
            labels = algorithm.fit_predict(X)
        else:
            algorithm.fit(X)
            labels = algorithm.labels_
            
        execution_time = time() - start_time
        
        # Calculate metrics
        results = {'Algorithm': algorithm_name, 'Execution Time': execution_time}
        
        for metric_name, metric_func in self.metrics.items():
            try:
                score = metric_func(X, labels)
                results[metric_name] = score
            except Exception as e:
                results[metric_name] = None
                print(f"Warning: Could not calculate {metric_name} for {algorithm_name}: {str(e)}")
        
        if algorithm_name == 'SONSC':
            results['RCI Score'] = algorithm.best_rci
            results['Optimal K'] = algorithm.k
        
        return results, labels
    
    def evaluate_all(self, X, dataset_name):
        """Evaluate all clustering algorithms"""
        results = []
        labels_dict = {}
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        for algo_name, algorithm in self.algorithms.items():
            print(f"\nEvaluating {algo_name} on {dataset_name}...")
            result, labels = self.evaluate_algorithm(X_scaled, algo_name, algorithm)
            results.append(result)
            labels_dict[algo_name] = labels
            
        return pd.DataFrame(results), labels_dict
    
    def plot_results(self, results, dataset_name, save_path):
        """Plot evaluation results"""
        # Create directory for plots
        os.makedirs(save_path, exist_ok=True)
        
        # Plotting metrics comparison
        plt.figure(figsize=(15, 10))
        metrics = [col for col in results.columns if col not in ['Algorithm', 'Execution Time']]
        
        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 2, i)
            if results[metric].notna().any():
                sns.barplot(data=results, x='Algorithm', y=metric)
                plt.title(f'{metric} Comparison')
                plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/{dataset_name}_metrics_comparison.png')
        plt.close()
        
        # Execution time comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(data=results, x='Algorithm', y='Execution Time')
        plt.title('Execution Time Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{save_path}/{dataset_name}_execution_time.png')
        plt.close()

def evaluate_datasets():
    """Evaluate SONSC on multiple datasets"""
    evaluator = ClusteringEvaluator()
    all_results = {}
    
    # Setup save directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'results/evaluation_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. IRIS Dataset
    print("\nEvaluating IRIS dataset...")
    iris = load_iris()
    results_iris, labels_iris = evaluator.evaluate_all(iris.data, 'IRIS')
    all_results['IRIS'] = results_iris
    evaluator.plot_results(results_iris, 'IRIS', save_dir)
    
    # 2. MNIST Dataset (subset)
    print("\nEvaluating MNIST dataset...")
    (X_train_mnist, _), _ = keras.datasets.mnist.load_data()
    X_mnist = X_train_mnist[:5000].reshape(5000, -1)
    X_mnist = X_mnist.astype('float32') / 255.0
    results_mnist, labels_mnist = evaluator.evaluate_all(X_mnist, 'MNIST')
    all_results['MNIST'] = results_mnist
    evaluator.plot_results(results_mnist, 'MNIST', save_dir)
    
    # 3. CIFAR-10 Dataset (subset)
    print("\nEvaluating CIFAR-10 dataset...")
    (X_train_cifar, _), _ = keras.datasets.cifar10.load_data()
    X_cifar = X_train_cifar[:5000].reshape(5000, -1)
    X_cifar = X_cifar.astype('float32') / 255.0
    results_cifar, labels_cifar = evaluator.evaluate_all(X_cifar, 'CIFAR-10')
    all_results['CIFAR-10'] = results_cifar
    evaluator.plot_results(results_cifar, 'CIFAR-10', save_dir)
    
    # Save numerical results
    with pd.ExcelWriter(f'{save_dir}/evaluation_results.xlsx') as writer:
        for dataset_name, results in all_results.items():
            results.to_excel(writer, sheet_name=dataset_name, index=False)
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name} Dataset Results:")
        print(results.to_string(index=False))
    
    return all_results, save_dir

if __name__ == "__main__":
    results, save_dir = evaluate_datasets()
    print(f"\nAll evaluation results saved to: {save_dir}") 