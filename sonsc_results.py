import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.datasets import load_iris
from datetime import datetime
import os
from sonsc_algorithm import SONSC
from sklearn.preprocessing import StandardScaler

def setup_directories():
    """Create directories for saving results"""
    base_dir = 'results'
    dirs = {
        'figures': os.path.join(base_dir, 'figures'),
        'data': os.path.join(base_dir, 'data'),
        'models': os.path.join(base_dir, 'models')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def run_sonsc_iris(dirs, timestamp):
    """Run SONSC on IRIS dataset"""
    print("\n=== IRIS Dataset Analysis ===")
    
    # Load and preprocess data
    iris = load_iris()
    X = iris.data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run SONSC
    sonsc = SONSC(k_initial=2)
    sonsc.fit(X_scaled)
    
    # Save results
    results = {
        'dataset': 'IRIS',
        'optimal_k': sonsc.k,
        'rci_score': sonsc.best_rci,
        'n_samples': X.shape[0],
        'n_features': X.shape[1]
    }
    
    # Plotting
    plt.figure(figsize=(15, 5))
    
    # Plot first two features
    plt.subplot(1, 3, 1)
    for i in range(sonsc.k):
        mask = sonsc.labels == i
        plt.scatter(X_scaled[mask, 0], X_scaled[mask, 1], 
                   label=f'Cluster {i+1}')
    plt.title('IRIS Clustering (Features 1-2)')
    plt.legend()
    
    # Plot features 3-4
    plt.subplot(1, 3, 2)
    for i in range(sonsc.k):
        mask = sonsc.labels == i
        plt.scatter(X_scaled[mask, 2], X_scaled[mask, 3], 
                   label=f'Cluster {i+1}')
    plt.title('IRIS Clustering (Features 3-4)')
    plt.legend()
    
    # Plot RCI convergence
    plt.subplot(1, 3, 3)
    plt.bar(['RCI Score'], [sonsc.best_rci])
    plt.title('Final RCI Score')
    
    plt.tight_layout()
    plt.savefig(f"{dirs['figures']}/iris_results_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

def run_sonsc_mnist(dirs, timestamp):
    """Run SONSC on MNIST dataset"""
    print("\n=== MNIST Dataset Analysis ===")
    
    # Load and preprocess data
    (X_train, y_train), _ = keras.datasets.mnist.load_data()
    X = X_train[:5000].reshape(5000, -1)  # Using subset for speed
    X = X.astype('float32') / 255.0
    
    # Run SONSC
    sonsc = SONSC(k_initial=2)
    sonsc.fit(X)
    
    # Save results
    results = {
        'dataset': 'MNIST',
        'optimal_k': sonsc.k,
        'rci_score': sonsc.best_rci,
        'n_samples': X.shape[0],
        'n_features': X.shape[1]
    }
    
    # Plotting cluster representatives
    plt.figure(figsize=(15, 5))
    for i in range(min(5, sonsc.k)):
        plt.subplot(1, 5, i+1)
        cluster_samples = X[sonsc.labels == i][:1]
        if len(cluster_samples) > 0:
            plt.imshow(cluster_samples[0].reshape(28, 28), cmap='gray')
            plt.title(f'Cluster {i+1}')
        plt.axis('off')
    
    plt.suptitle('MNIST Cluster Representatives')
    plt.savefig(f"{dirs['figures']}/mnist_results_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

def run_sonsc_cifar10(dirs, timestamp):
    """Run SONSC on CIFAR-10 dataset"""
    print("\n=== CIFAR-10 Dataset Analysis ===")
    
    # Load and preprocess data
    (X_train, y_train), _ = keras.datasets.cifar10.load_data()
    X = X_train[:5000].reshape(5000, -1)  # Using subset for speed
    X = X.astype('float32') / 255.0
    
    # Run SONSC
    sonsc = SONSC(k_initial=2)
    sonsc.fit(X)
    
    # Save results
    results = {
        'dataset': 'CIFAR-10',
        'optimal_k': sonsc.k,
        'rci_score': sonsc.best_rci,
        'n_samples': X.shape[0],
        'n_features': X.shape[1]
    }
    
    # Plotting cluster representatives
    plt.figure(figsize=(15, 5))
    for i in range(min(5, sonsc.k)):
        plt.subplot(1, 5, i+1)
        cluster_samples = X[sonsc.labels == i][:1]
        if len(cluster_samples) > 0:
            plt.imshow(cluster_samples[0].reshape(32, 32, 3))
            plt.title(f'Cluster {i+1}')
        plt.axis('off')
    
    plt.suptitle('CIFAR-10 Cluster Representatives')
    plt.savefig(f"{dirs['figures']}/cifar10_results_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

def save_results(all_results, dirs, timestamp):
    """Save all results to files"""
    # Create DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Save to CSV
    csv_path = f"{dirs['data']}/sonsc_results_{timestamp}.csv"
    df_results.to_csv(csv_path, index=False)
    
    # Save to Excel with multiple sheets
    excel_path = f"{dirs['data']}/sonsc_detailed_results_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        df_results.to_excel(writer, sheet_name='Summary', index=False)
        
        # Add individual dataset sheets
        for result in all_results:
            dataset = result['dataset']
            df_detail = pd.DataFrame({
                'Metric': ['Optimal K', 'RCI Score', 'Number of Samples', 'Number of Features'],
                'Value': [
                    result['optimal_k'],
                    result['rci_score'],
                    result['n_samples'],
                    result['n_features']
                ]
            })
            df_detail.to_excel(writer, sheet_name=f'{dataset} Details', index=False)
    
    # Create comparative visualization
    plt.figure(figsize=(15, 5))
    
    # Plot Optimal K
    plt.subplot(1, 2, 1)
    sns.barplot(data=df_results, x='dataset', y='optimal_k')
    plt.title('Optimal Number of Clusters by Dataset')
    plt.xticks(rotation=45)
    
    # Plot RCI Scores
    plt.subplot(1, 2, 2)
    sns.barplot(data=df_results, x='dataset', y='rci_score')
    plt.title('RCI Scores by Dataset')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{dirs['figures']}/comparative_results_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nResults saved to:")
    print(f"- CSV: {csv_path}")
    print(f"- Excel: {excel_path}")
    print(f"- Figures: {dirs['figures']}")

def main():
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dirs = setup_directories()
    
    # Run analysis on all datasets
    results = []
    results.append(run_sonsc_iris(dirs, timestamp))
    results.append(run_sonsc_mnist(dirs, timestamp))
    results.append(run_sonsc_cifar10(dirs, timestamp))
    
    # Save all results
    save_results(results, dirs, timestamp)
    
    # Print summary
    print("\n=== Final Results Summary ===")
    df_summary = pd.DataFrame(results)
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    main() 