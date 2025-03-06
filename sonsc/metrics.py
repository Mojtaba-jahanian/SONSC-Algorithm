import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cohesion(X, centers, labels):
    """Calculate cluster cohesion"""
    C = 0
    for i in range(len(centers)):
        cluster_points = X[labels == i]
        if len(cluster_points) > 1:
            similarities = cosine_similarity(cluster_points, [centers[i]])
            C += np.sum(similarities)
    return C

def calculate_separation(centers):
    """Calculate cluster separation"""
    S = 0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            S += cosine_similarity([centers[i]], [centers[j]])[0][0]
    return S 