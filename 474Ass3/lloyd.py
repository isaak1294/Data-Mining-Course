import numpy as np


import numpy as np

def uniform_random_init(k, data):
    indices = np.random.choice(len(data), size=k, replace=False)
    centers = data[indices]
    return centers

def k_means_init(k, data):
    centers = []
    probabilites = np.zeros(len(data))  # Initialize with zeros
    c = np.random.randint(len(data))
    centers.append(data[c])
    
    for j in range(1, k):  # We already have the first center
        # Compute distances from current centers
        for d in range(len(data)):
            dist = np.inf
            for center in centers:
                dist = min(dist, np.linalg.norm(data[d] - center) ** 2)
            probabilites[d] = dist  # Store the minimum distance for each data point
        
        p_sum = np.sum(probabilites)  # Sum of distances
        probabilites = probabilites / p_sum  # Normalize the distances to probabilities
        
        # Choose a new center based on the probabilities
        new_center = np.random.choice(len(data), p=probabilites)
        centers.append(data[new_center])  # Append the selected center
    
    return centers

def lloyd(data, k, initialization="random", t=1e-4, max_iterations=100):
    if initialization == "random":
        centers = uniform_random_init(k, data)
    elif initialization == "kmeans":
        centers = k_means_init(k, data)
    else:
        raise ValueError("wrong.")
    
    for i in range(max_iterations):
        # Compute Euclidean distance
        distances = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
        
        # Assign each point to the closest center
        labels = np.argmin(distances, axis=1)

        # Update centers
        new_centers = np.array([
            data[labels == j].mean(axis=0) if np.any(labels == j) else np.random.choice(data)
            for j in range(k)
        ])
        
        # Check for convergence
        if np.linalg.norm(new_centers - centers) < t:
            break

        centers = new_centers

    # Compute clustering cost (WCSS)
    clustering_cost = np.sum((data - centers[labels]) ** 2)

    return centers, labels, clustering_cost
