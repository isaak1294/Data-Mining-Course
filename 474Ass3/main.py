from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import lloyd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from scipy.cluster.hierarchy import linkage, dendrogram
from mpl_toolkits.mplot3d import Axes3D

datapath1 = "dataset1.csv"
datapath2 = "dataset2.csv"

# Retrieve the data from the csv
data1 = np.genfromtxt(datapath1, delimiter=',', skip_header=0, filling_values=np.nan)
data2 = np.genfromtxt(datapath2, delimiter=',', skip_header=0, filling_values=np.nan)

datasets = [data1, data2]

for d, data in enumerate(datasets):
    k_values = [2, 5, 8, 20]
    costs = []
    for k in k_values:
        centroids, labels, cost = lloyd.lloyd(data, k, initialization="kmeans")

        print("Final centroids:\n", centroids)
        print("Cluster assignments:\n", labels)
        print(f"Clustering Cost: {cost}")

        costs.append(cost)

        # Plot scatterplot for each k
        if data.shape[1] == 2:  # 2D dataset
            plt.figure(figsize=(8, 6))
            plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)
            plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)
            plt.title(f'2D Scatterplot with {k} Clusters')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.savefig(f"figures\\2d_scatter_k{k}_d{d}.png", dpi=300, bbox_inches='tight')
            plt.close()
        elif data.shape[1] == 3:  # 3D dataset
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis', s=50)
            ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', marker='x', s=200)
            ax.set_title(f'3D Scatterplot with {k} Clusters')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_zlabel('Feature 3')
            plt.savefig(f"figures\\3d_scatter_k{k}_d{d}.png", dpi=300, bbox_inches='tight')
            plt.close()

        centroids, labels, cost = lloyd.lloyd(data, k, initialization="random")

        print("Final centroids:\n", centroids)
        print("Cluster assignments:\n", labels)
        print(f"Clustering Cost: {cost}")

        # Plot scatterplot for each k
        if data.shape[1] == 2:  # 2D dataset
            plt.figure(figsize=(8, 6))
            plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)
            plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)
            plt.title(f'2D Scatterplot with {k} Clusters')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.savefig(f"figures\\random_2d_scatter_k{k}_d{d}.png", dpi=300, bbox_inches='tight')
            plt.close()
        elif data.shape[1] == 3:  # 3D dataset
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis', s=50)
            ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', marker='x', s=200)
            ax.set_title(f'3D Scatterplot with {k} Clusters')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_zlabel('Feature 3')
            plt.savefig(f"figures\\random_3d_scatter_k{k}_d{d}.png", dpi=300, bbox_inches='tight')
            plt.close()

    # Plot k vs cost
    plt.plot(k_values, costs, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Clustering Cost')
    plt.title('k vs Clustering Cost')
    plt.grid(True)
    plt.savefig(f"figures\\k_vs_cost{d}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Perform Agglomerative Clustering with connectivity constraints
    max_clusters = 5  # Set the desired number of clusters
    n_neighbors = 10  # Number of neighbors for connectivity matrix
    linkage_method = 'ward'  # Linkage method for dendrogram and clustering

    # Create a connectivity matrix
    connectivity = kneighbors_graph(data, n_neighbors=n_neighbors, mode='connectivity', include_self=True)

    # Perform Agglomerative Clustering
    clustering = AgglomerativeClustering(n_clusters=max_clusters, linkage=linkage_method, connectivity=connectivity)
    labels = clustering.fit_predict(data)

    print("Agglomerative Clustering Labels:", labels)

    # Compute the linkage matrix for the dendrogram
    linkage_matrix = linkage(data, method=linkage_method)

    # Plot the dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix, truncate_mode='level', p=5)  # Truncate for better visualization
    plt.title(f'Agglomerative Clustering Dendrogram (Linkage: {linkage_method})')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.savefig(f"figures\\dendrogram_d{d}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot scatterplots for Agglomerative Clustering results
    if data.shape[1] == 2:  # 2D dataset
        plt.figure(figsize=(8, 6))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)
        plt.title(f'Agglomerative Clustering (n_clusters={max_clusters}, linkage={linkage_method})')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.savefig(f"figures\\agglomerative_scatter_d{d}.png", dpi=300, bbox_inches='tight')
        plt.close()
    elif data.shape[1] == 3:  # 3D dataset
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis', s=50)
        ax.set_title(f'Agglomerative Clustering (n_clusters={max_clusters}, linkage={linkage_method})')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        plt.savefig(f"figures\\agglomerative_scatter_d{d}.png", dpi=300, bbox_inches='tight')
        plt.close()