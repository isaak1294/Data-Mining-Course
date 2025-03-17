from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import lloyd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist


datapath1 = "dataset1.csv"
datapath2 = "dataset2.csv"

# Retreive the data from the csv
data1 = np.genfromtxt(datapath1, delimiter=',', skip_header=0, filling_values=np.nan)
data2 = np.genfromtxt(datapath2, delimiter=',', skip_header=0, filling_values=np.nan)

datasets = [data1, data2]

for d, data in enumerate(datasets):
    k_values = [2, 3, 5, 8, 20]
    costs = []
    for k in k_values:

        centroids, labels, cost = lloyd.lloyd(data, k, initialization="kmeans")

        print("Final centroids:\n", centroids)
        print("Cluster assignments:\n", labels)
        print(f"Clustering Cost: {cost}")

        costs.append(cost)

    # Plot k vs cost
    plt.plot(k_values, costs, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Clustering Cost')
    plt.title('k vs Clustering Cost')
    plt.grid(True)
    plt.savefig(f"figures\\k_vs_cost{d}", dpi=300, bbox_inches='tight')

    distances = pdist(data)

    # 1. Single Linkage Clustering
    single_linkage = sch.linkage(distances, method='single')

    # 2. Average Linkage Clustering
    average_linkage = sch.linkage(distances, method='average')

    # 3. Generate Dendrograms for both Single Linkage and Average Linkage
    plt.figure(figsize=(12, 6))

    # Plot Single Linkage Dendrogram
    plt.subplot(1, 2, 1)
    sch.dendrogram(single_linkage)
    plt.title('Single Linkage Dendrogram')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    plt.savefig(f"figures\\sink{d}", dpi=300, bbox_inches='tight')

    # Plot Average Linkage Dendrogram
    plt.subplot(1, 2, 2)
    sch.dendrogram(average_linkage)
    plt.title('Average Linkage Dendrogram')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    plt.savefig(f"figures\\dink{d}", dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()

    # 4. Limit the Number of Clusters by Setting maxclust
    max_clusters = 5  # Set the desired number of clusters

    # Cut the dendrogram to get exactly 'max_clusters' clusters
    single_clusters = sch.fcluster(single_linkage, max_clusters, criterion='maxclust')
    average_clusters = sch.fcluster(average_linkage, max_clusters, criterion='maxclust')

    # Print the cluster assignments for both methods
    print("Single Linkage Clusters:", single_clusters)
    print("Average Linkage Clusters:", average_clusters)
