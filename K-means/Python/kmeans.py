import numpy as np
import pandas as pd
import numpy.matlib
import matplotlib.pyplot as plt


titanic = pd.read_csv('train.csv')

cluster_data = titanic[['Fare','Age']].copy(deep=True)
cluster_data.dropna(axis=0, inplace=True)
cluster_data.sort_values(by=['Fare','Age'], inplace=True)
cluster_array = np.array(cluster_data)

cluster_array

# Calculate Euclidean distance between two observations
def calc_distance(X1, X2):
    return (sum((X1 - X2)**2))**0.5

# Assign cluster clusters based on closest centroid
def assign_clusters(centroids, cluster_array):
    clusters = []
    for i in range(cluster_array.shape[0]):
        distances = []
        for centroid in centroids:
            distances.append(calc_distance(centroid, 
                                           cluster_array[i]))
        cluster = [z for z, val in enumerate(distances) if val==min(distances)]
        clusters.append(cluster[0])
    return clusters

# Calculate new centroids based on each cluster's mean
def calc_centroids(clusters, cluster_array):
    new_centroids = []
    cluster_df = pd.concat([pd.DataFrame(cluster_array),
                            pd.DataFrame(clusters, 
                                         columns=['cluster'])], 
                           axis=1)
    for c in set(cluster_df['cluster']):
        current_cluster = cluster_df[cluster_df['cluster']\
                                     ==c][cluster_df.columns[:-1]]
        cluster_mean = current_cluster.mean(axis=0)
        new_centroids.append(cluster_mean)
    return new_centroids

# Calculate variance within each cluster
def calc_centroid_variance(clusters, cluster_array):
    sum_squares = []
    cluster_df = pd.concat([pd.DataFrame(cluster_array),
                            pd.DataFrame(clusters, 
                                         columns=['cluster'])], 
                           axis=1)
    for c in set(cluster_df['cluster']):
        current_cluster = cluster_df[cluster_df['cluster']\
                                     ==c][cluster_df.columns[:-1]]
        cluster_mean = current_cluster.mean(axis=0)
        mean_repmat = np.matlib.repmat(cluster_mean, 
                                       current_cluster.shape[0],1)
        sum_squares.append(np.sum(np.sum((current_cluster - mean_repmat)**2)))
    return sum_squares



def k_means(k):

    cluster_vars = []

    centroids = [cluster_array[i+2] for i in range(k)]
    clusters = assign_clusters(centroids, cluster_array)
    initial_clusters = clusters
    print(0, round(np.mean(calc_centroid_variance(clusters, cluster_array))))

    for i in range(20):
        centroids = calc_centroids(clusters, cluster_array)
        clusters = assign_clusters(centroids, cluster_array)
        cluster_var = np.mean(calc_centroid_variance(clusters, 
                                                    cluster_array))
        cluster_vars.append(cluster_var)
        print(i+1, round(cluster_var))

    # Visualização
    plt.subplots(figsize=(9,6))
    plt.scatter(x=cluster_array[:,0], y=cluster_array[:,1], 
                c=initial_clusters, cmap=plt.cm.Spectral);
    plt.xlabel('Passenger Fare')
    plt.ylabel('Passenger Age');
    plt.savefig('initial_clusters', bpi=150)

    plt.subplots(figsize=(9,6))
    plt.scatter(x=cluster_array[:,0], y=cluster_array[:,1], 
                c=clusters, cmap=plt.cm.Spectral);
    plt.xlabel('Passenger Fare')
    plt.ylabel('Passenger Age');
    plt.savefig('final_clusters', bpi=150)
   
k_means(4)
