import numpy as np
import math
import matplotlib.pyplot as plt

class Dbscan_classifier:
    def __init__(self, data, epsilon, min_points):
        self.data = data
        self.epsilon = epsilon
        self.min_points = min_points
        self.clusters = []
        self.centroids = []

    def fit(self):
        self.clusters = [-1] * len(self.data)  # Initialize all points as unclassified (-1)
        self.centroids = []

        for i, data_point in enumerate(self.data):
            if self.clusters[i] != -1:
                continue

            neighbors = self.get_neighbors(i)

            #Se imprimen el número de vecinos encontrados para cada punto
            print(f"Vecinos encontrados para el punto {i}: {len(neighbors)}")

            if len(neighbors) < self.min_points:
                self.clusters[i] = 0  # Mark as noise
            else:
                self.expand_cluster(i, neighbors)

        # Remove noise points (-1) and calculate centroids
        self.clusters = [cluster for cluster in self.clusters if cluster != -1]
        self.centroids = [self.calculate_centroid(cluster) for cluster in self.clusters]

        # Imprime el número de clusters
        num_clusters = len(np.unique(self.clusters))
        print(f"Número de clusters encontrados: {num_clusters}")

    def calculate_centroid(self, cluster):
        # Verificar si el cluster no es ruido (-1)
        if cluster != -1:
            # Calcular el centroide solo para puntos que pertenecen al cluster
            centroid = np.mean([self.data[i] for i, c in enumerate(self.clusters) if c == cluster], axis=0)
            return centroid.tolist()
        else:
            return None  # Puedes manejar los puntos de ruido de la forma que desees


    def euclidean_distance(self, data_point1, data_point2):
        return np.linalg.norm(data_point1 - data_point2)

    def get_neighbors(self, index):
        neighbors = []
        for i, data_point in enumerate(self.data):
            if self.euclidean_distance(self.data[index], data_point) < self.epsilon and i != index:
                neighbors.append(i)
        return neighbors

    def expand_cluster(self, index, neighbors):
        cluster_id = len(self.clusters)
        self.clusters[index] = cluster_id

        i = 0
        while i < len(neighbors):
            neighbor = neighbors[i]
            if self.clusters[neighbor] == -1:
                self.clusters[neighbor] = cluster_id
                new_neighbors = self.get_neighbors(neighbor)
                if len(new_neighbors) >= self.min_points:
                    neighbors.extend(new_neighbors)
            i += 1

    def visualize_clusters(self):
        #Se visualizan los datos en 2D y en 3D

        # Visualizamos los datos en 2D
        self.visualize_2d_clusters()

        # Visualizamos los datos en 3D
        self.visualize_3d_clusters()



    def visualize_2d_clusters(self):
        unique_clusters = np.unique(self.clusters)
        n_clusters = len(unique_clusters)

        colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
        
        for i, cluster_id in enumerate(unique_clusters):
            cluster_points = self.data[self.clusters == cluster_id]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[colors[i]], label=f'Cluster {cluster_id}')

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()

    def visualize_3d_clusters(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        unique_clusters = np.unique(self.clusters)
        n_clusters = len(unique_clusters)

        colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
        
        for i, cluster_id in enumerate(unique_clusters):
            cluster_points = self.data[self.clusters == cluster_id]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], c=[colors[i]], label=f'Cluster {cluster_id}')

        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        plt.legend()
        plt.show()
