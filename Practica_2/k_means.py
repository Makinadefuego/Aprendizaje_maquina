import numpy as np
import matplotlib.pyplot as plt
import random
import math





class Kmeans_classifier:
    def __init__(self, data, classes, k):
        self.data = data
        self.classes = classes
        self.k = k
        self.centroids = None
        self.clusters = None

    def initialize_centroids(self):
        # Inicializar los centroides de manera aleatoria con puntos generados
        # aleatoriamente dentro del rango de valores de cada característica
        self.centroids = []
        for i in range(self.k):
            centroid = []
            for j in range(len(self.data[0])):
                centroid.append(random.uniform(min(self.data[:, j]), max(self.data[:, j])))
            self.centroids.append(centroid)
        

    def assign_to_clusters(self):
        # Asignar cada punto de datos al clúster más cercano
        self.clusters = [[] for _ in range(self.k)]
        
        for data_point in self.data:
            min_distance = float('inf')
            closest_cluster = None
            
            for i, centroid in enumerate(self.centroids):
                distance = self.euclidean_distance(data_point, centroid)
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster = i
            
            self.clusters[closest_cluster].append(data_point)

    def update_centroids(self):
        # Calcular nuevos centroides basados en los puntos asignados a cada clúster
        new_centroids = []
        for cluster in self.clusters:
            if cluster:
                new_centroid = [sum(x) / len(cluster) for x in zip(*cluster)]
                new_centroids.append(new_centroid)
            else:
                # Si un clúster está vacío, el centroide permanece igual
                new_centroids.append(self.centroids[len(new_centroids)])
        
        self.centroids = new_centroids

    def fit(self, max_iterations=100):
        # Implementar el algoritmo k-means: inicialización, asignación y actualización
        self.initialize_centroids()
        
        for _ in range(max_iterations):
            self.assign_to_clusters()
            old_centroids = self.centroids[:]
            self.update_centroids()
            
            # Comprobar si los centroides han convergido
            if old_centroids == self.centroids:
                break

    def predict(self, new_data):
        # Predecir el clúster al que pertenecen nuevos datos
        predictions = []
        for data_point in new_data:
            min_distance = float('inf')
            closest_cluster = None
            
            for i, centroid in enumerate(self.centroids):
                distance = self.euclidean_distance(data_point, centroid)
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster = i
            
            predictions.append(closest_cluster)
        
        return predictions

    # def evaluate(self):
    #     # Evaluar el rendimiento del modelo (puedes implementar métricas como la suma de cuadrados intra-cluster)
    #     pass

    def visualize_clusters(self):
        # Visualizar los clústeres y los centroides (puedes usar gráficos de dispersión)
        for i, cluster in enumerate(self.clusters):
            cluster = np.array(cluster)
            plt.scatter(cluster[:, 0], cluster[:, 1], c='C' + str(i))

        centroids = np.array(self.centroids)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x')
        plt.xlabel('num_reactions')
        plt.ylabel('num_comments')
        plt.show()

        #también se plotea en 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, cluster in enumerate(self.clusters):
            cluster = np.array(cluster)
            ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], c='C' + str(i))

        centroids = np.array(self.centroids)
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='black', marker='x')
        ax.set_xlabel('num_reactions')
        ax.set_ylabel('num_comments')
        ax.set_zlabel('num_shares')
        ax.view_init(elev=10, azim=20)
        plt.show()

    @staticmethod
    def euclidean_distance(point1, point2):
        # Calcular la distancia euclidiana entre dos puntos
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))
