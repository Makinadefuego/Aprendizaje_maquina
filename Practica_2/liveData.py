import numpy as np
import matplotlib.pyplot as plt
    

class LiveData:
    def __init__(self, path):
        self.path = path
        self.data = self.read()
        self.classes = None

    def read(self):
        # Leemos los datos del archivo CSV excluyendo las columnas innecesarias
        data = np.genfromtxt(self.path, delimiter=',', dtype=str, skip_header=1, usecols=( 3, 4, 5))

        # Convertimos los datos a enteros, para el caso de status_type, lo convertimos a un valor numérico
        # 0: photo, 1: video, 2: link, 3: status
        classes = np.genfromtxt(self.path, delimiter=',', dtype=str, skip_header=1, usecols=( 1))
        classes[classes == 'photo'] = 0
        classes[classes == 'video'] = 1
        classes[classes == 'link'] = 2
        classes[classes == 'status'] = 3

        self.classes = classes.astype(int)
        return data
    
    def pemutar(self):
        # Se permutan los datos y las clases
        permutation = np.random.permutation(len(self.data))
        self.data = self.data[permutation]


    def clean(self):
        # Se eliminan las filas que tengan valores faltantes o sean 0 en alguna de las columnas
        self.data = self.data[(self.data != '0').all(axis=1) & (self.data != '').all(axis=1)]

        
        #Se convierte el array de strings a enteros
        self.data = self.data.astype(int)

    def visualizarDatos2d(self):
        # Visualizamos los datos en 2D

        # num_reactions,num_comments son las columnas que se van a visualizar
        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.classes)
        plt.xlabel('num_reactions')
        plt.ylabel('num_comments')
        plt.show()

        # num_reactions,num_shares son las columnas que se van a visualizar
        plt.scatter(self.data[:, 0], self.data[:, 2], c=self.classes)
        plt.xlabel('num_reactions')
        plt.ylabel('num_shares')
        plt.show()

        # num_comments,num_shares son las columnas que se van a visualizar
        plt.scatter(self.data[:, 1], self.data[:, 2], c=self.classes)
        plt.xlabel('num_comments')
        plt.ylabel('num_shares')
        plt.show()

    def visualizarDatos3d(self):
        # Visualizamos los datos en 3D

        # num_reactions,num_comments,num_shares son las columnas que se van a visualizar
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.data[:, 0], self.data[:, 1], self.data[:, 2], c=self.classes)
        ax.set_xlabel('num_reactions')
        ax.set_ylabel('num_comments')
        ax.set_zlabel('num_shares')
    

        #además se le hace un zoom out para que se vea mejor
        ax.view_init(elev=10, azim=-20)


        plt.show()

