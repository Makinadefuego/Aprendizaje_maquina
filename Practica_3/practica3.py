#Se importan las librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Se lee el archivo iris.data
iris = pd.read_csv('iris.data', header=None)

#Se asignan los nombres de las columnas
iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']



#Se normalizan los datos para que tengan media 0 y varianza 1

#Se separan los datos de las etiquetas
X = iris.iloc[:,0:4].values
y = iris.iloc[:,4].values

#Se normalizan los datos
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

#Se calcula la varianza de cada característica
#Se calcula la matriz de covarianza
cov_mat = np.cov(X_std.T)





#Se aplica PCA a los datos normalizados
from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=4)
Y_sklearn = sklearn_pca.fit_transform(X_std)

#Se determina el número de componentes principales a retener
#Se visualizan los valores  propios
eig_vals = sklearn_pca.explained_variance_
print('Valores propios \n%s' %eig_vals)


#La varianza total es la suma de las varianzas de cada componente original
total_var = np.sum(eig_vals)
print('Varianza total \n%s' %total_var)

#Se calcula la varianza acumulada
var_acum = np.cumsum(eig_vals)/total_var

#Se grafica la varianza acumulada
plt.plot(range(1,5), var_acum)
plt.xlabel('Número de componentes principales')
plt.ylabel('Varianza acumulada')
plt.title('Varianza acumulada en PCA del conjunto de datos Iris')
plt.grid(True)
plt.show()


#Se visualizan los datos en 2 dimensiones
pd.plotting.scatter_matrix(iris, figsize=(10,10))
plt.show()

# Define a mapping of labels to colors
label_to_color = {
    'Iris-setosa': 'red',
    'Iris-versicolor': 'blue',
    'Iris-virginica': 'green'
}

# Assuming 'y' contains the labels for your data points
colors = [label_to_color[label] for label in y]

# Now you can create the scatter plot
plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], c=colors)
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.title('Visualización de los datos en 2 dimensiones')
plt.show()



