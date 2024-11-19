import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Cargar el dataset
data = pd.read_csv('Mall_Customers.csv')  # Asegúrate de tener este archivo en la misma carpeta que este script

# Seleccionar las características relevantes
features = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Estandarizar los datos
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Método del codo para encontrar el número óptimo de clusters
inertia = []
cluster_range = range(1, 11)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Graficar el método del codo
plt.figure(figsize=(8, 5))
plt.plot(cluster_range, inertia, marker='o')
plt.title('Método del Codo para K-Means')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inertia')
plt.show()

# Aplicar K-Means con el número óptimo de clusters (por ejemplo, k=5)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_features)

# Añadir la etiqueta del cluster al dataset original
data['Cluster'] = clusters

# Visualización de los clusters
plt.figure(figsize=(8, 6))
for cluster in range(5):
    cluster_data = data[data['Cluster'] == cluster]
    plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'], label=f'Cluster {cluster}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='black', marker='X', label='Centroides')
plt.title('Clustering K-Means')
plt.xlabel('Ingreso Anual (k$)')
plt.ylabel('Puntuación de Gasto (1-100)')
plt.legend()
plt.show()
