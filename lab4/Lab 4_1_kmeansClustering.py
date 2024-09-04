# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the sample dataset from a CSV file
data = pd.read_csv("D:/FPTMaterial/ky 7/DBM/lab4/numerical_dataset.csv")

# Select the features for clustering
X = data[['Feature1', 'Feature2']]

# Choose the number of clusters (you can adjust this based on the dataset)
num_clusters = 3

# Apply K-Means clustering
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(X)

# Get cluster labels and add them to the DataFrame
data['Cluster'] = kmeans.labels_

# Visualize the clustered data
plt.scatter(data['Feature1'], data['Feature2'], c=data['Cluster'], cmap='rainbow')
plt.title("K-Means Clustering")
plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.show()

# Print cluster centers
cluster_centers = kmeans.cluster_centers_
print("Cluster Centers:")
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i+1}: Feature1={center[0]}, Feature2={center[1]}")
