# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np

# Load the sample spatial dataset from a CSV file
data = pd.read_csv("D:/FPTMaterial/ky 7/DBM/lab4/spatial_dataset.csv")

# Select the spatial features (X, Y)
X = data[['X', 'Y']]

# Apply DBSCAN with chosen epsilon and minimum points
epsilon = 1.0  # Adjust the epsilon as needed
min_samples = 3  # Adjust the minimum points as needed
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
data['DBSCAN_Cluster'] = dbscan.fit_predict(X)

# Visualize the DBSCAN clustering results
colors = np.array(['red', 'green', 'blue', 'purple', 'orange'])
plt.scatter(data['X'], data['Y'], c=colors[data['DBSCAN_Cluster']], s=50)
plt.title("DBSCAN Clustering")
plt.xlabel("X")
plt.ylabel("Y")

# Plot noise points as black crosses
noise_points = data[data['DBSCAN_Cluster'] == -1]
plt.scatter(noise_points['X'], noise_points['Y'], marker='x', color='black', s=50, label='Noise')

plt.legend()
plt.show()

# Print cluster assignments
print("Cluster Assignments:")
print(data[['X', 'Y', 'DBSCAN_Cluster']])

