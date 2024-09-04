# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Load the sample dataset from a CSV file
data = pd.read_csv("D:/FPTMaterial/ky 7/DBM/lab4/numerical_dataset.csv")

# Select the features for hierarchical clustering
X = data[['Feature1', 'Feature2']]

# Apply hierarchical clustering with a chosen linkage method
Z = linkage(X, method='ward')

# Cut the dendrogram to create clusters and add labels
num_clusters = 3  # Adjust the number of clusters as needed
data['Hierarchical_Cluster'] = fcluster(Z, t=num_clusters, criterion='maxclust')

# Visualize the hierarchical clustering results with customized labels
sns.scatterplot(x="Feature1", y="Feature2", hue="Hierarchical_Cluster", data=data, palette='Set1')

# Create custom labels for the clusters
custom_labels = {1: 'Cluster A', 2: 'Cluster B', 3: 'Cluster C'}
data['Cluster_Label'] = data['Hierarchical_Cluster'].map(custom_labels)

plt.title("Hierarchical Clustering")
plt.xlabel("Feature1")
plt.ylabel("Feature2")

# Print the custom cluster labels
for label, cluster in custom_labels.items():
    print(f"Cluster {label}: {cluster}")

# Show the scatter plot with custom cluster labels
plt.legend(title="Custom Labels")
plt.show()

