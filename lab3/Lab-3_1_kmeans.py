# Import necessary libraries
import pandas as pd

# Import necessary libraries
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the sample dataset (replace with your dataset file)
df = pd.read_csv("example_features_dataset.csv")

# Data preprocessing (cleaning, missing value handling, feature selection)
# Example: Removing rows with missing values
df = df.dropna()

# Assuming your dataset is already preprocessed and features are selected
# Assuming you have selected two features "Feature1" and "Feature2"
data = df[["Feature1", "Feature2"]]

# Apply K-Means clustering to the data
kmeans = KMeans(n_clusters=3)  # Specify the number of clusters
kmeans.fit(data)
df["Cluster"] = kmeans.labels_  # Assign clusters to data points

# Visualize the clusters (optional)
plt.scatter(data["Feature1"], data["Feature2"], c=df["Cluster"], cmap='viridis')
plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.show()



