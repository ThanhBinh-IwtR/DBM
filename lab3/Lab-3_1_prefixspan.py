# Import necessary libraries
import pandas as pd

# Import necessary libraries
from prefixspan import PrefixSpan

# Load the sample dataset (replace with your dataset file)
df = pd.read_csv("example_sequenceid_dataset.csv")

# Data preprocessing (cleaning, missing value handling, feature selection)
# Example: Removing rows with missing values
df = df.dropna()

# Convert data to a list of sequences
# Assuming your dataset has a column "Actions" with lists of items in each transaction
sequences = df["Actions"].tolist()

# Apply PrefixSpan algorithm to discover sequential patterns
ps = PrefixSpan(sequences)
patterns = ps.frequent(2)  # Discover sequences occurring at least twice

# Print sequential patterns
for pattern in patterns:
    print("Pattern:", pattern)

