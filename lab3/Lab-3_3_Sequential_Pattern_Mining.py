# Import necessary libraries
from prefixspan import PrefixSpan
import pandas as pd

# Read the dataset from a CSV file
data = pd.read_csv("customer_transactions_dataset
.csv")

# Extract the sequence data and item labeling from the DataFrame
sequences = [list(map(str.strip, row["Sequence"].split(", ")) for _, row in data.iterrows()]
item_labels = list(set(item for seq in sequences for item in seq))

# Create a mapping of item labels to integers for efficient pattern representation
item_to_int = {item: i for i, item in enumerate(item_labels)}

# Convert the sequences to a list of integer sequences
sequences_int = [[item_to_int[item] for item in seq] for seq in sequences]

# Apply PrefixSpan algorithm to discover frequent sequential patterns
ps = PrefixSpan(sequences_int)

# Discover frequent sequential patterns (e.g., with a minimum support of 2)
min_support = 2
patterns = ps.frequent(min_support)

# Create a DataFrame to store the patterns and their support
pattern_data = {
    "Pattern": [", ".join(item_labels[i] for i in pattern[1]) for pattern in patterns],
    "Support": [pattern[0] for pattern in patterns],
    "Pattern Length": [len(pattern[1]) for pattern in patterns],
}

result_df = pd.DataFrame(pattern_data)

# Print the frequent sequential patterns and save them to a CSV file
print("Frequent Sequential Patterns:")
print(result_df)

# Save the results to a CSV file
result_df.to_csv("frequent_patterns.csv", index=False)
