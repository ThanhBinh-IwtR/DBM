# Import necessary libraries
from collections import Counter
import math
import pandas as pd

# Load the text dataset from a CSV file
data = pd.read_csv("product_reviews.csv")

# Extract review text from the DataFrame
reviews = data["ReviewText"].tolist()

# Tokenize the text into words
def tokenize(text):
    return text.lower().split()

# Calculate the PMI for all pairs of words
def calculate_pmi(texts):
    word_count = Counter()
    pair_count = Counter()
    
    for text in texts:
        words = tokenize(text)
        word_count.update(words)
        pairs = set()
        
        for i in range(len(words) - 1):
            for j in range(i + 1, len(words)):
                pair = (words[i], words[j])
                pairs.add(pair)
        
        pair_count.update(pairs)
    
    pmi = {}
    
    for pair, count in pair_count.items():
        word1, word2 = pair
        pmi[pair] = math.log((count * len(texts)) / (word_count[word1] * word_count[word2]))
    
    return pmi

# Extract meaningful quality phrases based on PMI
def extract_quality_phrases(pmi, threshold=1.0):
    quality_phrases = []
    
    for pair, score in pmi.items():
        if score >= threshold:
            quality_phrases.append(" ".join(pair))
    
    return quality_phrases

# Calculate PMI and extract quality phrases
pmi_scores = calculate_pmi(reviews)
threshold = 1.0  # Adjust the threshold as needed
quality_phrases = extract_quality_phrases(pmi_scores, threshold)

# Print quality phrases
print("Quality Phrases:")
for phrase in quality_phrases:
    print(phrase)
