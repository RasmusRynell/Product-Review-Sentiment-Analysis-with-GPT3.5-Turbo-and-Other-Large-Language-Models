import kaggle
import numpy as np
import os
import pandas as pd
import torch
import re
from sklearn.model_selection import train_test_split


import food_baseline
import food_LLM

# Read in data from data/food/zomato_reviews.csv
df = pd.read_csv('data/food/tripadvisor_hotel_reviews.csv',usecols=['Review', 'Rating'], encoding='utf-8')

# Rename columns Review and Rating to review and rating
df.columns = ['review', 'rating']

# Clean data
# Remove empty reviews
df = df[df['review'].notna()]
# Remove emojis
df['review'] = df['review'].apply(lambda x: re.sub(r'[^\w\s]','',x))

# Remove text after 500 words
df['review'] = df['review'].apply(lambda x: ' '.join(x.split()[:200]))

# Print the average length of reviews
print(f"Average length of reviews: {np.mean(df['review'].apply(lambda x: len(x.split())))}", flush=True)

# Preprocess data
# Set rating to 0-4 instead of 1-5
df['rating'] = df['rating'] - 1

# Divide data into train, validation and test sets
train_df = df.sample(frac=0.8, random_state=0)
test_df = df.drop(train_df.index)

# Divide train set into train and validation sets
train_df, validation_df = train_test_split(train_df, test_size=0.2, random_state=0)

print(f"Train set size: {len(train_df)}", flush=True)
print(f"Validation set size: {len(validation_df)}", flush=True)
print(f"Test set size: {len(test_df)}", flush=True)

# Run on baseline
#food_baseline.run(train_df, validation_df, test_df)

# Run LLM
food_LLM.run(train_df, validation_df, test_df)