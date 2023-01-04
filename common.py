import pandas as pd
import kaggle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import random
from datasets import load_dataset
import transformers

print(f"Path to models {transformers.__path__}", flush=True)

seed = 41
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


data = [{
            'name': 'IMDB',
            'address': 'lakshmi25npathi/imdb-dataset-of-50k-movie-reviews',
            'file': 'IMDB_Dataset.csv',
            'x_column': 'review',
            'y_column': 'sentiment',
            'link': 'https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews',
            'encoding': 'utf-8'
        },
        {
            'name': 'News Headlines Dataset For Sarcasm Detection',
            'address': 'rmisra/news-headlines-dataset-for-sarcasm-detection',
            'file': 'Sarcasm_Headlines_Dataset_v2.json',
            'x_column': 'headline',
            'y_column': 'is_sarcastic',
            'link': 'https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection',
            'encoding': 'latin-1'
        },
        {
            'name': 'Amazon',
            'address': 'snap/amazon-fine-food-reviews',
            'file': 'Reviews.csv',
            'x_column': 'Text',
            'y_column': 'Score',
            'link': 'https://www.kaggle.com/snap/amazon-fine-food-reviews',
            'encoding': 'utf-8'
        },
        {
            'name': 'Womens E-Commerce Clothing Reviews',
            'address': 'nicapotato/womens-ecommerce-clothing-reviews',
            'file': 'Womens Clothing E-Commerce Reviews.csv',
            'x_column': 'Review Text',
            'y_column': 'Rating',
            'link': 'https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews',
            'encoding': 'utf-8'
        },
        {
            'name': 'Financial Sentiment Analysis',
            'address': 'sbhatti/financial-sentiment-analysis',
            'file': 'data.csv',
            'x_column': 'Sentence',
            'y_column': 'Sentiment',
            'link': 'https://www.kaggle.com/sbhatti/financial-sentiment-analysis',
            'encoding': 'utf-8'
        }
        ]

def download_data():
    for curr_data in data:
        if not os.path.exists(f'data/{curr_data["name"]}'):
            os.makedirs(f'data/{curr_data["name"]}')
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(curr_data['address'], path=f'data/{curr_data["name"]}', unzip=True)
            print(f"Downloaded {curr_data['name']} dataset", flush=True)
        #else:
        #    print(f"Dataset {curr_data['name']} already exists", flush=True)

download_data()

def split_data(df, test_size, validation_size, seed):
    train_size = 1 - test_size - validation_size

    # Split the data into train test and validation sets
    train_df, test_val_df = train_test_split(df, train_size=train_size, \
        random_state=seed)
    validation_df, test_df = train_test_split(test_val_df, \
        train_size=validation_size/(test_size+validation_size), random_state=seed)
    
    return df, train_df, validation_df, test_df

def clean_up_dataset(df, should_be_lower=False):
    # Remove every column except x and y
    df = df[['x', 'y']]

    # Drop rows with NaN values
    df = df.dropna()

    if should_be_lower:
        df['x'] = df['x'].str.lower()

    # For each row check if x column is larger than 512, if it is then truncate it
    df.loc[df['x'].apply(len) > 512, 'x'] = df['x'].apply(lambda x: x[:512])

    #print(df.head(10))

    # If Neutral is not in, only change positive to 1 and negative to 0
    if 'neutral' not in df['y'].unique():
        df.loc[df['y'] == 'positive', 'y'] = 1
        df.loc[df['y'] == 'negative', 'y'] = 0
    else:
        # Neutral is in, so change positive to 2, neutral to 1 and negative to 0
        df.loc[df['y'] == 'positive', 'y'] = 2
        df.loc[df['y'] == 'neutral', 'y'] = 1
        df.loc[df['y'] == 'negative', 'y'] = 0

    # Set y column to character
    df['y'] = df['y'].astype(int)

    # If there is a 1 but not a 0, lower all by 1
    if 1 in df['y'].unique() and 0 not in df['y'].unique():
        df['y'] = df['y'] - 1
    
    
    return df



def get_dataset(idx, should_be_lower=False):
    curr_data = data[idx]
    print(f"Using {curr_data['name']} dataset", flush=True)

    try:
        df = pd.read_csv(f'data/{curr_data["name"]}/{curr_data["file"]}', encoding=curr_data['encoding'])
    except:
        df = pd.read_json(f'data/{curr_data["name"]}/{curr_data["file"]}', lines=True)

    # Check if curr_data['x_column'] and curr_data['y_column'] is in the dataset
    if curr_data['x_column'] not in df.columns or curr_data['y_column'] not in df.columns:
        # Set columns
        df.columns = curr_data['columns']

    # Rename
    df = df.rename(columns={curr_data['x_column']: 'x', curr_data['y_column']: 'y'})

    df = clean_up_dataset(df, should_be_lower=should_be_lower)

    

    df = df.sample(frac=1, random_state=seed)

    test_size = 0.1
    validation_size = 0.1

    return curr_data['name'], split_data(df, test_size, validation_size, seed)