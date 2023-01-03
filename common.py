import pandas as pd
import kaggle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import random
from datasets import load_dataset

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
            'link': 'https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews'
        },
        {
            'name': 'Twitter',
            'address': 'kazanova/sentiment140',
            'file': 'training.1600000.processed.noemoticon.csv',
            'columns': ['target', 'id', 'date', 'flag', 'user', 'text'],
            'x_column': 'text',
            'y_column': 'target',
            'link': 'https://www.kaggle.com/kazanova/sentiment140'
        },
        {
            'name': 'Amazon',
            'address': 'snap/amazon-fine-food-reviews',
            'file': 'Reviews.csv',
            'x_column': 'Text',
            'y_column': 'Score',
            'link': 'https://www.kaggle.com/snap/amazon-fine-food-reviews'
        },
        {
            'name': 'Womens E-Commerce Clothing Reviews',
            'address': 'nicapotato/womens-ecommerce-clothing-reviews',
            'file': 'Womens Clothing E-Commerce Reviews.csv',
            'x_column': 'Review Text',
            'y_column': 'Rating',
            'link': 'https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews'
        },
        {
            'name': 'Financial Sentiment Analysis',
            'address': 'sbhatti/financial-sentiment-analysis',
            'file': 'data.csv',
            'x_column': 'Sentence',
            'y_column': 'Sentiment',
            'link': 'https://www.kaggle.com/sbhatti/financial-sentiment-analysis'
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

def clean_up_dataset(df, x_column, y_column, should_be_lower=False):
    # Drop rows with NaN values
    df = df.dropna()

    if should_be_lower:
        df[x_column] = df[x_column].str.lower()

    # For each row check if x column is larger than 512, if it is then truncate it
    df[x_column] = df[x_column].apply(lambda x: x[:512] if len(x) > 512 else x)

    df.loc[:, x_column] = df[x_column].apply(lambda x: x[:512] if len(x) > 512 else x)


    # Rename positive to 2 neutral to 1 and negative to 0
    df[y_column] = df[y_column].replace('positive', 2)
    df[y_column] = df[y_column].replace('neutral', 1)
    df[y_column] = df[y_column].replace('negative', 0)
    
    return df



def get_dataset(idx, should_be_lower=False):
    curr_data = data[idx]
    print(f"Using {curr_data['name']} dataset", flush=True)

    try:
        df = pd.read_csv(f'data/{curr_data["name"]}/{curr_data["file"]}', encoding='utf8')
    except:
        df = pd.read_csv(f'data/{curr_data["name"]}/{curr_data["file"]}', encoding='latin-1')

    # Check if curr_data['x_column'] and curr_data['y_column'] is in the dataset
    if curr_data['x_column'] not in df.columns or curr_data['y_column'] not in df.columns:
        # Set columns
        df.columns = curr_data['columns']

    df = clean_up_dataset(df, curr_data['x_column'], curr_data['y_column'], should_be_lower=should_be_lower)

    df = df.rename(columns={curr_data['x_column']: 'x', curr_data['y_column']: 'y'})
    

    df = df.sample(frac=1, random_state=seed)

    test_size = 0.1
    validation_size = 0.1

    return curr_data['name'], split_data(df, test_size, validation_size, seed)