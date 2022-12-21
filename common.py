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
        'type': 'classification',
        'link': 'https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews'
        },
        {
        'name': 'Fake_News',
        'address': 'hassanamin/textdb3',
        'file': 'fake_or_real_news.csv',
        'x_column': 'text',
        'y_column': 'label',
        'type': 'classification',
        'link': 'https://www.kaggle.com/hassanamin/textdb3'
        },
        {
        'name': 'Disneyland_Reviews',
        'address': 'arushchillar/disneyland-reviews',
        'file': 'disneyland_reviews.csv',
        'x_column': 'review',
        'y_column': 'sentiment',
        'type': 'multi-classification',
        'link': 'https://www.kaggle.com/arushchillar/disneyland-reviews'
        }
        ]


def clean_up_dataset(df, df_col_x, df_col_y, drop_zeros=False):
    # Drop rows with missing values in the df_col_x or df_col_y columns
    df = df.dropna(subset=[df_col_x, df_col_y])
    # Drop rows with zero values in the df_col_y column
    if drop_zeros:
        df = df[df[df_col_y] != 0]

    # Convert the df_col_y column to float
    df[df_col_y] = df[df_col_y].astype(float)

    # Convert to lowercase
    df[df_col_x] = df[df_col_x].str.lower()

    return df[[df_col_x, df_col_y]]

def split_data(df, test_size, validation_size, seed):
    train_size = 1 - test_size - validation_size

    # Split the data into train test and validation sets
    train_df, test_val_df = train_test_split(df, train_size=train_size, \
        random_state=seed)
    validation_df, test_df = train_test_split(test_val_df, \
        train_size=validation_size/(test_size+validation_size), random_state=seed)
    
    return df, train_df, validation_df, test_df


def get_dataset(idx):
    curr_data = data[idx]
    print(f"Using {curr_data['name']} dataset", flush=True)

    df = pd.read_csv(f'data/{curr_data["name"]}/{curr_data["file"]}', encoding='utf8')
    #df = clean_up_dataset(df, curr_data['x_column'], curr_data['y_column'], drop_zeros=True)
    df = df.rename(columns={curr_data['x_column']: 'x', curr_data['y_column']: 'y'})
    df = df.sample(frac=1, random_state=seed)

    test_size = 0.1
    validation_size = 0.1

    return curr_data['name'], split_data(df, test_size, validation_size, seed)

def save_results():
    pass