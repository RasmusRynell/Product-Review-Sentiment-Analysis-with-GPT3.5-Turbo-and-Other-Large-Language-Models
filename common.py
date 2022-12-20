import pandas as pd
import kaggle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import random

seed = 41
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data = [{
        'name': 'movie',
        'address': 'akshaypawar7/millions-of-movies',
        'file': 'movies.csv',
        'x_column': 'overview',
        'y_column': 'budget'
        },
        {
        'name': 'musk',
        'address': 'marta99/elon-musks-tweets-dataset-2022',
        'file': 'cleandata.csv',
        'x_column': 'Cleaned_Tweets',
        'y_column': 'Likes'
        }]

try:
    kaggle.api.authenticate()
    for dataset in data:
        if not os.path.exists(f"data/{dataset['name']}") or \
            not os.path.exists(f"data/{dataset['name']}/{dataset['file']}"):
            print(f"Downloading {dataset['name']} dataset", flush=True)
            kaggle.api.dataset_download_files(dataset['address'], \
                path=f"data/{dataset['name']}", unzip=True)
except:
    print("Kaggle API not working, either download the data manually or use the Kaggle CLI")
    for dataset in data:
        print(f"Download {dataset['name']} dataset from https://www.kaggle.com/{dataset['address']} and place its content it in data/{dataset['name']} folder")


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
    
    return train_df, validation_df, test_df


def get_dataset(idx):
    curr_data = data[idx]
    print(f"Using {curr_data['name']} dataset")

    # Read data
    text_column = curr_data['x_column']
    ans_column = curr_data['y_column']
    df = pd.read_csv(f'data/{curr_data["name"]}/{curr_data["file"]}', encoding='utf8')
    df = clean_up_dataset(df, text_column, ans_column, drop_zeros=True)

    # Rename columns
    df = df.rename(columns={text_column: 'x', ans_column: 'y'})

    # Shuffle the data
    df = df.sample(frac=1, random_state=seed)


    train_df, validation_df, test_df = split_data(df, test_size=0.2, validation_size=0.2, seed=seed)
    print(f"Train set size: \t{len(train_df)} ({len(train_df)/len(df)*100:.2f}%)", flush=True)
    print(f"Validation set size: \t{len(validation_df)} ({len(validation_df)/len(df)*100:.2f}%)", flush=True)
    print(f"Test set size: \t\t{len(test_df)} ({len(test_df)/len(df)*100:.2f}%)", flush=True)

    return df, train_df, validation_df, test_df

def save_results():
    pass