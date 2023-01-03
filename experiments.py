from transformers import pipeline
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification
import numpy as np
import random
import matplotlib.pyplot as plt
import kaggle
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)


# This file contains tests for LLM models for different sentiment analysis datasets
# These models are english only, and a verity of small, medium and large models are tested
# They include:
# - bert-base-uncased, 110M parameters, english only, https://huggingface.co/bert-base-uncased
# - bert-large-uncased, 340M parameters, english only, https://huggingface.co/bert-large-uncased

# - RoBERTa-base, 125M parameters, english only, https://huggingface.co/roberta-base
# - RoBERTa-large, 355M parameters, english only, https://huggingface.co/roberta-large

# - DistilBert-base-uncased, 66M parameters, english only, https://huggingface.co/distilbert-base-uncased

# - albert-base-v1, 12M parameters, english only, https://huggingface.co/albert-base-v1
# - albert-large-v1, 18M parameters, english only, https://huggingface.co/albert-large-v1
# - albert-xlarge-v1, 60M parameters, english only, https://huggingface.co/albert-xlarge-v1
# - albert-xxlarge-v1, 235M parameters, english only, https://huggingface.co/albert-xxlarge-v1

# - albert-base-v2, 12M parameters, english only, https://huggingface.co/albert-base-v2
# - albert-large-v2, 18M parameters, english only, https://huggingface.co/albert-large-v2
# - albert-xlarge-v2, 60M parameters, english only, https://huggingface.co/albert-xlarge-v2
# - albert-xxlarge-v2, 235M parameters, english only, https://huggingface.co/albert-xxlarge-v2

# - xlnet-base-cased, 110M parameters, english only, https://huggingface.co/xlnet-base-cased
# - xlnet-large-cased, 340M parameters, english only, https://huggingface.co/xlnet-large-cased

# Fine-tuned models:
# - finbert, 110M parameters, english only, https://huggingface.co/ProsusAI/finbert
# - bertweet-base, 110M parameters, english only, https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment
# - bertweet-large, 340M parameters, english only, https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment

non_fine_tuned_models = [
    "bert-base-uncased",
    "bert-large-uncased",
    "roberta-base",
    "roberta-large",
    "distilbert-base-uncased",
    "albert-base-v1",
    "albert-large-v1",
    "albert-xlarge-v1",
    "albert-xxlarge-v1",
    "albert-base-v2",
    "albert-large-v2",
    "albert-xlarge-v2",
    "albert-xxlarge-v2",
    "xlnet-base-cased",
    "xlnet-large-cased",
]



import common
idx = 0

# For each dataset
for idx in range(0, 1): # 3
    dataset_name, others = common.get_dataset(idx)
    df, train_df, validation_df, test_df = others

    # Reset all indexes
    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Print how long the train, validation and test sets are
    print(f'Train set length: {len(train_df)}', flush=True)
    print(f'Validation set length: {len(validation_df)}', flush=True)
    print(f'Test set length: {len(test_df)}', flush=True)

    # Print how many unique values there are in y_true
    print(f'Unique values in y_true: {len(train_df["y"].unique())}', flush=True)
    # Print how many of each unique value there are in y_true
    print(f'Unique values in y_true: {train_df["y"].value_counts().to_dict()}', flush=True)

    # Loop trough all different models, train and test them
    for model_name in non_fine_tuned_models:
        # Train the model
