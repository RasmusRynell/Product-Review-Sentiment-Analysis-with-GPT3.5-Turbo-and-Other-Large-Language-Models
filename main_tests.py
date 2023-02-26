import numpy as np
import sklearn
import matplotlib.pyplot as plt
from transformers import pipeline
import torch
import time

from read_data import read_clean_data
from test_models import test_naive_bayes, test_simple_sentiment, test_my_model

if __name__ == '__main__':
    cleaned_data = read_clean_data()

    # Randomly split the data into train and test
    seed = 42
    
    train = cleaned_data.sample(frac=0.8, random_state=seed)
    test = cleaned_data.drop(train.index)

    # Naive Bayes
    #test_naive_bayes(train, test)

    # bert
    #test_simple_sentiment(test)
    #test_simple_sentiment(cleaned_data)

    # my model
    test_my_model(test, "models/my/distilbert-base-uncased_done")
    