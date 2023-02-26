import numpy as np
import sklearn
import matplotlib.pyplot as plt
from transformers import pipeline
import torch
import time

from read_data import read_clean_data
from baselines import test_naive_bayes, test_simple_sentiment, test_my_model

if __name__ == '__main__':
    cleaned_data = read_clean_data()

    # Randomly split the data into train and test
    seed = 42
    np.random.seed(seed)
    np.random.shuffle(cleaned_data)
    train = cleaned_data[:int(len(cleaned_data) * 0.8)]
    test = cleaned_data[int(len(cleaned_data) * 0.8):]

    # Naive Bayes
    test_naive_bayes(train, test)

    # bert
    test_simple_sentiment(train, test)

    # my model
    test_my_model(train, test)