import numpy as np
import sklearn
import matplotlib.pyplot as plt
from transformers import pipeline
import torch
import time

from data_processing import *
from test_models import *
from data_analysis import *

if __name__ == '__main__':
    data = read_clean_data()

    train, test = split_data(data, random_state=42, validation=False, over_sample_train=False)

    # # Naive Bayes
    find_optimal_parameters(train, test) # Default parameters best...
    # test_naive_bayes(train, test)
    # test_optimized_naive_bayes(train, test) # No need to optimize, default parameters are best

    # # bert
    #test_simple_sentiment(test)

    # # my model
    # test_my_model(test, "models/my/distilbert-base-uncased_done")
        