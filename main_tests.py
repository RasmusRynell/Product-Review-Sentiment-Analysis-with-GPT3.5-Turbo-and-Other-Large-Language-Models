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

    for over_sample in [False, True]:
        train, test = split_data(data, random_state=42, validation=False, over_sample_train=over_sample)

        test_naive_bayes(train, test, over_sample)

    # bert
    _, test = split_data(data, random_state=42)
    test_simple_sentiment(test)

    # my models
    test_my_model(test, "models/my/distilbert-base-uncased_no_over_sample_done")
    test_my_model(test, "models/my/distilbert-base-uncased_over_sample_done")