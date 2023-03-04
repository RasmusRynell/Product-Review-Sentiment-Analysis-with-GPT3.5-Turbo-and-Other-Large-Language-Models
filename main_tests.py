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
    cleaned_data = read_clean_data()

    # Randomly split the data into train and test
    seed = 42

    #for seed in range(1, 5):

    train = cleaned_data.sample(frac=0.8, random_state=seed)
    test = cleaned_data.drop(train.index)

    #analyze_data(train, save=True, concat_string=f"_before_{seed}")
    
    #train = over_sample(train, seed)

    #analyze_data(train, save=True, concat_string=f"_after_{seed}")



    # Naive Bayes
    #find_optimal_parameters(train, test) # Default parameters best...
    test_naive_bayes(train, test)
    test_optimized_naive_bayes(train, test) # No need to optimize, default parameters are best

    # bert
    #test_simple_sentiment(test)
    #test_simple_sentiment(cleaned_data)

    # my model
    test_my_model(test, "models/my/distilbert-base-uncased_done")
    