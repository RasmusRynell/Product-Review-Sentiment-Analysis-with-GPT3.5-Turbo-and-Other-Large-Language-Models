import numpy as np
import sklearn
import matplotlib.pyplot as plt
from transformers import pipeline
import torch
import time

from data import read_clean_data
from baselines import test_naive_bayes, test_svm, test_simple_sentiment

if __name__ == '__main__':
    cleaned_data = read_clean_data()

    # Set seed
    np.random.seed(0)

    # Split data into train, test, validation
    train, test, validation = np.split(cleaned_data.sample(frac=1), [int(.6*len(cleaned_data)), int(.8*len(cleaned_data))])

    test = test[:100]

    # Naive Bayes
    test_naive_bayes(train, test, validation)

    # SVM
    #test_svm(train, test, validation)

    # bert
    test_simple_sentiment(train, test, validation)