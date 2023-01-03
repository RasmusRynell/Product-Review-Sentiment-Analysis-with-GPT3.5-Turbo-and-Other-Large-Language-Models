import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
import time

threads = 8

import common






def run_evaluate_and_save_pipeline(pipeline, train_df, validation_df, test_df, name):
    # Train the pipeline
    train_time_start = time.time()
    pipeline.fit(train_df['x'], train_df['y'])
    train_time_end = time.time()

    # Evaluate the pipeline
    va_score = pipeline.score(validation_df['x'], validation_df['y'])
    test_score = pipeline.score(test_df['x'], test_df['y'])
    print(f'{name} validation score: {va_score}', flush=True)
    print(f'{name} test score: {test_score}', flush=True)

    # More metrics
    y_true = test_df['y']
    y_pred = pipeline.predict(test_df['x'])
    cm = confusion_matrix(y_true, y_pred)
    print(f'{name} confusion matrix: \n{cm}', flush=True)


    print()

    # Save the results
    # common.save_results(name, pipeline, train_time_end - train_time_start, va_score, test_score, cm)



idx = 0
for idx in range(0, 5):
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



    # Baseline: MultinomialNB
    # Bag of words
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])
    run_evaluate_and_save_pipeline(pipeline, train_df, validation_df, test_df, f'{dataset_name}: BOW - MultinomialNB')

    # TF-IDF
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ])
    run_evaluate_and_save_pipeline(pipeline, train_df, validation_df, test_df, f'{dataset_name}: TF-IDF - MultinomialNB')