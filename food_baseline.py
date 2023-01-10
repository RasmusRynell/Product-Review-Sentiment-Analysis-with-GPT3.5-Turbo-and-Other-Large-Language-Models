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
from sklearn.metrics import roc_auc_score
import time
from transformers import BertTokenizer
from sklearn.base import TransformerMixin


# Set random seed for reproducibility
random.seed(0)


class BertTokenizerTransformer(TransformerMixin):
    def __init__(self, name, max_length=512):
        self.tokenizer = BertTokenizer.from_pretrained(name)
        self.max_length = max_length

    def transform(self, X, **transform_params):
        return [self.tokenizer.encode(x, max_length=self.max_length, \
            truncation=True, padding='max_length', add_special_tokens=True) for x in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {'max_length':self.max_length}

    def set_params(self, **parameters):
        self.max_length = parameters['max_length']

# Things to do:
# 1. Save models after each epoch
# 2. Return best model (highest validation accuracy)



def run(train_df, validation_df, test_df):
    results = []

    # Random baseline
    pipeline = Pipeline([
        ('clf', DummyClassifier(strategy='uniform')),
    ])
    pipeline.fit(train_df['review'], train_df['rating'])
    va_score = pipeline.score(validation_df['review'], validation_df['rating'])
    test_score = pipeline.score(test_df['review'], test_df['rating'])
    print(f'Random baseline validation acc: {va_score}', flush=True)
    print(f'Random baseline test acc: {test_score}', flush=True)
    results.append(('Uniform', 'Random baseline', test_score))

    # Random baseline
    pipeline = Pipeline([
        ('clf', DummyClassifier(strategy='stratified')),
    ])
    pipeline.fit(train_df['review'], train_df['rating'])
    va_score = pipeline.score(validation_df['review'], validation_df['rating'])
    test_score = pipeline.score(test_df['review'], test_df['rating'])
    print(f'Stratified baseline validation acc: {va_score}', flush=True)
    print(f'Stratified baseline test acc: {test_score}', flush=True)
    results.append(('Stratified', 'Random baseline', test_score))

    # Baseline 1: CountVectorizer, Logistic regression
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', LogisticRegression(random_state=0, max_iter=1000)),
    ])
    pipeline.fit(train_df['review'], train_df['rating'])
    va_score = pipeline.score(validation_df['review'], validation_df['rating'])
    test_score = pipeline.score(test_df['review'], test_df['rating'])
    print(f'CountVectorizer, Logistic regression validation acc: {va_score}', flush=True)
    print(f'CountVectorizer, Logistic regression test acc: {test_score}', flush=True)
    results.append(('CountVectorizer', 'Logistic regression', test_score))

    # Baseline 2: CountVectorizer, Naive Bayes
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', MultinomialNB()),
    ])
    pipeline.fit(train_df['review'], train_df['rating'])
    va_score = pipeline.score(validation_df['review'], validation_df['rating'])
    test_score = pipeline.score(test_df['review'], test_df['rating'])
    print(f'CountVectorizer, Naive Bayes validation acc: {va_score}', flush=True)
    print(f'CountVectorizer, Naive Bayes test acc: {test_score}', flush=True)
    results.append(('CountVectorizer', 'Naive Bayes', test_score))

    # Baseline 3: TfidfVectorizer, Logistic regression
    pipeline = Pipeline([
        ('vect', TfidfVectorizer()),
        ('clf', LogisticRegression(random_state=0, max_iter=10000)),
    ])
    pipeline.fit(train_df['review'], train_df['rating'])
    va_score = pipeline.score(validation_df['review'], validation_df['rating'])
    test_score = pipeline.score(test_df['review'], test_df['rating'])
    print(f'TfidfVectorizer, Logistic regression validation acc: {va_score}', flush=True)
    print(f'TfidfVectorizer, Logistic regression test acc: {test_score}', flush=True)
    results.append(('TfidfVectorizer', 'Logistic regression', test_score))

    # Baseline 4: TfidfVectorizer, Naive Bayes
    pipeline = Pipeline([
        ('vect', TfidfVectorizer()),
        ('clf', MultinomialNB()),
    ])
    pipeline.fit(train_df['review'], train_df['rating'])
    va_score = pipeline.score(validation_df['review'], validation_df['rating'])
    test_score = pipeline.score(test_df['review'], test_df['rating'])
    print(f'TfidfVectorizer, Naive Bayes validation acc: {va_score}', flush=True)
    print(f'TfidfVectorizer, Naive Bayes test acc: {test_score}', flush=True)
    results.append(('TfidfVectorizer', 'Naive Bayes', test_score))

    # Baseline 5: BertTokenizer, Logistic regression
    pipeline = Pipeline([
        ('vect', BertTokenizerTransformer('bert-base-uncased')),
        ('clf', LogisticRegression(random_state=0, max_iter=10000)),
    ])
    pipeline.fit(train_df['review'], train_df['rating'])
    va_score = pipeline.score(validation_df['review'], validation_df['rating'])
    test_score = pipeline.score(test_df['review'], test_df['rating'])
    print(f'BertTokenizer, Logistic regression validation acc: {va_score}', flush=True)
    print(f'BertTokenizer, Logistic regression test acc: {test_score}', flush=True)
    results.append(('BertTokenizer', 'Logistic regression', test_score))

    # Baseline 6: BertTokenizer, Naive Bayes
    pipeline = Pipeline([
        ('vect', BertTokenizerTransformer('bert-base-uncased')),
        ('clf', MultinomialNB()),
    ])
    pipeline.fit(train_df['review'], train_df['rating'])
    va_score = pipeline.score(validation_df['review'], validation_df['rating'])
    test_score = pipeline.score(test_df['review'], test_df['rating'])
    print(f'BertTokenizer, Naive Bayes validation acc: {va_score}', flush=True)
    print(f'BertTokenizer, Naive Bayes test acc: {test_score}', flush=True)
    results.append(('BertTokenizer', 'Naive Bayes', test_score))

    # Save results
    results_df = pd.DataFrame(results, columns=['tokenizer', 'model', 'test_acc'])
    results_df.to_csv('results.csv', index=False)
    