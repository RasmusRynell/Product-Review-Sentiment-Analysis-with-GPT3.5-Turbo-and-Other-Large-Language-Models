import json
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from transformers import pipeline
from sklearn.metrics import accuracy_score
import torch
import time


def read_clean_data():
    data = pd.read_csv('sentiment.csv', encoding='latin-1')
    
    # Line (17301 in csv / 17299 in df) is corrupted, just remove it
    data = data.drop([17299])

    print(data.columns)

    # Drop columns that are not needed
    data = data.drop(['ProductPrice', 'Review', 'Rate', 'ProductName'], axis=1)

    data = data.dropna()

    # Convert to same values
    data['Sentiment'] = data['Sentiment'].replace(['Positive', 'Negative', 'Neutral'], ['positive', 'negative', 'neutral'])
    #data['Rate'] = data['Rate'].replace([1.0, 2.0, 3.0, 4.0, 5.0], ['1', '2', '3', '4', '5'])

    # Convert to ints
    data['Sentiment'] = data['Sentiment'].replace(['positive', 'neutral', 'negative'], [2, 1, 0])
    #data['Rate'] = data['Rate'].astype(int)

    # Drop text longer than 128 and shorter than 10
    data = data[data['Summary'].str.len() < 128]
    #data = data[data['Summary'].str.len() > 10]

    return data

def test_naive_bayes(train, test, validation):
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import accuracy_score

    # Train
    vectorizer = CountVectorizer()
    train_vectors = vectorizer.fit_transform(train['Summary'])
    classifier = MultinomialNB()
    classifier.fit(train_vectors, train['Sentiment'])

    # Test
    test_vectors = vectorizer.transform(test['Summary'])
    predictions = classifier.predict(test_vectors)
    print("Naive Bayes Accuracy: ", accuracy_score(test['Sentiment'], predictions))
    
def test_svm(train, test, validation):
    from sklearn.svm import SVC
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import accuracy_score

    # Train
    vectorizer = CountVectorizer()
    train_vectors = vectorizer.fit_transform(train['Summary'])
    classifier = SVC()
    classifier.fit(train_vectors, train['Sentiment'])

    # Test
    test_vectors = vectorizer.transform(test['Summary'])
    predictions = classifier.predict(test_vectors)
    print("SVM Accuracy: ", accuracy_score(test['Sentiment'], predictions))



def test_bert(train, test, validation):
    models = [
        "finiteautomata/bertweet-base-sentiment-analysis",
        "distilbert-base-uncased",
        "distilbert-base-uncased-finetuned-sst-2-english"
    ]

    results = []
    for model in models:
        print(model)
        sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=model, padding=True)
        # Test
        data = test['Summary'].tolist()

        predictions = sentiment_pipeline(data)

        #print(predictions)

        # Convert to ints
        predictions_done = []
        for i in range(len(predictions)):
            if predictions[i]['label'] in ['POSITIVE', 'Positive', 'positive', 'POS', 'Pos', 'pos', 'LABEL_2', '5 stars', '4 stars']:
                predictions_done.append(2)
            elif predictions[i]['label'] in ['NEUTRAL', 'neutral', 'neutral', 'NEU', 'Neu', 'neu', 'LABEL_1', '3 stars']:
                predictions_done.append(1)
            elif predictions[i]['label'] in ['NEGATIVE', 'Negative', 'negative', 'NEG', 'Neg', 'neg', 'LABEL_0', '2 star', '1 stars']:
                predictions_done.append(0)
        
        predictions = predictions_done

        results.append(accuracy_score(test['Sentiment'], predictions))
        # Print accuracy
        #print(f"{model} Accuracy: ", {accuracy_score(test['Sentiment'], predictions)})

    print(results)

def analysis(data):
    # Analysis
    print(data['Sentiment'].value_counts())
    print(data['Sentiment'].value_counts(normalize=True))

    # Plot
    #data['Sentiment'].value_counts().plot(kind='bar')
    #plt.show()

    # Plot
    #data['Sentiment'].value_counts(normalize=True).plot(kind='bar')
    #plt.show()

def equalize_data(data):
    # Oversample
    max_size = data['Sentiment'].value_counts().max()
    lst = [data]
    for class_index, group in data.groupby('Sentiment'):
        lst.append(group.sample(max_size-len(group), replace=True))
    data = pd.concat(lst)

    return data

if __name__ == '__main__':
    cleaned_data = read_clean_data()


    cleaned_data = equalize_data(cleaned_data)
    analysis(cleaned_data)


    for seed in range(10):
        # Set seed
        np.random.seed(seed)

        # Split data into train, test, validation
        train, test, validation = np.split(cleaned_data.sample(frac=1), [int(.6*len(cleaned_data)), int(.8*len(cleaned_data))])

        test = test[:100]

        # Naive Bayes
        test_naive_bayes(train, test, validation)

        # SVM
        #test_svm(train, test, validation)

        # bert
        test_bert(train, test, validation)