from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from transformers import pipeline as transformers_pipeline

def test_naive_bayes(train, test, validation):
    curr_pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])

    curr_pipeline.fit(train['Summary'], train['Sentiment'])
    predictions = curr_pipeline.predict(test['Summary'])
    print("Naive Bayes Accuracy: ", accuracy_score(test['Sentiment'], predictions))
    print(classification_report(test['Sentiment'], predictions))

def test_svm(train, test, validation):
    curr_pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', SVC())
    ])

    curr_pipeline.fit(train['Summary'], train['Sentiment'])
    predictions = curr_pipeline.predict(test['Summary'])
    print("SVM Accuracy: ", accuracy_score(test['Sentiment'], predictions))
    print(classification_report(test['Sentiment'], predictions))

def test_simple_sentiment(train, test, validation):
    models = [
        "finiteautomata/bertweet-base-sentiment-analysis",
        "distilbert-base-uncased",
        "distilbert-base-uncased-finetuned-sst-2-english"
    ]

    positive = ["pos", "Pos", "positive", "Positive", "POSITIVE"]
    negative = ["neg", "Neg", "negative", "Negative", "NEGATIVE"]
    neutral = ["neu", "Neu", "neutral", "Neutral", "NEUTRAL"]

    results = []
    for model in models:
        curr_pipeline = transformers_pipeline("sentiment-analysis", model=model, tokenizer=model)
        predictions = curr_pipeline(test['Summary'].tolist())
        predictions = [2 if x['label'] in positive else 0 if x['label'] in negative else 1 for x in predictions]
        results.append(predictions)

    for i in range(len(models)):
        print(models[i], "Accuracy: ", accuracy_score(test['Sentiment'], results[i]))
        print(classification_report(test['Sentiment'], results[i]))

def test_my_model(train, test, validation):
    pass