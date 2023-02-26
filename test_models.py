from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from transformers import pipeline as transformers_pipeline
import json

positive = ["pos", "Pos", "positive", "Positive", "POSITIVE", "LABEL_2"]
negative = ["neg", "Neg", "negative", "Negative", "NEGATIVE", "LABEL_0"]
neutral = ["neu", "Neu", "neutral", "Neutral", "NEUTRAL", "LABEL_1"]


def save_results(test, predictions, model_name):
    report = classification_report(test['Sentiment'], predictions, output_dict=True)
    print(f"Model: {model_name}")
    print(classification_report(test['Sentiment'], predictions))

    with open(f"results/{model_name}.csv", 'w') as f:
        f.write(json.dumps(report, indent=4))


def test_naive_bayes(train, test):
    curr_pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])

    curr_pipeline.fit(train['Summary'], train['Sentiment'])
    predictions = curr_pipeline.predict(test['Summary'])
    save_results(test, predictions, "naive_bayes")


def test_simple_sentiment(test):
    test = test[:100]
    global positive, negative, neutral

    simple_model = "cardiffnlp/twitter-roberta-base-sentiment"
    latest_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    def test_model(test, model):
        curr_pipeline = transformers_pipeline("sentiment-analysis", model=model, tokenizer=model)
        predictions = curr_pipeline(test['Summary'].tolist())
        predictions = ["positive" if x['label'] in positive else "negative" if x['label'] in negative else "neutral" for x in predictions]

        save_results(test, predictions, model.replace("/", "_") + f"_{len(test)}")

    test_model(test, simple_model)
    test_model(test, latest_model)

def test_my_model(train, test):
    pass