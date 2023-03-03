from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from transformers import pipeline as transformers_pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from evaluate import evaluator
import json
import torch

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


def find_optimal_parameters(train, test):
    pipe = Pipeline([('vectorizer', CountVectorizer()), ('MNB', MultinomialNB())])
    GSCV = GridSearchCV(estimator=pipe,
             param_grid={
                    "vectorizer__binary" : [False, True],
                    "vectorizer__ngram_range" : [(1,1),(2,2),(3,3),(4,4),(5,5)],
                    "MNB__alpha" : [0.1,
                                    0.5,
                                    1.0,
                                    1.5,
                                    2.0]},
                    n_jobs=-1)

    GSCV.fit(train['Summary'], train['Sentiment'])
    print(GSCV.best_params_)
    predictions = GSCV.predict(test['Summary'])
    save_results(test, predictions, "optimized_naive_bayes")


def test_optimized_naive_bayes(train, test):
    curr_pipeline = Pipeline([
        ('vectorizer', CountVectorizer(binary = True, ngram_range=(2,2))),
        ('classifier', MultinomialNB(alpha=0.1))
    ])

    curr_pipeline.fit(train['Summary'], train['Sentiment'])
    predictions = curr_pipeline.predict(test['Summary'])
    save_results(test, predictions, "optimized_naive_bayes")


def test_simple_sentiment(test):
    global positive, negative, neutral

    simple_model = "cardiffnlp/twitter-roberta-base-sentiment"
    latest_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    def test_model(test, model):
        curr_pipeline = transformers_pipeline("sentiment-analysis", model=model, tokenizer=model)
        predictions = curr_pipeline(test['Summary'].tolist())
        predictions = [2 if x['label'] in positive else 0 if x['label'] in negative else 1 for x in predictions]

        save_results(test, predictions, model.replace("/", "_") + f"_{len(test)}")

    test_model(test, simple_model)
    test_model(test, latest_model)


def test_my_model(test, path):
    global positive, negative, neutral

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=3)
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        # Prepare data
        test = test.reset_index(drop=True)

        batch_size = 64

        # Predict
        predictions = []
        for i in range(0, len(test), batch_size):
            batch = test[i:i+batch_size]
            batch = tokenizer(batch['Summary'].tolist(), return_tensors="pt", max_length=128, truncation=True, padding=True)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions.extend(torch.argmax(outputs.logits, dim=1).tolist())

        save_results(test, predictions, path.replace("/", "_") + f"_{len(test)}")
