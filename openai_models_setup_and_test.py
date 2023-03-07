import os
import openai
import json
import random
from sklearn.metrics import classification_report
from tqdm import tqdm

from data_processing import *

ints_to_sentiments = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

sentiments_to_ints = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

seed = 42
random.seed(seed)

def read_cache(model_name):
    full_path = f"cache/{model_name}.json"
    if not os.path.exists(full_path):
        return {}
    with open(full_path, 'r') as f:
        return json.load(f)

def save_cache(model_name, new_cache):
    full_path = f"cache/{model_name}.json"
    with open(full_path, 'w') as f:
        json.dump(new_cache, f, indent=4)

def save_results(test, predictions, model_name, shots_for_each_sentiment, should_be_one):
    report = classification_report(test['Sentiment'], predictions, output_dict=True)

    print(f"Model: {model_name}")
    print(classification_report(test['Sentiment'], predictions))

    name = f"results/{model_name}_0_shot.csv"
    if shots_for_each_sentiment > 0:
        if should_be_one:
            name = f"results/{model_name}_{shots_for_each_sentiment}_shot_one.csv"
        else:
            name = f"results/{model_name}_{shots_for_each_sentiment*3}_shot.csv"

    # Add in "_new" before .csv
    name = name[:-4] + "_new2" + name[-4:]

    with open(name, 'w') as f:
        f.write(json.dumps(report, indent=4))

def print_models():
    models = openai.Model.list()
    for model in models.data:
        if "curie" in model.id:
            print("\t", end="")
        print(model.id)

def setup_openai():
    api_key = open('api_key.txt', 'r').read()
    org_id = open('org_id.txt', 'r').read()

    openai.organization =  org_id
    openai.api_key = api_key


def add_random_shots(data, row, shot_for_each_sentiment=0, seed=42):
    # Take out "shot_for_each_sentiment" number of reviews for each sentiment
    template = f"Review: \"(summary)\" \nSentiment: (sentiment)"
    all_texts = []
    for sentiment_int, sentiment_text in ints_to_sentiments.items():
        all_rows_with_sentiment = data[data['Sentiment'] == sentiment_int]

        # If "row" is in the list, remove it
        if row['Summary'] in all_rows_with_sentiment['Summary'].values:
            all_rows_with_sentiment = all_rows_with_sentiment[all_rows_with_sentiment['Summary'] != row['Summary']]

        sampled_rows = all_rows_with_sentiment.sample(n=shot_for_each_sentiment, random_state=seed)

        for index, sampled_row in sampled_rows.iterrows():
            all_texts.append(template.replace("(summary)", sampled_row['Summary']).replace("(sentiment)", sentiment_text))

    random.shuffle(all_texts)

    return all_texts

def add_pre_determined_shots(data, row, seed=42, should_be_one=False):
    positive = ["Works great, a little expensive but worth it."]
    negative = ["Very low quality, I would not recommend this product."]
    neutral = ["It has a very high price but seems to be good quality."]

    template = f"Review: (summary) \nSentiment: (sentiment)"

    all_texts = []
    for sentiment_int, sentiment_text in ints_to_sentiments.items():
        if sentiment_int == 0:
            texts = negative
        elif sentiment_int == 1:
            texts = neutral
        elif sentiment_int == 2:
            texts = positive

        for text in texts:
            all_texts.append(template.replace("(summary)", text).replace("(sentiment)", sentiment_text))

    random.shuffle(all_texts)

    if should_be_one:
        return [all_texts[0]]
    return all_texts


def test_open_ai_model(data, model_name, settings, shot_for_each_sentiment=0, use_cache=True, should_be_one=False):
    starting_text = f"This is a product review sentiment analysis. You can choose your response from the following options: [{', '.join(ints_to_sentiments.values())}]\n\n"

    cache = read_cache(model_name) if use_cache else {}

    predictions = []
    for index, row in tqdm(data.iterrows()):
        this_rows_input = f"Review: {row['Summary']} \nSentiment: "

        prompt = starting_text
        if shot_for_each_sentiment > 0:
            for shot in add_pre_determined_shots(data, row, seed, should_be_one):
                prompt += shot + "\n\n"
        else:
            prompt += "\n\n"

        prompt += this_rows_input

        if prompt in cache:
            print("Using cache...", flush=True)
            answer = cache[prompt]
        else:
            print("Cant find in cache, calling openai...", flush=True)
            try:
                completion = openai.Completion.create(model=model_name, prompt=prompt, **settings)
                answer = completion.choices[0].text.strip().lower()
                answer = "positive" if "positive" in answer else "negative" if "negative" in answer else "neutral"
                cache[prompt] = answer
            except Exception as e:
                save_cache(model_name, cache)
                print(e)
                return

        try:
            predictions.append(sentiments_to_ints[answer])
        except Exception as e:
            print("Error with prompt: ", prompt)
            print("Error with answer: ", answer)
            print(e)
            return

    print("Saving cache...")
    save_cache(model_name, cache)

    print(f"Model: {model_name}")
    print(f"Predictions: {predictions}")

    save_results(data, predictions, model_name, shot_for_each_sentiment, should_be_one)


def test_gpt3(data, model_name, settings, shot_for_each_sentiment=0, use_cache=True, should_be_one=False):
    system_text = "Predict the sentiment of a product review as either positive, neutral, or negative. You provide only ONE word as your answer, without any additional information, such as notes, comments, numbers, or explanations."

    cache = read_cache(model_name) if use_cache else {}

    predictions = []
    for index, row in tqdm(data.iterrows()):
        this_rows_input = f"Review: {row['Summary']}"

        messages = [{"role": "system", "content": system_text}]
        if shot_for_each_sentiment > 0:
            shot_texts = add_pre_determined_shots(data, row, 42, should_be_one)
            for shot in shot_texts:
                user, assistant = shot.split(" \n")
                messages.append({"role": "user", "content": user})
                messages.append({"role": "assistant", "content": assistant})
        
        messages.append({"role": "user", "content": this_rows_input})
        messages_in_text_form = str(messages).replace("'", '"')
        if messages_in_text_form in cache:
            print("Using cache...", flush=True)
            answer = cache[messages_in_text_form]
        else:
            try:
                completion = openai.ChatCompletion.create(
                                model=model_name,
                                messages=messages,
                                **settings)

                answer = completion.choices[0].message.content.strip().lower()
                answer = "positive" if "positive" in answer else "negative" if "negative" in answer else "neutral"
                cache[messages_in_text_form] = answer
            except Exception as e:
                save_cache(model_name, cache)
                print(e)
                return

        try:
            predictions.append(sentiments_to_ints[answer])
        except Exception as e:
            print("Error with answer: ", answer)
            print(e)
            return

    save_cache(model_name, cache)

    save_results(data, predictions, model_name, shot_for_each_sentiment, should_be_one)

if __name__ == "__main__":
    settings = {
        "temperature": 0.3,
        "top_p":0.33,
    }

    setup_openai()
    #print_models()

    cleaned_data = read_clean_data()
    _, test_data = split_data(cleaned_data, random_state=seed, validation=False, over_sample_train=False)

    # Randomly select 300 rows
    test_data = test_data.sample(n=300, random_state=seed)

    # 0 and 3 shot classification
    for num_sentiments in [0, 1]:
        test_open_ai_model(test_data, "text-curie-001", settings, shot_for_each_sentiment=num_sentiments, use_cache=True, should_be_one=False)
        test_gpt3(test_data, "gpt-3.5-turbo", settings, shot_for_each_sentiment=num_sentiments, use_cache=True, should_be_one=False)

    # 1 shot classification
    test_open_ai_model(test_data, "text-curie-001", settings, shot_for_each_sentiment=1, use_cache=True, should_be_one=True)
    test_gpt3(test_data, "gpt-3.5-turbo", settings, shot_for_each_sentiment=1, use_cache=True, should_be_one=True)

''' 300, pre-determined (1)
Model: text-davinci-003
              precision    recall  f1-score   support

           0       0.71      0.88      0.79        40
           1       0.22      0.50      0.30        14
           2       1.00      0.89      0.94       246

    accuracy                           0.87       300
   macro avg       0.64      0.75      0.68       300
weighted avg       0.92      0.87      0.89       300
'''

''' 300, pre-determined (2,1,4)
Model: text-davinci-003
              precision    recall  f1-score   support

           0       0.69      0.90      0.78        40
           1       0.24      0.43      0.31        14
           2       1.00      0.90      0.95       246

    accuracy                           0.88       300
   macro avg       0.64      0.74      0.68       300
weighted avg       0.92      0.88      0.89       300
'''

''' 300, 0
Model: text-davinci-003
              precision    recall  f1-score   support

           0       0.68      0.90      0.77        40
           1       0.12      0.36      0.18        14
           2       1.00      0.84      0.91       246

    accuracy                           0.82       300
   macro avg       0.60      0.70      0.62       300
weighted avg       0.92      0.82      0.86       300
'''

''' 300, 1
Model: text-davinci-003
              precision    recall  f1-score   support

           0       0.71      0.85      0.77        40
           1       0.21      0.43      0.28        14
           2       1.00      0.90      0.95       246

    accuracy                           0.87       300
   macro avg       0.64      0.73      0.67       300
weighted avg       0.92      0.87      0.89       300
'''

''' 300, 2
Model: text-davinci-003
              precision    recall  f1-score   support

           0       0.71      0.88      0.79        40
           1       0.27      0.57      0.36        14
           2       1.00      0.89      0.94       246

    accuracy                           0.88       300
   macro avg       0.66      0.78      0.70       300
weighted avg       0.92      0.88      0.89       300
'''