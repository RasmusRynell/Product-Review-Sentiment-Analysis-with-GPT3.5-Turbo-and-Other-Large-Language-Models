import os
import openai
import json
import random
from sklearn.metrics import classification_report
from tqdm import tqdm

from data_processing import *
#from test_models import save_results

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

def save_results(test, predictions, model_name):
    report = classification_report(test['Sentiment'], predictions, output_dict=True)

    print(f"Model: {model_name}")
    print(classification_report(test['Sentiment'], predictions))

    #with open(f"results/{model_name}.csv", 'w') as f:
    #    f.write(json.dumps(report, indent=4))

def print_models():
    models = openai.Model.list()
    for model in models.data:
        print(model.id)

def setup_openai():
    # Read in api key from file
    api_key = open('api_key.txt', 'r').read()
    org_id = open('org_id.txt', 'r').read()

    openai.organization =  org_id
    openai.api_key = api_key

def add_shots(data, row, shot_for_each_sentiment=0, seed=42):
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
    #print(all_texts)

    return all_texts

def add_pre_determined_shots(data, row, seed=42):
    positive = ["Works great, A little expensive but worth it."]
    negative = ["Very low quality, I would not recommend this product."]
    neutral = ["It has a very high price but seems to be good quality."]

    template = f"Review: \"(summary)\" \nSentiment: (sentiment)"

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

    return all_texts


def test_open_ai_model(data, model_name, shot_for_each_sentiment=0, use_cache=True):
    starting_text = f"The following text is a sentiment analysis of product reviews. The following answers are available: [{', '.join(ints_to_sentiments.values())}]\n"
    
    cache = read_cache(model_name) if use_cache else {}

    predictions = []
    for index, row in tqdm(data.iterrows()):
        #print("---"*20)
        this_rows_input = f"Review: \"{row['Summary']}\" \nSentiment: "

        prompt = starting_text
        #shot_texts = add_shots(data, row, shot_for_each_sentiment)
        shot_texts = add_pre_determined_shots(data, row) # doesn't use "shot_for_each_sentiment"
        for shot in shot_texts:
            prompt += shot + "\n\n"

        prompt += this_rows_input

        if prompt in cache:
            print("Using cache...", flush=True)
            answer = cache[prompt]
        else:
            print("Cant find in cache, calling openai...", flush=True)
            try:
                completion = openai.Completion.create(model=model_name, prompt=prompt)
                answer = completion.choices[0].text.strip().lower()
                cache[prompt] = answer
            except Exception as e:
                save_cache(model_name, cache)
                print(e)
                exit()

        try:
            predictions.append(sentiments_to_ints[answer])
        except Exception as e:
            print("Error with answer: ", answer)
            print(e)
            exit()

    print("Saving cache...")
    save_cache(model_name, cache)

    print(f"Model: {model_name}")
    print(f"Predictions: {predictions}")

    save_results(data, predictions, model_name)

def test_gpt3(data, model_name, shot_for_each_sentiment=0, use_cache=True):
    system_text = "You are an expert in product reviews. \
        You are given a product review and must predict the sentiment of the review. \
        The following answers are available: [negative, neutral, positive]. \
        You dont answer in any other way, whatever happens. \
        Even if the input words are not in english or to short, \
        you still answer in one of the following ways always in only one word: [negative, neutral, positive]. \
        You never provide any other answer or additional information such as notes, comments, numbers or explanations."

    cache = read_cache(model_name) if use_cache else {}

    predictions = []
    for index, row in tqdm(data.iterrows()):
        this_rows_input = f"Review: \"{row['Summary']}\" \nSentiment: "

        messages = [{"role": "system", "content": system_text}]
        #shot_texts = add_shots(data, row, shot_for_each_sentiment)
        shot_texts = add_pre_determined_shots(data, row) # doesn't use "shot_for_each_sentiment"
        for shot in shot_texts:
            user, assistant = shot.split("\n")
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})

        messages.append({"role": "user", "content": this_rows_input})
        messages_in_text_form = str(messages).replace("'", '"')
        if messages_in_text_form in cache:
            print("Using cache...", flush=True)
            answer = cache[messages_in_text_form]
        else:
            print("Cant find in cache, calling openai...", flush=True)
            try:
                completion = openai.ChatCompletion.create(
                                model=model_name,
                                messages=messages)

                answer = completion.choices[0].message.content.strip().lower()
                answer = "positive" if "positive" in answer else "negative" if "negative" in answer else "neutral"
                print(completion.choices[0].message.content.strip().lower(), answer, flush=True)
                cache[messages_in_text_form] = answer
            except Exception as e:
                save_cache(model_name, cache)
                print(e)
                exit()

        try:
            predictions.append(sentiments_to_ints[answer])
        except Exception as e:
            print("Error with answer: ", answer)
            print(e)
            exit()

    print("Saving cache...")
    save_cache(model_name, cache)

    print(f"Model: {model_name}")
    print(f"Predictions: {predictions}")

    save_results(data, predictions, model_name)

if __name__ == "__main__":
    setup_openai()
    #print_models()

    cleaned_data = read_clean_data()
    _, test_data = split_data(cleaned_data, random_state=seed, validation=False, over_sample_train=False)

    model_name = "text-davinci-003"
    # "ada" or "text-davinci-003"

    # Randomly select 300 rows
    test_data = test_data.sample(n=300, random_state=seed)

    #print(test_data.head())
    #print(test_data['Sentiment'].value_counts())

    # test_open_ai_model(test_data, model_name, shot_for_each_sentiment=1, use_cache=True)
    test_gpt3(test_data, "gpt-3.5-turbo", shot_for_each_sentiment=1, use_cache=True)

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