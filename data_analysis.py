import matplotlib.pyplot as plt
import spacy
from collections import Counter
from tabulate import tabulate
import json
import pandas as pd
import numpy as np


nlp = spacy.load('en_core_web_sm')


from data_processing import *

def tokens(text):
    return [t.lower() for t in text.split() if t.isalpha()]

def word_counts(data):
    counter = Counter()
    for text in data['Summary']:
        counter.update(tokens(text))

    # Remove stop words
    for word in nlp.Defaults.stop_words:
        counter.pop(word, None)

    # Convert to list of tuples
    word_counts = [(word, count) for word, count in counter.items()]
    word_counts = sorted(word_counts, key=lambda x: x[1], reverse=True)
    return word_counts

def plot_word_count(data, save=False, concat_string=''):
    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(bottom=0.30)
    plt.grid(axis='y', alpha=0.75)
    plt.title(f'Word count{concat_string.replace("_", " ")}')
    plt.xlabel('Word')
    plt.ylabel('Count')

    plt.ylim(0, 75000)

    counts = word_counts(data)
    keep = counts[:50]
    names = [x[0] for x in keep]
    values = [x[1] for x in keep]
    plt.xticks(rotation=90)
    plt.bar(names, values)
    plt.savefig(f'plots/word_count{concat_string}.png') if save else plt.show()

def plot_text_length(data, save=False, concat_string=''):
    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(bottom=0.15)
    plt.grid(axis='y', alpha=0.75)
    plt.grid(axis='x', alpha=0.0)
    plt.title(f'Distribution of text length{concat_string.replace("_", " ")}')
    plt.xlabel('Text length')
    plt.ylabel('Count')

    plt.ylim(0, 70000)

    data['Summary'].str.len().hist(bins=50)
    plt.savefig(f'plots/text_length{concat_string}.png') if save else plt.show()

    # Print the length of the longest text
    print(f"Longest text: {data['Summary'].str.len().max()}")
    print(f"Shortest text: {data['Summary'].str.len().min()}")
    print(f"Average text length: {data['Summary'].str.len().mean()}")

def plot_distribution_sentiment(data, save=False, concat_string=''):
    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(bottom=0.15)
    plt.grid(axis='y', alpha=0.75)
    plt.title(f'Distribution of sentiment{concat_string.replace("_", " ")}')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')

    plt.ylim(0, 90000)

    data['Sentiment'].value_counts().plot(kind='bar')

    # Change the x-axis labels to "Positive", "Neutral" and "Negative"
    plt.xticks([0, 1, 2], ['Positive', 'Neutral', 'Negative'], rotation=0)

    plt.savefig(f'plots/sentiment_distribution{concat_string}.png') if save else plt.show()


def analyze_data(data, save=False, concat_string=''):
    # Plot distribution of data
    plt.rcParams.update({'font.size': 25})
    plot_distribution_sentiment(data, save, concat_string)

    # Plot distribution word count
    plt.rcParams.update({'font.size': 25})
    plot_word_count(data, save, concat_string)

    # Plot distribution of text length
    plt.rcParams.update({'font.size': 25})
    plot_text_length(data, save, concat_string)


def initial_analysis(data):
    sentiments_dict = {0: 'negative', 1: 'neutral', 2: 'positive'}

    # How many times does the word "good" appear for each sentiment? relative to how many times each sentiment appears
    positive_length = len(data[data['Sentiment'] == 2])
    neutral_length = len(data[data['Sentiment'] == 1])
    negative_length = len(data[data['Sentiment'] == 0])

    results = {}

    # Top 10 words for each sentiment
    for key, sentiment in sentiments_dict.items():
        #print(f"Top 10 words for {sentiment}")
        top_10 = word_counts(data[data['Sentiment'] == key])[:10]

        results[key] = {}

        for word in top_10:
            word = word[0]
            positive_count = data[data['Sentiment'] == 2]['Summary'].str.lower().str.count(word).sum() / positive_length
            neutral_count = data[data['Sentiment'] == 1]['Summary'].str.lower().str.count(word).sum() / neutral_length
            negative_count = data[data['Sentiment'] == 0]['Summary'].str.lower().str.count(word).sum() / negative_length

            # print(f"Word: {word}")
            # print(f"Positive count: {positive_count}")
            # print(f"Neutral count: {neutral_count}")
            # print(f"Negative count: {negative_count}")

            # print()

            results[key][word] = {'positive': positive_count, 'neutral': neutral_count, 'negative': negative_count}

        print()

    # Sort all results
    for key, sentiment in sentiments_dict.items():
        results[key] = {k: v for k, v in sorted(results[key].items(), key=lambda item: item[1]['positive'], reverse=True)}

    # Plot results
    for key, sentiment in sentiments_dict.items():
        plt.clf()
        plt.figure(figsize=(20, 10))
        plt.subplots_adjust(bottom=0.30)
        plt.grid(axis='y', alpha=0.75)
        plt.title(f'Top 10 words for {sentiment}')
        plt.xlabel('Word')
        plt.ylabel('Count')

        plt.ylim(0, 0.5)

        names = [x for x in results[key].keys()]
        values = [results[key][x]['positive'] for x in results[key].keys()]
        plt.xticks(rotation=90)
        plt.bar(names, values)
        plt.savefig(f'plots/top_10_words_{sentiment}.png')

    # Print the value for "good" for each sentiment
    for key, sentiment in sentiments_dict.items():
        print(f"Value for 'good' for {sentiment}: {results[key]['good']}")

def smooth(data, max_smooth=10):
    smoothed = []
    # Rolling average
    for i in range(len(data)):
        j = i - max_smooth
        j = 0 if j < 0 else j
        smoothed.append(sum(data[j:i + 1]) / (i - j + 1))
        
    return smoothed

def loss_plot(file, save=False, concat_string=''):
    plt.rcParams.update({'font.size': 25})
    with open(file, 'r') as f:
        data = json.load(f)

    train_loss = []
    train_loss_step = []
    val_loss = []
    val_loss_step = []
    for time_step in data:
        if "loss" in time_step:
            train_loss.append(time_step['loss'])
            train_loss_step.append(time_step['step'])
        elif "eval_loss" in time_step:
            val_loss.append(time_step['eval_loss'])
            val_loss_step.append(time_step['step'])
        else:
            print("Stats")
            print(json.dumps(time_step, indent=4))


    # Smoothing
    train_loss = smooth(train_loss, max_smooth=1)
    val_loss = smooth(val_loss, max_smooth=1)

    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(bottom=0.15)
    plt.grid(axis='y', alpha=0.75)
    plt.title(f'Loss over time{concat_string.replace("_", " ")}')
    plt.xlabel('Step')
    plt.ylabel('Loss')

    plt.plot(train_loss_step, train_loss, label='Training loss')
    plt.plot(val_loss_step, val_loss, label='Validation loss')
    plt.legend()
    plt.savefig(f'plots/loss_over_time{concat_string}.png') if save else plt.show()



if __name__ == '__main__':
    cleaned_data = read_clean_data()
    # analyze_data(cleaned_data, save=True, concat_string=f"_all_data")

    # initial_analysis(cleaned_data)

    # Plots changes for over sampling
    train, test = split_data(cleaned_data, random_state=42, validation=False, over_sample_train=False)

    # # Plot changes for over sampling
    analyze_data(train, save=True, concat_string=f"_training_data_before_over_sample")
    over_samples_train = over_sample(train, 42)
    analyze_data(over_samples_train, save=True, concat_string=f"_training_data_after_over_sample")


    # Plot loss over time
    loss_plot('models/my/distilbert-base-uncased_log_history_no_over_sample.json', save=True, concat_string=f"_no_over_sample")
    loss_plot('models/my/distilbert-base-uncased_log_history_over_sample.json', save=True, concat_string=f"_over_sample")
    