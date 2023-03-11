import matplotlib.pyplot as plt
import spacy
from collections import Counter
from tabulate import tabulate
import json
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_theme(style='darkgrid', rc={'figure.dpi': 147},              
              font_scale=2)

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

    dist = [len(data[data['Sentiment'] == 2]), \
            len(data[data['Sentiment'] == 1]), \
            len(data[data['Sentiment'] == 0])]

    max_value = max(dist)
    plt.ylim(0, max_value + 10000)

    plt.bar(['Positive', 'Neutral', 'Negative'], dist)

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

    sns.set_style("darkgrid")
    plt.plot(train_loss_step, train_loss, label='Train loss')
    plt.plot(val_loss_step, val_loss, label='Validation loss')

    plt.legend()
    plt.savefig(f'plots/loss_over_time{concat_string}.png') if save else plt.show()

def over_sample_analysis(data, save=False):

    not_split_train, test = split_data(cleaned_data, random_state=42, validation=False, over_sample_train=False)
    over_sampled_train, _ = split_data(cleaned_data, random_state=42, validation=False, over_sample_train=True)

    plt.rcParams.update({'font.size': 25})
    # Plot distribution of "Summary" length
    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(bottom=0.15)
    plt.grid(axis='y', alpha=0.75)
    plt.title(f'Distribution of text length')
    plt.xlabel('Text length')
    plt.ylabel('Count')
    plt.ylim(0, 70000)

    # Plot both distributions on the same plot, 50 bins
    bins = [x for x in range(0, 400, 10)]

    sns.histplot(over_sampled_train['Summary'].str.len(), bins=bins, label='Oversampled', kde=False)
    sns.histplot(data['Summary'].str.len(), bins=bins, label='All data', kde=False)
    sns.histplot(not_split_train['Summary'].str.len(), bins=bins, label='Original', kde=False)

    plt.legend()
    plt.savefig(f'plots/text_length_distribution_over_sample.png') if save else plt.show()


    # Plot distribution of sentiment
    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(bottom=0.15)
    plt.grid(axis='y', alpha=0.75)
    plt.title(f'Distribution of sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.ylim(0, 120000)

    # Plot both distributions on the same plot
    data_dist = [len(data[data['Sentiment'] == 2]), \
                len(data[data['Sentiment'] == 1]), \
                len(data[data['Sentiment'] == 0])]

    test_dist = [len(test[test['Sentiment'] == 2]), \
                len(test[test['Sentiment'] == 1]), \
                len(test[test['Sentiment'] == 0])]

    train_not_over_dist = [len(not_split_train[not_split_train['Sentiment'] == 2]), \
                          len(not_split_train[not_split_train['Sentiment'] == 1]), \
                          len(not_split_train[not_split_train['Sentiment'] == 0])]

    # Change the x-axis labels to "Positive", "Neutral" and "Negative"
    plt.xticks([0, 1, 2], ['Positive', 'Neutral', 'Negative'], rotation=0)

    plt.bar([0, 1, 2], data_dist, label='All data', alpha=1)
    plt.bar([0, 1, 2], train_not_over_dist, label='Train not oversampled', alpha=1)
    plt.bar([0, 1, 2], test_dist, label='Test', alpha=1)
    
    plt.legend(bbox_to_anchor=(1, 1), title='Datasets')

    # Print percentages
    print("All data")
    print(f"Positive: {data_dist[0] / sum(data_dist) * 100:.2f}%")
    print(f"Neutral: {data_dist[1] / sum(data_dist) * 100:.2f}%")
    print(f"Negative: {data_dist[2] / sum(data_dist) * 100:.2f}%")

    print("Train not oversampled")
    print(f"Positive: {train_not_over_dist[0] / sum(train_not_over_dist) * 100:.2f}%")
    print(f"Neutral: {train_not_over_dist[1] / sum(train_not_over_dist) * 100:.2f}%")
    print(f"Negative: {train_not_over_dist[2] / sum(train_not_over_dist) * 100:.2f}%")

    print("Test")
    print(f"Positive: {test_dist[0] / sum(test_dist) * 100:.2f}%")
    print(f"Neutral: {test_dist[1] / sum(test_dist) * 100:.2f}%")
    print(f"Negative: {test_dist[2] / sum(test_dist) * 100:.2f}%")

    plt.savefig(f'plots/sentiment_distribution_over_sample.png') if save else plt.show()

    # Plot distribution word count before and after oversampling
    plot_word_count(not_split_train, save, "_before_over_sample")

    # Plot distribution word count
    plot_word_count(over_sampled_train, save, "_after_over_sample")



if __name__ == '__main__':
    cleaned_data = read_clean_data()
    analyze_data(cleaned_data, save=True, concat_string=f"_all_data")

    # initial_analysis(cleaned_data)

    # Plots changes for oversampling
    over_sample_analysis(cleaned_data, save=True)

    # Plot loss over time
    loss_plot('models/my/distilbert-base-uncased_log_history_no_over_sample.json', save=True, concat_string=f"_no_over_sample")
    loss_plot('models/my/distilbert-base-uncased_log_history_over_sample.json', save=True, concat_string=f"_over_sample")
