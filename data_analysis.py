import matplotlib.pyplot as plt
import spacy
from collections import Counter

nlp = spacy.load('en_core_web_sm')


from data_processing import *

def tokens(text):
    return [t.lower() for t in text.split() if t.isalpha()]

def word_counts(data):
    counter = Counter()
    for text in data['Summary']:
        counter.update(tokens(text))

    # Remove stop words
    stop_words = nlp.Defaults.stop_words
    for word in stop_words:
        counter.pop(word, None)

    # Convert to list of tuples
    word_counts = [(word, count) for word, count in counter.items()]
    word_counts = sorted(word_counts, key=lambda x: x[1], reverse=True)
    return word_counts

def plot_word_count(data, save=False, concat_string=''):
    #plt.clf()
    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(bottom=0.30)
    plt.grid(axis='y', alpha=0.75)
    plt.title('Word count')
    plt.xlabel('Word')
    plt.ylabel('Count')

    counts = word_counts(data)
    keep = counts[:50]
    names = [x[0] for x in keep]
    values = [x[1] for x in keep]
    plt.xticks(rotation=90)
    plt.bar(names, values)
    plt.savefig(f'plots/word_count{concat_string}.png') if save else plt.show()

def plot_text_length(data, save=False, concat_string=''):
    plt.clf()
    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(bottom=0.15)
    plt.grid(axis='y', alpha=0.75)
    plt.grid(axis='x', alpha=0.0)
    plt.title('Distribution of text length')
    plt.xlabel('Text length')
    plt.ylabel('Count')

    data['Summary'].str.len().hist(bins=50)
    plt.savefig(f'plots/text_length{concat_string}.png') if save else plt.show()

    # Print the length of the longest text
    print(f"Longest text: {data['Summary'].str.len().max()}")
    print(f"Shortest text: {data['Summary'].str.len().min()}")
    print(f"Average text length: {data['Summary'].str.len().mean()}")

def plot_distribution_sentiment(data, save=False, concat_string=''):
    plt.clf()
    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(bottom=0.15)
    plt.grid(axis='y', alpha=0.75)
    plt.title('Distribution of sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')

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



if __name__ == '__main__':
    cleaned_data = read_clean_data()
    analyze_data(cleaned_data, save=True, concat_string='all_data')

    # Plots changes for over sampling
    train, test = split_data(cleaned_data, random_state=42, validation=False, over_sample_train=False)
    analyze_data(train, save=True, concat_string=f"_training_data_before_over_sample")
    train = over_sample(train, 42)
    analyze_data(train, save=True, concat_string=f"_training_data_after_over_sample")