import matplotlib.pyplot as plt
import spacy
nlp = spacy.load('en_core_web_sm')

from data import read_clean_data

def word_counts(data):
    word_counts = {}
    for i in range(len(data)):
        for word in data.iloc[i]['Summary'].split(' '):
            word_counts[word] = word_counts.get(word, 0) + 1

    # Remove stop words
    stop_words = nlp.Defaults.stop_words
    for word in stop_words:
        word_counts.pop(word, None)

    # Remove words that are not in the english dictionary
    for word in word_counts.copy():
        if not nlp.vocab[word].is_oov:
            word_counts.pop(word, None)

    # Remove unwanted characters
    word_counts.pop('', None)

    word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)


    return word_counts

def plot_word_count(data, save=False):
    plt.clf()
    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(bottom=0.15)
    plt.grid(axis='y', alpha=0.75)
    plt.title('Word count')
    plt.xlabel('Word')
    plt.ylabel('Count')

    counts = word_counts(data)
    keep = counts[:100]

    names = [x[0] for x in keep]
    values = [x[1] for x in keep]

    plt.xticks(rotation=90)
    plt.bar(names, values)

    plt.savefig('plots/word_count.png') if save else plt.show()

def plot_text_length(data, save=False):
    plt.clf()
    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(bottom=0.15)
    plt.grid(axis='y', alpha=0.75)
    plt.grid(axis='x', alpha=0.0)
    plt.title('Distribution of text length')
    plt.xlabel('Text length')
    plt.ylabel('Count')

    data['Summary'].str.len().hist(bins=50)

    plt.savefig('plots/text_length.png') if save else plt.show()

def plot_distribution_sentiment(data, save=False):
    plt.clf()
    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(bottom=0.15)
    plt.grid(axis='y', alpha=0.75)
    plt.title('Distribution of sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')

    data['Sentiment'].value_counts().plot(kind='bar')

    plt.savefig('plots/sentiment_distribution.png') if save else plt.show()

# Analyze distribution of data
def analyze_data(data, save=False):
    # Plot distribution of data
    plot_distribution_sentiment(data, save)

    # Plot distribution word count
    plot_word_count(data, save)

    # Plot distribution of text length
    plot_text_length(data, save)



if __name__ == '__main__':
    cleaned_data = read_clean_data()
    analyze_data(cleaned_data, save=True)