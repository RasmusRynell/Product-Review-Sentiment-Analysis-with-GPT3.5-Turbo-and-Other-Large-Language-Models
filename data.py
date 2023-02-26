import pandas as pd

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

    # Convert to ints
    #data['Sentiment'] = data['Sentiment'].replace(['positive', 'neutral', 'negative'], [2, 1, 0])

    # Drop text longer than 128 and shorter than 10
    data = data[data['Summary'].str.len() < 128]

    return data

