import pandas as pd

def read_clean_data():
    data = pd.read_csv('data/sentiment.csv', encoding='latin-1')
    
    # Line (17301 in csv / 17299 in df) is corrupted, just remove it
    data = data.drop([17299])

    print(data.columns)

    # Drop columns that are not needed
    data = data.drop(['ProductPrice', 'Review', 'Rate', 'ProductName'], axis=1)

    # Remember how many rows we have
    old_len = len(data)

    data = data.dropna()

    # Check if we have dropped any rows
    print(f"Removed {old_len - len(data)} rows with NaN values")

    # Convert to same values
    data['Sentiment'] = data['Sentiment'].replace(['Positive', 'Negative', 'Neutral'], ['positive', 'negative', 'neutral'])

    # Convert to ints
    data['Sentiment'] = data['Sentiment'].replace(['positive', 'neutral', 'negative'], [2, 1, 0])

    old_len = len(data)
    remove_len = 450
    data = data[data['Summary'].str.len() < remove_len]
    print(f"Removed {old_len - len(data)} rows with text longer than {remove_len} characters")

    # This many rows are left
    print(f"Rows left: {len(data)}")

    # Convert all to lowercase
    data['Summary'] = data['Summary'].str.lower()

    return data

