import pandas as pd
from imblearn.over_sampling import RandomOverSampler

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

    return data

def over_sample(data, random_state):
    # Random oversampling
    ros = RandomOverSampler(random_state=random_state)
    X_resampled, y_resampled = ros.fit_resample(data['Summary'].values.reshape(-1, 1), data['Sentiment'])

    # Convert to dataframe
    X_resampled = pd.DataFrame(X_resampled)
    y_resampled = pd.DataFrame(y_resampled)
    resampled = pd.concat([X_resampled, y_resampled], axis=1)
    resampled.columns = ['Summary', 'Sentiment']

    return resampled

def split_data(data, random_state=42, validation=False, over_sample_train=False):
    train_org = data.sample(frac=0.8, random_state=random_state)
    test = data.drop(train_org.index)

    if validation:
        train_new = train_org.sample(frac=0.9, random_state=random_state)
        validation = train_org.drop(train_new.index)

        if over_sample_train:
            train_new = over_sample(train_new, random_state)

        return train_new, validation, test

    if over_sample_train:
        train_org = over_sample(train_org, random_state)

    return train_org, test
