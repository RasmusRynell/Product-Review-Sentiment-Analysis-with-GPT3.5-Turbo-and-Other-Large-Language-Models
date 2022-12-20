import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression

threads = 8

import common
idx = 1
df, train_df, validation_df, test_df = common.get_dataset(idx)

# Reset all indexes
train_df = train_df.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# Debug, print CountVectorizer of first text
test_idx = 1
print(train_df.iloc[test_idx]['x'], flush=True)


# Scale the data
# y_scaler = StandardScaler()
# y_scaler.fit(train_df['y'].values.reshape(-1, 1))
# train_df['y'] = y_scaler.transform(train_df['y'].values.reshape(-1, 1))
# validation_df['y'] = y_scaler.transform(validation_df['y'].values.reshape(-1, 1))
# test_df['y'] = y_scaler.transform(test_df['y'].values.reshape(-1, 1))


# Bag of words
# Baseline 1: Linear regression
linear_regression = Pipeline([('vect', CountVectorizer()), ('model', LinearRegression())])
linear_regression.fit(train_df['x'], train_df['y'])
y_pred = linear_regression.predict(validation_df['x'])
mse = ((validation_df['y'] - y_pred) ** 2).mean()
print(f'Linear regression MSE:\t {mse:.2f}', flush=True)
print(f'{y_pred[test_idx]:.2f}, {validation_df["y"][test_idx]:.2f} = {y_pred[test_idx] - validation_df["y"][test_idx]:.2f}', flush=True)

# Baseline 2: Ridge regression
ridge_regression = Pipeline([('vect', CountVectorizer()), ('model', Ridge())])
ridge_regression.fit(train_df['x'], train_df['y'])
y_pred = ridge_regression.predict(validation_df['x'])
mse = ((validation_df['y'] - y_pred) ** 2).mean()
print(f'Ridge regression MSE:\t {mse:.2f}', flush=True)
print(f'{y_pred[test_idx]:.2f}, {validation_df["y"][test_idx]:.2f} = {y_pred[test_idx] - validation_df["y"][test_idx]:.2f}', flush=True)

print()

# TF-IDF
# Baseline 4: Linear regression
linear_regression = Pipeline([('vect', TfidfVectorizer()), ('model', LinearRegression())])
linear_regression.fit(train_df['x'], train_df['y'])
y_pred = linear_regression.predict(validation_df['x'])
mse = ((validation_df['y'] - y_pred) ** 2).mean()
print(f'Linear regression MSE:\t {mse:.2f}', flush=True)
print(f'{y_pred[test_idx]:.2f}, {validation_df["y"][test_idx]:.2f} = {y_pred[test_idx] - validation_df["y"][test_idx]:.2f}', flush=True)

# Baseline 5: Ridge regression
ridge_regression = Pipeline([('vect', TfidfVectorizer()), ('model', Ridge())])
ridge_regression.fit(train_df['x'], train_df['y'])
y_pred = ridge_regression.predict(validation_df['x'])
mse = ((validation_df['y'] - y_pred) ** 2).mean()
print(f'Ridge regression MSE:\t {mse:.2f}', flush=True)
print(f'{y_pred[test_idx]:.2f}, {validation_df["y"][test_idx]:.2f} = {y_pred[test_idx] - validation_df["y"][test_idx]:.2f}', flush=True)

print()

# Print text for largest y_true
print('Largest y_true:')
print(train_df.iloc[train_df['y'].idxmax()]['x'], flush=True)
print(validation_df.iloc[validation_df['y'].idxmax()]['x'], flush=True)
print(test_df.iloc[test_df['y'].idxmax()]['x'], flush=True)


# Show a bar chart of y_true for train, validation and test in different colors
plt.bar(range(len(train_df)), train_df['y'], color='b', label='train')
plt.bar(range(len(train_df), len(train_df) + len(validation_df)), validation_df['y'], color='r', label='validation')
plt.bar(range(len(train_df) + len(validation_df), len(train_df) + len(validation_df) + len(test_df)), test_df['y'], color='g', label='test')
plt.legend()
plt.show()