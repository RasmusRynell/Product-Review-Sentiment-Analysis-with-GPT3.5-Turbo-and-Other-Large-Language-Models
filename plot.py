import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Read from results.csv
results = pd.read_csv('results.csv')

print(results)

# Create new names from tokenizer and model
results['name'] = results['tokenizer'] + ' ' + results['model']


sns.set_theme(style="whitegrid")

# Plot with seaborn, barplot
ax = sns.barplot(x="name", y="test_acc", data=results)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()
