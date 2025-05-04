import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import seasborn as sns

# Task1
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target 
df['species'] = df['species'].map(dict(zip(range(3), iris.target_names)))

print(df.head())
print(df.info())
print(df.isnull().sum())

df.dropna(inplace=True)

# Task2 
print(df.describe())

grouped = df.groupby('species').mean()
print(grouped)

#  Pattern Analysis:
# Setosa has shorter petals compared to Versicolor and Virginica.
# Virginica generally has the highest average measurements across all features,
# while Versicolor lies between the two. These patterns show clear differences 
# between species based on size, useful for classification.


# Task3
df['index'] = df.index  # simulate time
df.sort_values(by='index', inplace=True)

plt.plot(df['index'], df['sepal length (cm)'])
plt.title('Sepal Length Over Index')
plt.xlabel('Index (Fake Time)')
plt.ylabel('Sepal Length (cm)')
plt.show()

df.groupby('species')['petal length (cm)'].mean().plot(kind='bar', color='skyblue')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()

df['petal length (cm)'].plot(kind='hist', bins=20, color='purple')
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.show()

sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Sepal vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.show()

