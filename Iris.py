import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("1) iris.csv")

df.isnull().sum()

df.duplicated().sum()

df = df.drop_duplicates()

df.columns = df.columns.str.lower()

print(df.describe())

print(df.median(numeric_only=True))
print(df.mode())


df.hist(figsize=(10,6))
plt.show()

sns.boxplot(data=df)
plt.show()

sns.scatterplot(x='sepal_length', y='petal_length', hue='species', data=df)
plt.show()

df.corr(numeric_only=True)
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()