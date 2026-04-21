import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("1) iris.csv")

X = df.drop('species', axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters

sns.scatterplot(
    x=df['sepal_length'],
    y=df['petal_length'],
    hue=df['Cluster'],
    palette='viridis'
)

plt.title("K-Means Clustering")
plt.show()

sns.scatterplot(
    x=df['sepal_length'],
    y=df['petal_length'],
    hue=df['species']
)

plt.title("Actual Species")
plt.show()