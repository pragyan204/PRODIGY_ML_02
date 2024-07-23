import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load dataset
Mall_Customer_path= r"C:\Users\PRAGYAN\Desktop\ProdigyProjects\Mall_customers"
df = pd.read_csv('Mall_Customer_path')

print(df.head())

# check for missing values
print(df.isnull().sum())

# drop null values
df = df.dropna()

# select features for clustering
features = ['total_spent', 'num_purchases', 'avg_purchase_value', 'purchase_frequency', 'product_categories']
X = df[features]

# normalize the feature data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# determine the optimal number of clusters using the Elbow Method
from sklearn.cluster import KMeans
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# plot the Elbow Method
plt.figure(figsize=(10, 5))
plt.plot(K, inertia, marker='o', linestyle='--', color='b')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Clusters')
plt.show()

# from the Elbow Method, assume the optimal number of clusters is 3
optimal_clusters = 3

# apply K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# analyze the clusters
cluster_summary = df.groupby('Cluster').mean()
print(cluster_summary)

# visualize the clusters
sns.pairplot(df, hue='Cluster', vars=features, palette='Set1')
plt.show()