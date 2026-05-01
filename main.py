import pandas as pd
data=pd.read_csv("Mall_Customers.csv")
print(data.head())
print(data.info())
print(data.describe())

import matplotlib.pyplot as plt

#Visualization of Income vs Spending Score:
#--Customers with high income have a variable spending score
#--Similar goes for low income customers
#--Customers with income from 40k to 80k $ mostly have spending score b/w 40 and 60. A cluster is visible.
#--Groups visible - Low income, high spending; High income, high spending; High Income, low spending; Low income, low spending
plt.figure()
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'])
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("Income vs Spending")
plt.show()

from sklearn.cluster import KMeans

X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

import matplotlib.pyplot as plt
print("Reached elbow method")
plt.figure()
plt.plot(range(1, 11), wcss)
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

kmeans = KMeans(n_clusters=5, random_state=0)
data['Cluster'] = kmeans.fit_predict(X)

plt.figure()
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=data['Cluster'])
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("Customer Segments")
plt.show()

print("Insights:")

print("Cluster 0: High income, high spending → Premium customers")
print("Cluster 1: Low income, low spending → Low-value customers")
print("Cluster 2: High income, low spending → Target customers")
print("Cluster 3: Low income, high spending → Overspending customers")
print("Cluster 4: Average income and spending → Normal customers")

data.to_csv("segmented_customers.csv", index=False)




