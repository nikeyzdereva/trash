import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('dataset/Mall_Customers.csv')
dataset.head()

X = dataset.iloc[:, [ 3, 4]].values


from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
print(wcss)

clust_list = []
prop = []
for i in range(len(wcss) - 1):
    print(wcss[i])
    if i == 0:
        pass
    elif i > 0:
        prop.append(wcss[i+1]/wcss[i])
        prop.sort()
        print(prop)
        clust_list = round(prop[-1])
print(clust_list)
opt_num_clust = round(clust_list)
print(opt_num_clust)
