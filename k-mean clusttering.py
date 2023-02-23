import matplotlib.pyplot as p
import pandas as pd
from sklearn.cluster import KMeans

data = pd.DataFrame(([[1.0,1.0],[1.5,2.0],[3.0,4.0],[5.0,7.0],[3.5,5.0],[4.5,5.0],[3.5,4.5]]),columns = {'A','B'}, index = range(1,8))
X = data.values
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

p.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'cluster-1')
p.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'cluster-2')
p.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = 'green', label = 'centroid')
p.xlabel('A')
p.ylabel('B')
p.legend()
p.show()