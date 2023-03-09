import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

d = pd.read_csv('D:/MLA/Wine.csv')
X = d.iloc[:, 0:13].values
sc = StandardScaler()
X = sc.fit_transform(X)
pca = PCA(n_components = 3)
X = pca.fit_transform(X)
ev = pca.explained_variance_
evr = pca.explained_variance_ratio_

print(ev)
print(evr)