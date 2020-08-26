import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA
X = np.array([[-1, -1, 5], [-2, -1, 2], [-3, -2, 4], [1, 1, -1], [2, 1, 2], [3, 2, 2]])
ipca = IncrementalPCA(n_components=2, batch_size=3)
ipca.fit(X)

y = ipca.transform(X) # doctest: +SKIP
print(y)

kmeans = KMeans(n_clusters=4, random_state=0).fit(y)
kmeans.labels_

print(kmeans.predict([[-2.74841712, -0.03719462], [000, 0]]))

print(kmeans.cluster_centers_)
input("end")



""""
from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
kmeans.labels_

kmeans.predict([[0, 0], [12, 3]])


print(str(kmeans.cluster_centers_))
input("end")
#https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA""
"""
""""
import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(X)
print(pca.fit(X))
print(pca.explained_variance_ratio_)

print(pca.singular_values_)
"""