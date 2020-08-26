from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA
import numpy as np
import csv

data = []

""""
for row in csv:
    data.append([row[0], row[10].....])


with open('wage_data.csv', newline='') as csvfile:
    wage_data = csv.DictReader(csvfile)
    for row in wage_data:
      if user_country in row['LOCATION']:
        if user_time in row['TIME']:
          wageGap = float(row['Value'])
          print("Wagegap:", wageGap, "percent")
        else:
          otherTimes.append(row['TIME'])
      else:
        continue
"""
with open("hygdata_v3.csv", newline = "") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    i = 0
    for row in csv_reader:
        print(i)
        i += 1
try:
    X = np.array(data)
    ipca = IncrementalPCA(n_components=2, batch_size=3)
    ipca.fit(X)

    y = ipca.transform(X) # doctest: +SKIP
    print(y)

    kmeans = KMeans(n_clusters=4, random_state=0).fit(y)
    kmeans.labels_

    print(kmeans.predict([[-2.74841712, -0.03719462], [000, 0]]))

    print(kmeans.cluster_centers_)
    input("end")
except:
    print("finsh csv reader")













""""
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
kmeans.labels_

kmeans.predict([[0, 0], [12, 3]])
print(4E-8 * 10)

print(kmeans.cluster_centers_)
input("end")
#https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA """