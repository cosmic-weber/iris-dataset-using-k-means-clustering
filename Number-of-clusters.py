import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans

iris = datasets.load_iris()

samples = iris.data

num_clusters = list(range(1,9))
inertias = []

for i in num_clusters:
  model = KMeans(i)
  model.fit(samples)
  inertias.append(model.inertia_)

print(inertias)

# [681.3705999999996, 152.34795176035797, 78.851441426146, 57.350880212954756, 46.44618205128204, 39.03998724608725, 34.47097835497838, 30.064593073593088]

plt.plot(num_clusters, inertias, '-o')
 
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
 
plt.show()
