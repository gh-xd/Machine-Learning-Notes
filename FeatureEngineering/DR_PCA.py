from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

iris = load_iris()
category = iris.target

pca = PCA(n_components=2) # 降到2维

print("before PCA", iris.data.shape)
x_pca = pca.fit_transform(iris.data)
print(x_pca.shape)

plt.scatter(x_pca[:,0], x_pca[:, 1], c=category)
plt.show()