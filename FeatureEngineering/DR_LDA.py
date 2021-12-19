from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target

lda = LinearDiscriminantAnalysis(n_components=2)

X_r2 = lda.fit(X, y).transform(X)

plt.scatter(X_r2[:, 0], X_r2[:, 1], c=target_names)
plt.show()