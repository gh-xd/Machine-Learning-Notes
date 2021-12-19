# EDA - ExploratoryDataAnalysis-
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np


iris = load_iris()

# iris = pd.DataFrame(iris, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

data = iris.data
data = pd.DataFrame(data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], dtype=np.float32)
target = iris.target
target = pd.DataFrame(target, columns=['target'], dtype=np.int16)

# print(data.head())
# print(data.shape)

# print(target.head())
# print(target.shape)

# 1. Exploratory Data Analysis (EDA)

print(data.describe())
print(target.describe())

# 2. Classes
print(target['target'].unique())

# 3. vis
data.boxplot()

data.plot(kind='density')


