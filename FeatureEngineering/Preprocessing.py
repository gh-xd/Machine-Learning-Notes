from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, Binarizer, OneHotEncoder, PolynomialFeatures, FunctionTransformer
from sklearn.impute import SimpleImputer
from numpy import log1p



iris = load_iris()

# iris = pd.DataFrame(iris, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

data = iris.data
# target = iris.target
# target = pd.DataFrame(target, columns=['target'], dtype=np.int16)


# 1 - 数据预处理
# 1.A - 0均值标准化(Z-score standardization) --> 转化为单位向量（均值为0，方差为1）
# StandardScaler()
data = pd.DataFrame(data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], dtype=np.float32)
print('Before standardization\n',data.describe())

data = iris.data
data = StandardScaler().fit_transform(data)
data = pd.DataFrame(data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], dtype=np.float32)

print('After Z-score standardization\n',data.describe())

# 1.B - 线性函数归一化 Min-Max Scaling --> 缩放尺寸到（0，1）
# MinMaxScaler()
data = iris.data
data = MinMaxScaler().fit_transform(data)
data = pd.DataFrame(data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], dtype=np.float32)

print('After Min-Max scaling\n',data.describe())

# 1.C - 归一化 Normalizer --> 同一行不同特征的量纲进行规范 -> 例如1米和10000元，拉近(BatchNorm)
# Normalizer()
data = iris.data
data = Normalizer().fit_transform(data)
data = pd.DataFrame(data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], dtype=np.float32)

print('After normalizing\n',data.describe())

# 1.D - 二值化处理，0 <= threshold <= 1 --> 只有0和1的列
# Binarizer()
data = iris.data
data = Binarizer(threshold=3).fit_transform(data)
data = pd.DataFrame(data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], dtype=np.float32)

print('After Binarization\n',data.head(10))

# 1.E - 热独编码
# OneHotEncoder().fit_transform(xx)
# 这里全是数字，不展示了

# 1.F - 缺失值计算
# Imputer() 

# 手动添加缺失值
data = np.array([np.nan, 2, 6, np.nan, 7, 6]).reshape(3,2)
print('Before imputing\n', data)
# Imputer() - 使用均值进行填充
Imp = SimpleImputer(missing_values=np.nan, strategy='mean')
data = Imp.fit_transform(data)
print('After imputing\n', data)


# 1.F - 数据变换 -> 基于多项式、指数函数、对数函数的
# PolynomialFeatures()
data = iris.data
print('Before polynomial\n', data.shape)

# 4个特征，度为2: (x1, x2, ...) --> (1, x1, x2, x1^2, x2^2, ...)
data = PolynomialFeatures().fit_transform(data)
# data = pd.DataFrame(data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], dtype=np.float32)

print('After polynomial\n', data.shape)

# 1.G - 单变元 - 对数函数的数据变换(log1p)
# FunctionTransformer() <- log1p
data = iris.data
data = FunctionTransformer(log1p).fit_transform(data)
data = pd.DataFrame(data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], dtype=np.float32)


print('After log transforming\n', data)
