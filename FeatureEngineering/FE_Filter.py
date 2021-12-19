from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2
from scipy.stats import pearsonr



iris = load_iris()
# iris = pd.DataFrame(iris, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

data = iris.data
# target = iris.target
# target = pd.DataFrame(target, columns=['target'], dtype=np.int16)

# 1 - 特征选择
# 1.A 方差阈值（最简单的特征选择方法） -> 移除低方差的特征
# VarianceThreshold() <- threshold = 3
# Step 1: Calculate variance of each feature

data = VarianceThreshold(threshold=0.5).fit_transform(data)
print('After variance threshod:\n',data.shape)

# 1.B 单变量特征选择
# 1.B1 卡方检验（Chi2） - 适用于分类问题（y离散） - X, y都要用到
X, y = iris.data, iris.target
print('Before chi2: \n', X.shape)

X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
print('After chi2:\n', X_new.shape)

# 1.C 相关系数(Correlation) - 适用于回归问题（y连续）- X, y都要用到
np.random.seed(0)
size = 300
x = np.random.normal(0,1,size)
# pearsonr(x,y)为特征矩阵和目标向量
# output: (score, p-value)
print("Lower Noise:\n", pearsonr(x, x+np.random.normal(0,1,size)))
print("Higher Noise:\n", pearsonr(x, x+np.random.normal(0,10,size)))


# 1.D 互信息和最大信息系数 (Mutual information and maximal information coefficient (MIC))
# ...

# 1.E 距离相关系数
# https://zhuanlan.zhihu.com/p/74198735