from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC



iris = load_iris()
# iris = pd.DataFrame(iris, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

data = iris.data
# target = iris.target
# target = pd.DataFrame(target, columns=['target'], dtype=np.int16)

# 1 递归特征消除法 Recursive Feature Elimination
# 1.A - 基模型：逻辑回归
rfe_lr = RFE(estimator=LogisticRegression(), n_features_to_select=2)
rfe_lr.fit_transform(iris.data, iris.target)
print('RFE based on Logistic Regression', rfe_lr.ranking_)

# 1.B - 基模型：支持向量机
svc = SVC(kernel="linear", C=1)
rfe_svc = RFE(estimator=svc, n_features_to_select=2, step=1)
rfe_svc.fit(iris.data, iris.target)
print('RFE based on SVC > ', rfe_svc.ranking_)
