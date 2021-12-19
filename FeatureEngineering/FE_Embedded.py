from sklearn.datasets import load_boston
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor

boston = load_boston()
# iris = pd.DataFrame(iris, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

X, y = boston.data, boston.target
names = boston.feature_names

# L1
# lsvc = LinearSVC(C=0.01, penalty="l1", dual=False) # 惩罚项为L1
# lsvc = lsvc.fit(X, y)
# model = SelectFromModel(lsvc, prefit=True)
# X_new = model.transform(X)
# print(X_new[:10,:])

# Boston Housing price
# n_estimators 为森林中树木数量，max_depth树的最大深度
rf = RandomForestRegressor(n_estimators=20, max_depth=4)
scores = []
for i in range(X.shape[1]):
    # 每次选择一个特征，交叉验证，比例7：3
    # SchuffleSplit()用于随机抽样
    cv = ShuffleSplit(n_splits=len(X), test_size=0.3, random_state=3)
    score = cross_val_score(rf, X[:, i:i+1], y, scoring="r2", cv=cv)
    scores.append((round(np.mean(score), 3), names[i]))

print(sorted(scores, reverse=True))
