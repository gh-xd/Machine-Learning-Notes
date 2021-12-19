# 特征工程

1. **数据探索性分析 Exploratory Data Analysis (EDA)**
    - 目的：了解数据大致情况
2. **预处理 Preprocessing**
    - 目的：使数据可以被机器学习
3. **特征选择 Feature Selection**
    - 目的：寻找重要的、相关的特征进行学习，使学习效果更好
4. **降维 Decomposition**
    - 目的：加快机器学习速度

## 1. EDA 数据探索性分析
拿到数据后，如果是Dataframe格式，一般可以利用pandas内置的函数，简单查看数据类型、分布、缺失等情况：
- 使用`df.info()`查看特征类型
- 使用`df.describe()`查看数值型数据的分布情况
- 使用`df.columns.values.unique()`查看特征名字/特征下的类别
- 使用`df.isna().all(axis=0/1)`查看整列/整行缺失的情况
- 使用`df.isna().any(axis=0/1)`查看某列/某行有缺失值的情况

:white_check_mark: **个人总结**
> 拿到数据可以先观察特征类型，数值型和类别型数据(categorical and numerical data)的处理方式和训练方法不一样。确认缺失行列/值所涉及的特征具体指的是什么，从而在预处理环节有针对性地对其进行处理。

## 2. Preprocessing 预处理 
- sklearn.processing
- sklearn.impute
- numpy.log1p

|类|功能|说明|
|----|----|----|
|StandardScaler|标准化|特征值服从标准正态分布|
|MinMaxScaler|标准化|区间缩放，特征值到[0,1]|
|Normalizer|归一化|BatchNorm，基于行，归一量纲 > 单位向量|
|Binarizer|二值化|按阈值划分二类|
|OneHotEncoder|热独编码|定性数据到定量数据|
|Imputer|缺失值处理|计算缺失值、填充缺失值|
|PolynomialFeatures|多项式数据转换|..|
|FUNCTIONTRANSFORMER|自定义单元数据转换|使用单变元函数转化数据|

## 2. 特征选择 Featuer Selection
数据预处理完以后，需要选择**有意义的特征**输入ML，一般会考虑：
- **特征自身分布**：特征是否发散？ 如果不发散（方差接近0），则特征差异极小，则对区分样本或许没有作用。
- **特征与目标的相关性**：高相关性或者不相关

三种特征选择方式：
1. Filter 过滤法
2. Wrapper 包装法
3. Embedded 嵌入法

|类|方法|说明|
|----|----|----|
|VarianceThreshold|方差阈值|移除方差小（分布差异小）的特征|
|SelectKBest, chi2|卡方检验|检验定性的自变量对定性的因变量的相关性。假设自变量有N种、因变量有M种取值，考虑自变量等于i且因变量等于j的样本频数的观察值与期望值的差距|
|scipy.stats.pearsonr|皮尔森相关系数|衡量变量之间的线性关系，结果的取值在[-1,1]之间，-1位完全负相关，1位完全正相关，0位不相关。缺点：只对线性关系敏感。如果非线性对应，相关性也还是0|
|f_regression, f_classif|批量皮尔森相关系数|...|
|...|互信息和最大信息系数|...|
|...|距离相关系数|为了克服Pearson相关系数的弱点出现：Pierson相关系数为0，并不代表不相关（可能非线性相关）。但是距离相关系数为0，那么这两个变量就是独立的|
|Recursive Feature Elimination(RFE)|包装法|递归地训练基模型，将权值系数较小的特征从特征集合中消除|
|SelectFromModel|嵌入法|训练基模型，选择权值系数较高的特征|

继续
https://blog.csdn.net/MrLevo520/article/details/78085650


### 包装法 Wrapper
基于hold-out方法，对每一个待选的特征子集，都在训练集上训练一遍模型

然后在测试集上根据误差大小，选择出特征子集。

算法选择，普遍效果较好的：Random Forest，SVM，kNN等。

贪婪搜索是局部最优算法，穷举算法（exhaustive search）计算复杂度太高

#### 前向搜索
从0增加到最优集：每次从剩下的特征中选出一个，加入特征集。达到阈值或者n，从所有的特征中选出错误率最小的。
1. 初始化特征集F为空
2. 从1到n，如果第i个特征不在F中，特征i和F放在一起作为F_i。在值使用F_i中的特征的情况下，利用**交叉验证**来得到F_i的错误率
3. 从第2步得到n个F_i中错误率最小的F_i，更新F_i
4. 如果F中的特征树达到了n或者设置的阈值，返回最好的特征

#### 后向搜索
从n减少到最优集：每次删除一个特征


#### 递归特征消除法
使用**基模型**进行多轮训练，每轮训练后返回**相关系数**或者**特征重要性**，排除权重较低的特征，再基于新的特征集进行下一轮训练
基模型例如：
1. 逻辑回归模型
2. 支持向量机
    - Support Vector Machine (SVM) 支持向量机
    - Support Vector Classification (SVC) 用于分类的支持向量机
    - Support Vector Regression (SVR) 使用回归的支持向量机

SVM的一些模型
- svm.LinearSVC
- svm.LinearSVR
- svm.NuSVC (Nu-Support)
- svm.NuSVR
- svm.OneClassSVM (unsupervised Outlier Detection)
- svm.SVC
- svm.SVR


### 嵌入法 Embedded
类似于Filter方法，但是通过训练，来确定特征值的优劣
- 使用某些机器学习的算法和模型进行训练
- 得到各特征的权值系数
- 根据系数从大到小选择特征

#### 基于惩罚项的特征选择法
使用带惩罚项的**基模型**，除了筛特征值，同时也降维

例如feature_selection库的SelectFromModel类结合带L1惩罚项的逻辑回归模型

##### 惩罚项降维
L1惩罚项原理是，保留多个对目标值具有同等想关心的特征值中的一个

因为只保留一个，并不代表没选到的特征不重要，所以，

要用L2进行交叉验证：若一个特征在L1中的权值为1，选择在L2中**权值差别不大**并且**在L1中权值为0**的特征，构成集合

将这一集和中的特征平分L1中的权值，所以需要构建一个信的逻辑回归模型

### 降维
特征矩阵可能过大，计算量过大，训练时间太长，所以降维游泳。

常见的降维方法：基于L1惩罚项的模型，还有PCA，LDA等。LDA本身也是一个分类模型。

PCA和LDA的映射目标不一样：
- PCA是为了让映射后的样本具有最大的发散性（无监督）
- LDA是为了让映射后的样本具有最好的分类性能（有监督）
#### 主成分分析 PCA
#### 线性判别分析 LDA


## 参考
除了自己的总结，专业内容和代码相关的信息基本来自与以下文章（最后打开日期：2021-12-19）。其中有部分python库的引入和代码不能正常使用：
- [总结：sklearn机器学习之特征工程](https://blog.csdn.net/MrLevo520/article/details/78085650)
- [机器学习特征选择(Feature Selection)方法汇总](https://zhuanlan.zhihu.com/p/74198735)
- [Comparison of LDA and PCA 2D projection of Iris dataset](https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-lda-py)