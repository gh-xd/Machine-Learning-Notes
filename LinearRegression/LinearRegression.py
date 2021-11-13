from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt


# load data from sklearn
diabetes = load_diabetes()

# get diabetes features, shape (442,10)
data = diabetes.data

# get label (diabete or not), shape (442,)
labels = diabetes.target

# shuffle data for splitting train/test set
X, y = shuffle(data, labels, random_state=5)

# set train/test ratio
ratio = 0.9
split_index = int(X.shape[0] * ratio)

# split data into train and test set
X_train, y_train = X[:split_index], y[:split_index]
X_test, y_test = X[split_index:], y[split_index:]

# reshape label from (442,) to (442,1)
y_train, y_test = y_train.reshape((-1, 1)), y_test.reshape((-1, 1))


# define loss function of linear regression
def calculate_loss(X, y, w, b):
    """
    :param X: train data
    :param y: train label
    :param w: weights to be updated
    :param b: weight to be updated
    :return:
    """

    num_input = X.shape[0]

    # prediction value
    y_hat = np.dot(X, w) + b

    # mean squared error (MSE) as loss function
    loss = np.sum((y_hat - y)**2) / num_input

    # calculate partial derivative
    dw = np.dot(X.T, (y_hat - y)) / num_input
    db = np.sum((y_hat - y)) / num_input

    return y_hat, loss, dw, db


# parameter random initialization
def param_random_init(dims):
    w = np.zeros(dims,1)
    b = 0
    return w, b