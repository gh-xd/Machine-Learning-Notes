# Linear Regression

## Theory



## Practice

5 steps to realize a simple linear regression

***


Step 1 - Import necessary libraries

```python
from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
```

Step 2 - Load an examplary dataset

```python
# load data from sklearn
diabetes = load_diabetes()

# get diabetes features, shape (442,10)
data = diabetes.data

# get label (diabete or not), shape (442,)
labels = diabetes.target
```

Step 3 - Make train and test data

```python
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

```

Step 4 - Define forward and backward propagation

```python

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


# parameters random initialization
def param_random_init(dims):
    w = np.random.normal(0, 0.1, size=(dims, 1)) # (442, 1)
    b = 0
    return w, b


def train(X, y, lr, epochs):
    # set random initial parameters
    feature_length = X.shape[1]
    w, b = param_random_init(feature_length)

    # calculate prediction, loss and partial derivatives
    loss_list = []
    for i in range(1, epochs):

        # calculate prediction(y_hat), loss, dw, db in each epochs
        y_hat, loss, dw, db = calculate_loss(X, y, w, b)

        # save loss
        loss_list.append(loss)

        # update parameters with gradient descent
        w += -lr * dw
        b += -lr * db

        # print epochs and loss
        if i % 500 == 0:
            print(f'epoch %d - loss %.4f'%(i, loss))

        # save parameters
        params = {
            'w':w,
            'b':b
        }

        # save gradients
        grads = {
            'dw':dw,
            'db':db
        }

    return loss_list, loss, params, grads

```

Step 5 - Train the model and predict on test data

```python

# fit data on linear regression model
loss_list, loss, params, grads = train(X_train, y_train, 0.5, 50000)


# function for test data
def predict(X, params):

    # the parameters updated by the last epoch
    w = params['w']
    b = params['b']

    y_predict = np.dot(X, w) + b

    return y_predict

y_predict = predict(X_test, params)

```