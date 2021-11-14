import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


class PCA():

    def calculate_covariance_matrix(self, X):

        # X.shape = (1797, 64)，1797 samples and 64 features
        m = X.shape[0]

        # X minus mean value
        X = X - np.mean(X, axis=0)

        # covariance matrix of X
        return 1 / m * np.matmul(X.T, X)

    def pca(self, X, n_components):

        # with X, first calculate covariance
        covariance_matrix = self.calculate_covariance_matrix(X)

        # using numpy to calculate eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        print(eigenvectors.shape)

        # sort the eigenvalues, argsort sorts elements ascending then retrieve corresponding index
        idx = eigenvalues.argsort()[::-1]

        # get the first n_components of vectors
        eigenvectors = eigenvectors[:, idx]
        eigenvectors = eigenvectors[:, :n_components]

        # calculate the result
        Y = np.matmul(X, eigenvectors)
        return Y


# load dataset from sklearn
data = datasets.load_digits()
X = data.data
y = data.target

# reduce the original diemensions to 2
pca = PCA().pca(X, 2)
x1 = pca[:, 0]
x2 = pca[:, 1]

# plot
cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]

class_distr = []

for i, l in enumerate(np.unique(y)):
    _x1 = x1[y == l]
    _x2 = x2[y == l]
    _y = y[y == l]
    class_distr.append(plt.scatter(_x1, _x2, color=colors[i]))

plt.legend(class_distr, y, loc=1)

plt.suptitle("PCA Dimensionality Reduction")
plt.title("Digit Dataset")
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

