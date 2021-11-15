# Regularization

Purpose: avoid overfitting

Overfitting: model fits data too much / model learns noise too much

Noise: data points which don't really represent the true properties

Overfitting avoidance approaches:
- cross validation
- **regularization**

---

### Ridge Regression (L2 Norm)


Loss function = original Loss function + **penalty**

**penalty** = tunning parameter and coefficient (square)

Ridge regression technique prevents **coefficients** from rising too high


### Lasso (L1 Norm)
Loss function = original Loss function + **penalty**

**penalty** = tunning parameter and coefficient (modulus)


### Dropout
Dropout can suppress learning

---

### Summary
Regularization reduces the *variance* of the model, without substantial increase in its bias


### References
[Regularization in Machine Learning](https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a)

More regularization methods:
[Regularization](https://paperswithcode.com/methods/category/regularization)