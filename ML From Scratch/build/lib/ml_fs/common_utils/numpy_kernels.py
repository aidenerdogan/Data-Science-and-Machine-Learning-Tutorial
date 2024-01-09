import numpy as np

# This python file contains sapmle kernels by numpy

# Linear kernel
def linear_kernel(**kwargs):
    def f(x1, x2):
        return np.inner(x1, x2)
    return f

# Polynomial kernel
def polynomial_kernel(power, coef, **kwargs):
    def f(x1, x2):
        return (np.inner(x1, x2) + coef)**power
    return f

# Radial Bases Function kernel
def rbf_kernel(gamma, **kwargs):
    def f(x1, x2):
        distance = np.linalg.norm(x1 - x2) ** 2
        return np.exp(-gamma * distance)
    return f
