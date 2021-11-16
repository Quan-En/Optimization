
import numpy as np
from numpy import array
from math import pi

# argmin function
def argmin(lst):
    return min(range(len(lst)), key=lst.__getitem__)

# Ackley function
## Input: array(D, ) -> float
def ackley_f(x):
    x = x.reshape(-1)
    lhs = 20 * np.exp((-0.2) * np.sqrt(np.mean((x ** 2))))
    rhs = np.exp(np.mean(np.cos(2 * pi * x)))
    result = -1 * (lhs + rhs) + 20 + np.exp(array([1]))
    return result.item()

# Weierstrass function
## Input: array(D, ) -> float
def weierstrass_f(x, a=0.5, b=3, kmax=20):
    x = (x * 0.5) / 100
    x = x.reshape(-1)
    D = x.shape[0]

    lhs = 0
    rhs = 0
    for k in range(kmax):
        lhs += np.sum((a ** k) * np.cos(2 * pi * (b ** k) * (x + 0.5)))
        rhs += (a ** k) * np.cos(array([pi * (b ** k)]))
    result = lhs - D * rhs + 600
    return result.item()