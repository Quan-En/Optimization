
import numpy as np
from numpy import array
from numpy.linalg import det
from numpy.random import rand
from math import exp
import random
from tqdm import tqdm

# main objective function
class main_obj_function(object):
    def __init__(self, theta=[0.05884, 4.298, 21.8]):
        self.theta = theta
    
    def sub_obj_function(self, x, w):
        value1 = x * self.theta[2] * exp(-self.theta[0] * x)
        value2 = x * self.theta[2] * exp(-self.theta[1] * x)
        value3 = exp(-self.theta[1] * x) - exp(-self.theta[0] * x)
        value = array([value1, value2, value3]).reshape(-1,1)
        return w * (value @ value.T)
    
    def forward(self, xs_and_ws):
        xs = xs_and_ws[:3]
        ws = xs_and_ws[3:]
        result = list(map(lambda x, w: self.sub_obj_function(x,w), xs, ws))
        result = det(np.sum(np.stack(result), axis=0))
        return result

## Softmax function
def Softmax(x):
    return np.exp(x) / np.sum(np.exp(x))