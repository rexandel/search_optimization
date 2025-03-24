import autograd.numpy as np
from autograd import grad

from functools import lru_cache

class GradientHelper:
    def __init__(self, function):
        self.function = function
        self._grad_x = grad(self.function, argnum=0)
        self._grad_y = grad(self.function, argnum=1)

    @lru_cache(maxsize=None)
    def compute(self, x, y):
        return np.array([self._grad_x(x, y), self._grad_y(x, y)])
