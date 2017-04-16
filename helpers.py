import numpy as np

def sum_product(x,y):
    x = np.array(x)
    y = np.array(y)
    return (x*y).sum(axis=tuple(range(x.ndim - y.ndim, x.ndim)))
