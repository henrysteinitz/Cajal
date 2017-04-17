import numpy as np

# Some simple classes, mostly for documentation
class SmoothMap:
    def __init__(self, calc, gradient):
        self.calc = calc # f: R^n -> R^m (or f could act on a list of tensors
                         # with different shapes)
        self.gradient = gradient # g: R^n -> R^n * R^m ~ R^(n + m) (when f acts
                                 # on a single tensor)
    def __call__(self, args):
        return self.calc(args)

class Map:
    def __init__(self, calc):
        self.calc = calc

    def __call__(self, args):
        return self.calc(*args)

class Gradient:
    def __init__(self, calc):
        self.calc = calc

    def __call__(self, activations):
        return self.calc(*activations)


# Useful smooth maps
def l2_map(x, y): return np.sum(np.multiply((x - y), (x - y)))
def l2_grad(x, y): return np.array([2*(x - y), 2*(y - x)])
l2_norm = SmoothMap(Map(l2_map), Gradient(l2_grad))

# TODO Generalize to higher dimensions and any number of arguments
def matrix_product_map(x,y): return np.matmul(x, y)
def matrix_product_grad(x,y):
    dx = np.zeros(shape=(x.shape[0], x.shape[1], x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            dx[i, j, i] = y[j]
    dy = np.zeros(shape=(y.shape[0], y.shape[1], x.shape[0], y.shape[1]))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            dy[i, j, :, j] = x[:, i]
    return [dx, dy]
matrix_product = SmoothMap(Map(matrix_product_map), Gradient(matrix_product_grad))

def add_map(x, y): return x + y
def add_grad(x, y):
    dx = np.zeros(shape=(x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            dx[i, j, i, j] = 1
    dy = np.zeros(shape=(y.shape[0], y.shape[1]))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            dy[i, j, i, j] = 1
    return [dx, dy]
add = SmoothMap(Map(add_map), Gradient(add_grad))

def multiply_map(x, y): return np.multiply(x, y)
def multiply_grad(x, y):
    dx = np.zeros(shape=(x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            dx[i, j, i, j] = y[i, j]
    dy = np.zeros(shape=(y.shape[0], y.shape[1]))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            dy[i, j, i, j] = x[i, j]
    return [dx, dy]
add = SmoothMap(Map(add_map), Gradient(add_grad))


# def sigmoid_map(x):
# def sigmoid_grad(x):
#
# def tanh_map(x):
# def tanh_grad(x):
