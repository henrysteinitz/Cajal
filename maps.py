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
    dx = np.zeros((*x.shape, *np.matmul(np.zeros_like(x), np.zeros_like(y)).shape))
    for pos in np.ndindex(*x.shape):
        dx[(*pos, pos[0])] = y[pos[1]]
    dy = np.zeros((*y.shape, *np.matmul(np.zeros_like(x), np.zeros_like(y)).shape))
    if y.ndim == 1:
        for pos in np.ndindex(*y.shape):
            dy[(*pos,)] = x[:, pos[0]]
    else:
        for pos in np.ndindex(*y.shape):
            dy[pos[0], pos[1], :, pos[1]] = x[:, pos[0]]
    return [dx, dy]
matrix_product = SmoothMap(Map(matrix_product_map), Gradient(matrix_product_grad))

def matrix_vector_product_map(x, y): return np.matmul(x,y)
def matrix_vector_product_grad(x, y):
    dx = np.zeros(shape=(x.shape[0], x.shape[1], x.shape[0]))
    for pos in np.ndindex(*x.shape):
        dx[(*pos, pos[0])] = y[pos[1]]
    dy = np.zeros(shape=(y.shape[0], x.shape[0]))
    for i in range(y.shape[0]):
        dy[i] = x[:, i]
    return [dx, dy]
matrix_vector_product = SmoothMap(Map(matrix_vector_product_map), Gradient(matrix_vector_product_grad))

def add_map(x, y): return x + y
def add_grad(x, y):
    dx = np.zeros(shape=(*x.shape, *x.shape))
    for pos in np.ndindex(*x.shape):
        dx[(*pos, *pos)] = 1
    dy = np.zeros(shape=(*y.shape, *y.shape))
    for pos in np.ndindex(*y.shape):
        dy[(*pos, *pos)] = 1
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
multiply = SmoothMap(Map(multiply_map), Gradient(multiply_grad))

def sigmoid_map(x): return 1 / (1 + np.exp(-x))
def sigmoid_grad(x):
    dx = np.diag(sigmoid_map(x) * (1 - sigmoid_map(x)))
    return [dx]
sigmoid = SmoothMap(Map(sigmoid_map), Gradient(sigmoid_grad))

def soft_max_map(x):
    s = x - max(x)
    return np.exp(s) / sum(np.exp(s))
def soft_max_grad(x):
    s = x - max(x)
    exp = np.exp(s)
    grad = np.zeros((len(x), len(x)))
    d = (exp * (sum(exp) - exp)) / (sum(exp) * sum(exp))
    for i in range(len(s)):
        for j in range(len(s)):
            if i == j:
                grad[i][j] = d[i]
            else:
                grad[i][j] = ((-exp[i])*(exp[j])) / (sum(exp) * sum(exp))
    return [grad]
soft_max = SmoothMap(Map(soft_max_map), Gradient(soft_max_grad))


# def tanh_map(x):
# def tanh_grad(x):
