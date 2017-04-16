import numpy as np

# Some simple classes, mostly for documentation
class SmoothMap:
    def __init__(self, calc, gradient):
        self.calc = calc
        self.gradient = gradient

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

# Some useful maps
def l2_map(y, y0): return (y - y0)*(y - y0)
def l2_grad(y, y0): return np.array([2*(y - y0), 2*(y0 - y)])
l2_norm = SmoothMap(Map(l2_map), Gradient(l2_grad))
