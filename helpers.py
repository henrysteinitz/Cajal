def sum_product(x,y):
    return (x*y).sum(axis=tuple(range(y.ndim - x.ndim, yndim)))
