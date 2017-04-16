from flow import Flow
import numpy as np
import random
from maps import *

# Simple forward computation
flow = Flow(inputs=['x', 'h', 'b'], outputs=['y'])
flow.connect_variable('x', None, ['f'])
flow.connect_variable('x', None, ['h'])
flow.connect_variable('y', 'g', [])
flow.connect_variable('a', 'f', ['g'])
flow.connect_map('f', Map(lambda x,h: x*h), ['x', 'h'], 'a')
flow.connect_map('g', Map(lambda a,b: a-b), ['a', 'b'], 'y')

print(flow.play({'x': 2, 'h': 8, 'b': 3}))

# Linear Regression w/ randomly generated data
flow2 = Flow(inputs=['x'], outputs=['y'])
flow2.connect_parameter(name='w',
                       value=np.array([1.0,1.0]),
                       sinks=['f'])
flow2.connect_parameter(name='b', value=1.0, sinks=['f'])
flow2.connect_variable(name='x', sinks=['f'])
flow2.connect_variable(name='y', source='f')

def linear_map(w, x, b): return np.dot(w,x) + b
def linear_grad(w, x, b):
    w_grad = [x[0], x[1]]
    x_grad = [w[0], w[1]]
    b_grad = 1
    return np.array([w_grad, x_grad, b_grad])
f = SmoothMap(Map(linear_map), Gradient(linear_grad))

flow2.connect_map(name='f', map=f, sources=['w','x','b'], sink='y')

print(flow2.play({'x': [1,2]}))

random.seed()
inputs = [np.array([10*random.random(), 10*random.random()]) for _ in range(1000)]
outputs = [np.dot(x, [3,2]) + 2 for x in inputs]

flow2.set_loss(sources=['y'], smooth_scalar_map=l2_norm, supervisors=1)
flow2.train(inputs={'x': inputs}, outputs=outputs, learning_rate=.01)

print(flow2.play({'x': [1,2]}))
print(flow2.nodes['w'])


# Multilayer Perceptron
