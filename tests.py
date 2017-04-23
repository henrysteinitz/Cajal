from cajal.flow import Flow
import numpy as np
import random
from cajal.maps import *
import unittest


class CajalTests(unittest.TestCase):

    # Simple forward computation
    def test_forward_computation(self):
        flow = Flow(inputs=['x', 'h', 'b'], outputs=['y'])
        flow.connect_variable('x', None, ['f'])
        flow.connect_variable('x', None, ['h'])
        flow.connect_variable('y', 'g', [])
        flow.connect_variable('a', 'f', ['g'])
        flow.connect_map('f', Map(lambda x,h: x*h), ['x', 'h'], 'a')
        flow.connect_map('g', Map(lambda a,b: a-b), ['a', 'b'], 'y')
        result = flow.play({'x': 2, 'h': 8, 'b': 3})
        self.assertEqual(result, {'y': 13})

    # Linear regression on random data
    def test_linear_regression(self):
        flow2 = Flow(inputs=['x'], outputs=['y'])
        # connect variables & parameters
        flow2.connect_parameter(name='w',
                                value=np.array([1.0,1.0]),
                                sinks=['f'])
        flow2.connect_parameter(name='b', value=1.0, sinks=['f'])
        flow2.connect_variable(name='x', sinks=['f'])
        flow2.connect_variable(name='y', source='f')
        # define & connect maps
        def linear_map(w, x, b): return np.dot(w,x) + b
        def linear_grad(w, x, b):
            w_grad = [x[0], x[1]]
            x_grad = [w[0], w[1]]
            b_grad = 1
            return np.array([w_grad, x_grad, b_grad])
        f = SmoothMap(Map(linear_map), Gradient(linear_grad))
        flow2.connect_map(name='f', map=f, sources=['w','x','b'], sink='y')
        result = flow2.play({'x': [1,2]})
        self.assertAlmostEqual(4.0, result['y'])
        # generate random data
        random.seed()
        inputs = [np.array([10*random.random(), 10*random.random()]) for _ in range(3000)]
        outputs = [np.dot(x, [3,2]) + 2 for x in inputs]
        # set loss map & train
        flow2.set_loss(sources=['y'], scalar_map=l2_norm, supervisors=1)
        flow2.train(inputs={'x': inputs}, outputs=outputs, learning_rate=.01)
        result = flow2.play({'x': [1,2]})
        self.assertAlmostEqual(9.0, result['y'], places=2)

    # Learn a matrix multiplication paramter
    def test_matrix_product(self):
        # build graph
        product_flow = Flow(inputs=['X'], outputs=['Y'])
        product_flow.connect_variable(name='X', sinks=['f'])
        product_flow.connect_variable(name='Y', source='f')
        product_flow.connect_parameter(name='W',
                                       value=np.array([[1.0,1.0], [1.0,1.0]]),
                                       sinks=['f'])
        product_flow.connect_map(name='f',
                                 map=matrix_product,
                                 sources=['W','X'],
                                 sink='Y')
        # test forward pass
        result = product_flow.play({'X': np.array([[1,2,3], [2,3,4]])})
        list_Y = [list(row) for row in list(result['Y'])]
        self.assertAlmostEqual(list_Y, [[3,5,7], [3,5,7]])
        # generate data
        inputs = [np.random.rand(2,3) for _ in range(2000)]
        real_W = np.array([[2.0,3.0],[5.0,4.0]])
        outputs = [np.matmul(real_W, inp) for inp in inputs]
        # test training
        product_flow.set_loss(sources=['Y'], scalar_map=l2_norm, supervisors=1)
        product_flow.train(inputs={'X': inputs}, outputs=outputs, learning_rate=.01)
        W = product_flow.nodes['W']
        self.assertAlmostEqual(W[0,0], 2.0, places=3)
        self.assertAlmostEqual(W[0,1], 3.0, places=3)
        self.assertAlmostEqual(W[1,0], 5.0, places=3)
        self.assertAlmostEqual(W[1,1], 4.0, places=3)



    # def test_flow_composition(self):
    #
    # def test_define_syntax(self):


if __name__ == '__main__':
    unittest.main()
