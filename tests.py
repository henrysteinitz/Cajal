from smooth import Flow, Map

flow = Flow(['x', 'h', 'b'], ['y'])
flow.connect_variable('x', [], ['f'])
flow.connect_variable('x', [], ['h'])
flow.connect_variable('y', ['g'], [])
flow.connect_variable('a', ['f'], ['g'])
flow.connect_map('f', lambda x,h: [x*h], ['x', 'h'], ['a'])
flow.connect_map('g', lambda a,b: [a-b], ['a', 'b'], ['y'])
print(flow.compute({'x': 2, 'h': 8, 'b': 3}))
