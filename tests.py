from smooth import Flow, Map

flow = Flow(['x'], ['y'])
flow.connect_variable('x', [], ['f'])
flow.connect_variable('y', ['f'], [])
flow.connect_map('f', lambda x: [3*x], ['x'], ['y'])
print(flow.compute({'x': 2}))
