import numpy as np

NODE_TYPES = ['MAP', 'CONSTANT', 'VARIABLE', 'PARAMETER']

class Flow:

    nodes = {}
    kinds = {}

    sources = {}
    sinks = {}

    inputs = set([])
    outputs = set([])

    def __init__(self, inputs=[], outputs=[]):
        self.inputs = set(inputs)
        self.outputs = set(outputs)
        # self.nodes.update({x: None for x in inputs})
        # self.nodes.update({y: None for y in outputs})

    def connect(self, kind, **kwargs):
        name = kwargs['name']
        value = kwargs.get('value', None)
        sources = kwargs.get('sources', [])
        sinks = kwargs.get('sinks', [])

        self.nodes[name] = value
        self.kinds[name] = kind
        self.sinks[name] = sinks
        self.sources[name] = sources

    def connect_map(self, name, map, sources, sinks):
        self.connect('MAP', name=name, value=map, sources=sources, sinks=sinks)

    def connect_constant(self, name, value, sinks=[]):
        self.connect('CONSTANT', name=name, value=value, sinks=sinks)

    def connect_variable(self, name, sources=[], sinks=[]):
        self.connect('VARIABLE', name=name, sources=sources, sinks=sinks)

    def connect_parameter(self, name, value=None, sinks=[]):
        self.connect('PARAMETER', name=name, value=value, sinks=sinks)

    def compute(self, input_values = {}):
        remaining_outputs = self.outputs.copy()
        traversed = set(input_values.keys())
        for inp in input_values:
            self.nodes[inp] = input_values[inp]

        while remaining_outputs:
            target = self.__choose(remaining_outputs)
            active_map = self.__first_node_before(target, traversed)
            active_sources = [self.nodes[source] for source in self.sources[active_map]]
            active_sinks = self.nodes[active_map](*active_sources)
            traversed.add(active_map)
            for i, sink in enumerate(self.sinks[active_map]):
                traversed.add(sink)
                self.nodes[sink] = active_sinks[i]
                remaining_outputs.discard(sink)

        return {out: self.nodes[out] for out in self.outputs}

    def __first_node_before(self, node, traversed):
        for source in self.sources[node]:
            if source not in traversed:
                return self.__first_node_before(source, traversed)
        return node

    def __last_node_after(self, node, traversed):
        for sink in self.sinks[node]:
            if sink not in traversed:
                return self.__last_node_after(sink, traversed)
        return node

    @staticmethod
    def __choose(collection):
        target = collection.pop()
        collection.add(target)
        return target

class Map:

    def __init__(self, map, gradient = None):
        self.map = map
        self.gradient = gradient

    def __call__(self, *args):
        self.map(*args)


        # inputs = { 'X': [1, 1, 0, 0, 1] }

# class Cell:
#
#     def gradient(self, error_tensor):
#
#     def __call__(self, tensor):
#
# class FeedforwardCell(Cell):
#
#     def gradient(self, error_tensor):
#
#     def __call__(self, tensor):
#
# class RecurrentCell(Cell):
