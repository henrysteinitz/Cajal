import numpy as np
from helpers import sum_product

NODE_KINDS = ['MAP', 'CONSTANT', 'VARIABLE', 'PARAMETER']

class Flow:

    nodes = {}
    gradients = {}
    kinds = {}

    sources = {}
    sinks = {}

    inputs = set([])
    outputs = set([])
    parameters = set([])
    constants = set([])

    loss_sources = []
    def loss_map(*args): return 0

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

        if kind == 'PARAMETER':
            self.parameters.add(name)
        elif kind == 'CONSTANT':
            self.constants.add(name)

    def connect_map(self, name, map, sources, sink):
        self.connect('MAP', name=name, value=map, sources=sources, sinks=[sink])

    def connect_constant(self, name, value, sinks=None):
        self.connect('CONSTANT', name=name, value=value, sinks=sinks)

    def connect_variable(self, name, source=None, sinks=None):
        self.connect('VARIABLE', name=name, sources=[source], sinks=sinks)

    def connect_parameter(self, name, value=None, sinks=None):
        self.connect('PARAMETER', name=name, value=value, sinks=sinks)

    def set_inputs(self, inputs):
        self.inputs = set(inputs)

    def set_outputs(self, outputs):
        self.outputs = set(outputs)

    def set_loss(self, sources, scalar_map):
        self.loss_sources = sources
        self.loss_map = scalar_map


    def play(self, input_values={}, outputs=None):
        outputs = outputs or self.outputs
        remaining_outputs = set(outputs)
        traversed = set(input_values.keys()).union(self.parameters).union(self.constants)
        for inp in input_values:
            self.nodes[inp] = input_values[inp]

        while remaining_outputs:
            target = self.__choose(remaining_outputs)
            active_map = self.__first_node_before(target, traversed)
            active_sources = self.sources[active_map]
            active_sink = self.sinks[active_map][0]
            source_activations = [self.nodes[source] for source in active_sources]
            sink_activation = self.nodes[active_map](*source_activations)
            print(source_activations)
            self.nodes[active_sink] = sink_activation
            traversed.add(active_map)
            traversed.add(active_sink)
            remaining_outputs.discard(active_sink)

        return {out: self.nodes[out] for out in outputs}


    def backpropagate(self, parameters=None):
        self.__reset_gradients()
        parameters = parameters or self.parameters
        remaining_parameters = set(parameters)
        for i, source in enumerate(self.loss_sources):
            #missing shit
            self.gradients[source] = self.loss_map.gradient(self.loss_sources)[i]
        traversed = set(self.loss_sources).copy()

        while remaining_parameters:
            target = self.__choose(remaining_parameters)
            active_map = self.__last_node_after(target, traversed)
            active_sources = self.sources[active_map]
            active_sink = self.sinks[active_map][0]
            sink_gradient = self.gradients[active_sink]
            source_activations = [self.nodes[source] for source in active_sources]
            source_gradients = self.nodes[active_map].gradient(source_activations)

            for i, source in enumerate(active_sources):
                self.gradients[source] += sum_product(source_gradients[i], sink_gradient)
                if self.__all_sinks_traversed(source, traversed):
                    traversed.add(source)
            traversed.add(active_map)

        return {param: self.gradients[param] for param in parameters}

    # private helpers
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

    def __all_sinks_traversed(self, node, traversed):
        for sink in self.sinks[node]:
            if sink not in traversed:
                return False
        return True

    def __reset_gradients(self):
        self.gradients = {x: 0 for x in self.gradients}

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
        return self.map(*args)


class Gradient:

    def __init__(self, calc=lambda x,y: 0):
        self.calc = calc

    def __call__(activation, error):
        self.calc(activation, error)

    def calc(activation, error):
        raise NotImplementedError

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
