import numpy as np
from cajal.helpers import sum_product
from cajal.maps import *

NODE_KINDS = ['MAP', 'CONSTANT', 'VARIABLE', 'PARAMETER']

class Flow:

    def loss_map(*args): return 0

    def __init__(self, inputs=[], outputs=[]):
        self.inputs = set(inputs)
        self.outputs = set(outputs)
        self.nodes = {}
        self.gradients = {}
        self.kinds = {}

        self.sources = {}
        self.sinks = {}

        self.parameters = set([])
        self.constants = set([])

        self.parameter_shapes = {}
        self.loss_sources = []
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
            shape = kwargs.get('shape', np.array(value).shape)
            self.parameter_shapes[name] = shape

        elif kind == 'CONSTANT':
            self.constants.add(name)

    def connect_map(self, name, map, sources, sink):
        self.connect('MAP', name=name, value=map, sources=sources.copy(), sinks=[sink])

    def connect_constant(self, name, value, sinks=[]):
        self.connect('CONSTANT', name=name, value=value, sinks=sinks.copy())

    def connect_variable(self, name, source=None, sinks=[]):
        sources = [source] if source else []
        self.connect('VARIABLE', name=name, sources=sources, sinks=sinks.copy())

    def connect_parameter(self, name, value=None, sinks=[], shape=None):
        if value == None and shape == None:
            raise 'Parameters must be connected with a value or a shape'
        self.connect('PARAMETER', name=name, value=value, sources=[],
                                  sinks=sinks.copy(), shape=shape)

    def set_inputs(self, inputs):
        self.inputs = set(inputs)

    def set_outputs(self, outputs):
        self.outputs = set(outputs)

    def set_loss(self, sources, scalar_map, supervisors=0):
        self.loss_sources = sources
        self.loss_map = scalar_map
        self.supervisors = supervisors # number of outputs provided during training

    def set_learning_rate(rate):
        self.learning_rate = rate

    def initialize_parameters(self, bandwidth = 2):
        for p in self.parameters:
            self.nodes[p] = np.random.rand(*self.parameter_shapes[p])
            self.nodes[p] *= bandwidth
            self.nodes[p] -= bandwidth / 2

    def print_parameters(self):
        for p in self.parameters:
            print(p)
            print(self.nodes[p])

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
            sink_activation = self.nodes[active_map](source_activations)
            self.nodes[active_sink] = sink_activation
            traversed.add(active_map)
            traversed.add(active_sink)
            remaining_outputs.discard(active_sink)

        return {out: self.nodes[out] for out in outputs}

    def backpropagate(self, parameters=None):
        self.__reset_gradients()
        parameters = parameters or self.parameters
        remaining_parameters = set(parameters)
        self.gradients['LOSS'] = np.array(1.0)
        traversed = set(['LOSS'])

        while remaining_parameters:
            target = self.__choose(remaining_parameters)
            active_map = self.__last_node_after(target, traversed)
            active_sources = self.sources[active_map]
            active_sink = self.sinks[active_map][0]
            sink_gradient = self.gradients[active_sink]
            source_activations = [self.nodes[source] for source in active_sources]
            source_gradients = self.nodes[active_map].gradient(source_activations.copy())

            traversed.add(active_map)
            for i, source in enumerate(active_sources):
                self.gradients[source] += sum_product(source_gradients[i], sink_gradient)
                if self.__all_sinks_traversed(source, traversed):
                    traversed.add(source)
                    remaining_parameters.discard(source)

        return {param: self.gradients[param] for param in parameters}

    def train(self, inputs, outputs, learning_rate=.01):
        self.__attach_supervisors()
        self.__attach_loss()

        for i in range(len(inputs[self.__choose(self.inputs)])):
            training_values = self.__build_training_values(inputs, outputs, i)
            self.play(input_values=training_values)
            self.backpropagate()

            for parameter in self.parameters:
                self.nodes[parameter] -= self.gradients[parameter]*learning_rate

        self.__detach_supervisors()
        self.__detach_loss()

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
        self.gradients = {x: np.zeros_like(self.nodes[x]) for x in self.nodes}

    def __attach_supervisors(self):
        self.supervisor_names = []
        for i in range(self.supervisors):
            self.supervisor_names.append('SUPERVISOR#{}'.format(i+1))
            self.connect_variable(name=self.supervisor_names[i], sinks=['LOSS_MAP'])

    def __attach_loss(self):
        self.connect_variable(name='LOSS', source='LOSS_MAP')
        self.outputs.add('LOSS')
        self.connect_map(name='LOSS_MAP', map=self.loss_map,
                         sources=(self.loss_sources + self.supervisor_names),
                         sink='LOSS')
        for source in self.loss_sources:
            self.sinks[source].append('LOSS_MAP')


    def __detach_supervisors(self):
        for supervisor in self.supervisor_names:
            self.nodes.pop(supervisor)
            self.sinks.pop(supervisor)
            self.sources.pop(supervisor)

    def __detach_loss(self):
        for source in self.loss_sources:
            self.sinks[source].pop()

        self.nodes.pop('LOSS')
        self.sinks.pop('LOSS')
        self.sources.pop('LOSS')
        self.outputs.remove('LOSS')

        self.nodes.pop('LOSS_MAP')
        self.sinks.pop('LOSS_MAP')
        self.sources.pop('LOSS_MAP')

    def __build_training_values(self, inputs, outputs, i):
        if not isinstance(outputs[i], list):
            outputs[i] = [outputs[i]]
        supervisor_values = {'SUPERVISOR#{}'.format(k+1): out \
            for k, out in enumerate(outputs[i])}
        input_values = {name: inputs[name][i] for name in self.inputs}
        input_values.update(supervisor_values)
        return input_values

    @staticmethod
    def __choose(collection):
        target = collection.pop()
        collection.add(target)
        return target
