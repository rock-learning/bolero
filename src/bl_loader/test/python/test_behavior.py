from bolero.representation import Behavior
import numpy as np
from numpy.testing import assert_array_equal


class TestBehavior(Behavior):
    """A simple test behavior.

    Multiplies every input by a value specified in the meta parameters.
    """
    def __init__(self, test_value_a, test_value_b):
        #test_value_a/b are hard coded in init.yaml
        self.can_step_counter = 5;
        assert test_value_a == 42.0; 
        assert test_value_b == 44.0;

    def init(self, n_inputs, n_outputs):
        assert n_inputs == n_outputs
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.inputs = np.empty(n_inputs)
        self.outputs = np.empty(n_outputs)

    def set_meta_parameters(self, keys, values):
        assert len(keys) == len(values)
        meta_parameters = dict(zip(keys, values))
        assert "expected_input" in meta_parameters
        self.expected_input = meta_parameters["expected_input"]
        assert "multiplier" in meta_parameters
        self.multiplier = meta_parameters["multiplier"]

    def set_inputs(self, inputs):
        assert len(self.expected_input) == self.n_inputs
        assert_array_equal(inputs, self.expected_input)
        self.inputs[:] = inputs

    def get_outputs(self, outputs):
        assert(len(outputs) == len(self.outputs))
        outputs[:] = self.outputs

    def step(self):
        self.outputs = self.inputs * self.multiplier
        self.can_step_counter = self.can_step_counter - 1
    
    def finish_step(self):
        pass

    def can_step(self):
        return self.can_step_counter > 0
