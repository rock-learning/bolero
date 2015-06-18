from bolero.representation import Behavior
import numpy
from numpy.testing import assert_array_almost_equal


class TestBehavior(Behavior):
    """A simple test behavior.

    Multiplies every input by a value specified in the meta parameters.
    """
    def __init__(self, num_inputs, num_outputs, test_value_a, test_value_b):
        super(TestBehavior, self).__init__(num_inputs, num_outputs)
        #test_value_a/b are hard coded in init.yaml
        self.can_step_counter = 5;
        assert test_value_a == 42.0; 
        assert test_value_b == 44.0;
        assert num_inputs == num_outputs


    def set_meta_parameters(self, keys, values):
        assert len(keys) == len(values)
        meta_parameters = dict(zip(keys, values))
        assert "expected_input" in meta_parameters
        self.expected_input = numpy.copy(meta_parameters["expected_input"])
        assert "multiplier" in meta_parameters
        self.multiplier = numpy.copy(meta_parameters["multiplier"])
        assert len(self.expected_input) == self.num_inputs

    def set_inputs(self, inputs):
        assert_array_almost_equal(inputs, self.expected_input)
        self.inputs = inputs

    def get_outputs(self, outputs):
        assert(len(outputs) == len(self.outputs))
        numpy.copyto(outputs, self.outputs)

    def step(self):
        self.outputs = self.inputs * self.multiplier
        self.can_step_counter = self.can_step_counter - 1

    def can_step(self):
        return self.can_step_counter > 0


