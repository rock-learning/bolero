import numpy as np
from bolero.representation import (Behavior, BlackBoxBehavior,
                                   HierarchicalBehaviorTemplate)
from nose.tools import assert_raises_regexp, assert_true


class MockupBehavior(Behavior):
    def init(self): pass
    def set_meta_parameters(self): pass
    def set_inputs(self): pass
    def step(self): pass
    def get_outputs(self): pass


def test_non_black_box_behavior():
    assert_raises_regexp(
        ValueError, "must be of type 'BlackBoxBehavior'",
        HierarchicalBehaviorTemplate, lambda s, *args, **kwargs: np.array([]),
        MockupBehavior())


def test_generate_behavior():
    class MockupBlackBoxBehavior(BlackBoxBehavior, MockupBehavior):
        def __init__(self):
            self.got_params = False
            self.resetted = False
        def get_n_params(self): pass
        def get_params(self): pass
        def set_params(self, p):
            if np.all(p == np.array([])):
                self.got_params = True
        def reset(self):
            self.resetted = True
        def check(self):
            return self.got_params and self.resetted

    bt = HierarchicalBehaviorTemplate(
        lambda s, *args, **kwargs: np.array([]), MockupBlackBoxBehavior())
    beh = bt.get_behavior(np.array([]))
    assert_true(beh.check())
