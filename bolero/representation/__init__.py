from .behavior import Behavior, BehaviorTemplate, BlackBoxBehavior
from .baseline import ConstantBehavior, DummyBehavior, RandomBehavior
from .hierarchical_behavior_template import HierarchicalBehaviorTemplate
from .linear import LinearBehavior


__all__ = [
    "Behavior", "BehaviorTemplate", "BlackBoxBehavior", "ConstantBehavior",
    "DummyBehavior", "RandomBehavior", "HierarchicalBehaviorTemplate",
    "LinearBehavior"]

try:
    from .dmp_behavior import DMPBehavior
    __all__.append("DMPBehavior")
    from .csdmp_behavior import CartesianDMPBehavior
    __all__.append("CartesianDMPBehavior")
    from .dmp_sequence import DMPSequence
    __all__.append("DMPSequence")
except ImportError:
    pass
