from .behavior import Behavior, BehaviorTemplate, BlackBoxBehavior
from .baseline import ConstantBehavior, DummyBehavior, RandomBehavior
from .hierarchical_behavior_template import HierarchicalBehaviorTemplate


__all__ = [
    "Behavior", "BehaviorTemplate", "BlackBoxBehavior", "ConstantBehavior",
    "DummyBehavior", "RandomBehavior", "HierarchicalBehaviorTemplate"]

try:
    from .dmp_behavior import DMPBehavior
    __all__.append("DMPBehavior")
except ImportError:
    pass
