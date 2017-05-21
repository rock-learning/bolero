"""
=======================
Load Module from Config
=======================

In bolero, we can load modules (optimizers, behavior search methods, behaviors,
environments) either from configuration dictionaries or from YAML files. This
example shows how we can load an optimizer from a configuration dictionary.
"""
print(__doc__)

from bolero.utils import from_dict


config = {
    "type": "bolero.optimizer.CMAESOptimizer",
    "variance": 10.0,
}

optimizer = from_dict(config)
optimizer.init(2)
params = [0, 0]
optimizer.get_next_parameters(params)
print(params)
