import os
from bolero.utils import from_dict, from_yaml
from bolero.optimizer import CMAESOptimizer
from nose.tools import assert_true, assert_equal, assert_raises_regexp


CURRENT_PATH = os.sep.join(__file__.split(os.sep)[:-1])
CONFIG_FILE = CURRENT_PATH + os.sep + "test_config.yaml"
if not CURRENT_PATH:
    CONFIG_FILE = "test_config.yaml"


def test_load_implicit_package():
    config = {"type": "bolero.optimizer.CMAESOptimizer"}
    optimizer = from_dict(config)
    assert_true(isinstance(optimizer, CMAESOptimizer))


def test_load_explicit_package():
    config = {
        "package": "bolero.optimizer",
        "type": "CMAESOptimizer"
    }
    optimizer = from_dict(config)
    assert_true(isinstance(optimizer, CMAESOptimizer))


def test_load_with_params():
    config = {
        "type": "bolero.optimizer.CMAESOptimizer",
        "variance": 10.0,
        "n_samples_per_update": 10
    }
    optimizer = from_dict(config)
    assert_true(isinstance(optimizer, CMAESOptimizer))
    assert_equal(optimizer.variance, 10.0)
    assert_equal(optimizer.n_samples_per_update, 10)


def test_load_missing_type():
    config = {"key": "value"}
    config2 = from_dict(config)
    assert_equal(config, config2)


def test_load_with_wrong_params():
    config = {
        "type": "bolero.optimizer.CMAESOptimizer",
        "varince": 10.0
    }
    assert_raises_regexp(TypeError, "unexpected keyword", from_dict, config)


def test_load_dict():
    opt_config = {"type": "bolero.optimizer.CMAESOptimizer"}
    config = {"optimizers": {0: opt_config, 1: opt_config}}
    result = from_dict(config)
    assert_true(isinstance(result["optimizers"][0], CMAESOptimizer))
    assert_true(isinstance(result["optimizers"][1], CMAESOptimizer))


def test_load_list():
    opt_config = {"type": "bolero.optimizer.CMAESOptimizer"}
    config = {"optimizers": [opt_config, opt_config]}
    result = from_dict(config)
    assert_true(isinstance(result["optimizers"][0], CMAESOptimizer))
    assert_true(isinstance(result["optimizers"][1], CMAESOptimizer))


def test_load_tuple():
    opt_config = {"type": "bolero.optimizer.CMAESOptimizer"}
    config = {"optimizers": (opt_config, opt_config)}
    result = from_dict(config)
    assert_true(isinstance(result["optimizers"][0], CMAESOptimizer))
    assert_true(isinstance(result["optimizers"][1], CMAESOptimizer))


def test_missing_package():
    config = {"type": "CMAESOptimizer"}
    assert_raises_regexp(ValueError, "Empty module name", from_dict, config)


def test_load_class_does_not_exist():
    config = {"type": "bolero.optimizer.DoesNotExist"}
    assert_raises_regexp(ValueError, "does not exist in", from_dict, config)


def test_load_from_yaml():
    optimizer = from_yaml(CONFIG_FILE)
    assert_true(isinstance(optimizer, CMAESOptimizer))


def test_load_from_yaml_with_conf_path():
    optimizer = from_yaml("test_config.yaml", CURRENT_PATH)
    assert_true(isinstance(optimizer, CMAESOptimizer))


def test_load_from_missing_yaml():
    assert_raises_regexp(ValueError, "does not exist", from_yaml, "dummy.yaml")


def test_load_cpp_lib():
    opt = from_dict({"Optimizer": {"type": "pso_optimizer"}})["Optimizer"]
    assert_true(hasattr(opt, "get_next_parameters"))
