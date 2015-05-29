import os
from bolero.utils import from_dict, from_yaml
from bolero.optimizer import CMAESOptimizer
from nose.tools import assert_true, assert_equal
from bolero.utils.testing import assert_raise_message


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
    assert_raise_message(TypeError, "got an unexpected keyword argument", from_dict, config)


def test_missing_package():
    config = {"type": "CMAESOptimizer"}
    assert_raise_message(ValueError, "Empty module name", from_dict, config)


def test_load_from_yaml():
    optimizer = from_yaml(CONFIG_FILE)
    assert_true(isinstance(optimizer, CMAESOptimizer))


def test_load_from_yaml_with_conf_path():
    optimizer = from_yaml("test_config.yaml", CURRENT_PATH)
    assert_true(isinstance(optimizer, CMAESOptimizer))
