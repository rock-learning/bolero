# Author: Alexander Fabisch <afabisch@informatik.uni-bremen.de>

import os
import yaml
import warnings
import inspect
import StringIO


def optimizer_from_yaml(filename="learning_config.yml", conf_path=None):
    """Create optimizer object from YAML configuration file."""
    return from_yaml(filename, conf_path)["Optimizer"]


def behavior_from_yaml(filename="learning_config.yml", conf_path=None):
    """Create behavior object from YAML configuration file."""
    return from_yaml(filename, conf_path)["Behavior"]


def behavior_search_from_yaml(filename="learning_config.yml", conf_path=None):
    """Create behavior search object from YAML configuration file."""
    return from_yaml(filename, conf_path)["BehaviorSearch"]


def environment_from_yaml(filename="learning_config.yml", conf_path=None):
    """Create environment object from YAML configuration file."""
    return from_yaml(filename, conf_path)["Environment"]


def from_yaml(filename, conf_path=None):
    """Create objects from YAML configuration file.

    See also
    --------
    See :func:`from_dict`.

    Parameters
    ----------
    filename : string
        The name of the YAML file to load.

    conf_path : string, optional (default: $CONF_PATH)
        You can specify a path that is searched for the configuration file.
        Otherwise we try to read it from the environment variable
        'CONF_PATH'. If that environment variable is not present we search
        in the current path.

    Returns
    -------
    objects : dict
        Objects created from each entry of config with the same keys.
    """
    config = __load_config_from_file(filename, conf_path)
    return from_dict(config)


def __load_config_from_file(filename, conf_path=None):
    """Load configuration dictionary from YAML file.

    Parameters
    ----------
    filename : string
        The name of the YAML file to load.

    conf_path : string, optional (default: $CONF_PATH)
        You can specify a path that is searched for the configuration file.
        Otherwise we try to read it from the environment variable
        'CONF_PATH'. If that environment variable is not present we search
        in the current path.
    """
    if conf_path is None:
        conf_path = os.environ.get("CONF_PATH", None)

    if conf_path is None:
        conf_filename = filename
    else:
        conf_filename = os.path.join(conf_path, filename)

    if os.path.exists(conf_filename):
        config = yaml.load(open(conf_filename, "r"))
        return config
    else:
        raise ValueError("'%s' does not exist" % conf_filename)


def from_yaml_string(yaml_str):
    """Create objects from YAML string.

    Parameters
    ----------
    yaml_str : string
        YAML configuration string

    Returns
    -------
    objects : dict
        Objects created from each entry of config with the same keys.
    """
    return from_dict(yaml.load(yaml_str))


def from_dict(config, name=None):
    """Create an object of a class that is fully specified by a config dict.

    This will recursively go through all lists, tuples and dicts that are
    in the configuration. It will start to construct all objects starting
    from the leaves. For example

    .. code-block:: python

        config = {"type": "Class1", "arg1": {"type": "Class2"}}
        obj = from_dict(config)

    is equivalent to

    .. code-block:: python

        tmp = Class2()
        obj = Class1(arg1=tmp)

    Parameters
    ----------
    config : dict
        Configuration dictionary of the object. Contains constructor
        arguments.

    Returns
    -------
    object : as specified in the config
        The object created from the configuration dictionary or 'config'.
    """
    if isinstance(config, dict):
        it = config.items()
        result = {}
    elif isinstance(config, list) or isinstance(config, tuple):
        it = enumerate(config)
        result = [None for _ in range(len(config))]
    else:
        it = []
        result = config

    for k, v in it:
        result[k] = from_dict(v, name=k)

    if isinstance(config, dict) and "type" in config:
        return _from_dict(name, config)
    else:
        return result


def _from_dict(name, config):
    """Create an object of a class that is fully specified by a config dict.

    No recursion happens here. This function will directly create objects
    from the configuration.

    Parameters
    ----------
    name : string
        Key of the entry in the configuration dictionary

    config : dict
        Configuration dictionary of the object. Contains constructor
        arguments.

    Returns
    -------
    object : as specified in the config
        The object created from the configuration dictionary or 'config'.
    """
    c = config.copy()

    try:
        package_name = c.pop("package")
        has_explicit_package = True
    except KeyError:
        has_explicit_package = False

    type_name = c.pop("type")

    if not has_explicit_package:
        type_parts = type_name.split(".")
        package_name = ".".join(type_parts[:-1])
        type_name = type_parts[-1]

    if package_name == "":
        cpp_lib = _load_cpp_library(name, type_name)
        if cpp_lib is None:
            raise ValueError(
                "Empty module name. Either you tried to load a C++ library "
                "that cannot be found or you forgot to specify the Python "
                "package where the class '%s' is located." % type_name)
        config_string = StringIO.StringIO()
        yaml.dump(c, config_string)
        cpp_lib.initialize_yaml(config_string.getvalue())
        config_string.close()
        return cpp_lib

    package = __import__(package_name, {}, {}, fromlist=["dummy"], level=0)
    class_dict = dict(inspect.getmembers(package))

    if type_name in class_dict:
        clazz = class_dict[type_name]
    else:
        raise ValueError("Class name '%s' does not exist in module '%s'."
                         % (type_name, package_name))

    try:
        return clazz(**c)
    except TypeError as e:
        raise TypeError("Parameters for type '%s' do not match: %r. Reason: "
                        "'%s'" % (type_name, c, e.message))


PERMITTED_BASECLASSES = [
    "behavior", "behavior_search", "contextual_environment", "environment",
    "optimizer"]


def _load_cpp_library(name, type_name):
    from bolero import wrapper
    if not wrapper.__available__ or name is None:
        return None

    baseclass = name.lower()
    if baseclass not in PERMITTED_BASECLASSES:
        return None

    loader = wrapper.CppBLLoader()
    loader.load_library(type_name)
    factory = getattr(loader, "acquire_" + baseclass)
    return factory(type_name)
