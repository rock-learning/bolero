# Author: Alexander Fabisch <afabisch@informatik.uni-bremen.de>

import os
import yaml
import warnings
import inspect


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
        conf_path = os.environ.get("CONF_PATH", ".")
    conf_filename = conf_path + os.sep + filename
    if os.path.exists(conf_filename):
        config = yaml.load(open(conf_filename, "r"))
        return config
    else:
        warnings.warn("Could not find any configuration file. CONF_PATH=%s, "
                      "conf_filename=%s" % (conf_path, conf_filename))
        return {}


def from_dict(config):
    """Create an object of a class that is fully specified by a config dict.

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
    elif isinstance(config, list):
        it = enumerate(config)
    elif isinstance(config, tuple):
        it = enumerate(config)
    else:
        it = []

    for k, v in it:
        config[k] = from_dict(v)

    if isinstance(config, dict) and "type" in config:
        return recursive_from_dict(config)
    else:
        return config


def recursive_from_dict(config):
    """Create an object of a class that is fully specified by a config dict.

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
    if not isinstance(config, dict):
        raise TypeError("Config must be of type 'dict' but is %s (%r)"
                        % (type(config), config))

    c = config.copy()

    try:
        package_name = c.pop("package")
        has_explicit_package = True
    except KeyError:
        has_explicit_package = False

    try:
        type_name = c.pop("type")
    except KeyError:
        raise ValueError("No type given")

    if not has_explicit_package:
        type_parts = type_name.split(".")
        package_name = ".".join(type_parts[:-1])
        type_name = type_parts[-1]

    package = __import__(package_name, {}, {}, fromlist=["dummy"], level=0)
    class_dict = dict(inspect.getmembers(package))

    if type_name in class_dict:
        clazz = class_dict[type_name]
    else:
        raise KeyError("Class name '%s' is unknown." % type_name)

    try:
        return clazz(**c)
    except TypeError as e:
        raise TypeError("Parameters for type '%s' do not match: %r. Reason: "
                        "'%s'" % (type_name, c, e.message))
