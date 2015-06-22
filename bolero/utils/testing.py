from nose.tools import assert_in, assert_true
import bolero
import pkgutil
import inspect
import os
import pickle


# Taken from scikit-learn
def assert_raise_message(exception, message, function, *args, **kwargs):
    """Helper function to test error messages in exceptions"""
    try:
        function(*args, **kwargs)
        raise AssertionError("Should have raised %r" % exception(message))
    except exception as e:
        error_message = str(e)
        assert_in(message, error_message)


def assert_pickle(name, obj):
    filename = name + ".pickle"
    try:
        pickle.dump(obj, open(filename, "w"))
        assert_true(os.path.exists(filename))
        pickle.load(open(filename, "r"))
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def all_subclasses(base_class, exclude_classes=[]):
    """Get a list of subclasses of the base class.

    Parameters
    ----------
    base_class : class
        Base class

    exclude_classes : list of strings
        List of classes that will be excluded

    Returns
    -------
    subclasses : list of tuples
        List of (name, class), where ``name`` is the class name as string
        and ``class`` is the actuall type of the class.
    """
    def is_abstract(c):
        if not(hasattr(c, '__abstractmethods__')):
            return False
        if not len(c.__abstractmethods__):
            return False
        return True

    all_classes = []
    path = bolero.__path__
    for importer, modname, ispkg in pkgutil.walk_packages(
            path=path, prefix='bolero.', onerror=lambda x: None):
        if ".test." in modname:
            continue
        module = __import__(modname, fromlist="dummy")
        classes = inspect.getmembers(module, inspect.isclass)
        all_classes.extend(classes)

    all_classes = set(all_classes)

    subclasses = [c for c in all_classes
                  if (issubclass(c[1], base_class)
                      and c[0] != base_class.__name__)]
    # get rid of abstract base classes
    subclasses = [c for c in subclasses if not is_abstract(c[1])]

    if exclude_classes:
        subclasses = [c for c in subclasses if not c[0] in exclude_classes]

    # drop duplicates, sort for reproducibility
    return sorted(set(subclasses))

