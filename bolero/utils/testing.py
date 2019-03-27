from nose.tools import assert_true
import bolero
import pkgutil
import inspect
import os
import pickle


def assert_pickle(name, obj):
    filename = name + ".pickle"
    try:
        with open(filename, "wb") as outf:
            pickle.dump(obj, outf)
        assert_true(os.path.exists(filename))
        with open(filename, "rb") as inf:
            pickle.load(inf)
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def all_subclasses(base_class, exclude_classes=[], root=bolero, verbose=0):
    """Get a list of subclasses of the base class.

    Parameters
    ----------
    base_class : class
        Base class

    exclude_classes : list of strings
        List of classes that will be excluded

    root : package, optional (default: bolero)
        Root package to search for subclasses

    verbose : int, optional (default: 0)
        Inform about modules that are skipped during the search

    Returns
    -------
    subclasses : list of tuples
        List of (name, class), where ``name`` is the class name as string
        and ``class`` is the actuall type of the class.
    """
    all_classes = []
    path = root.__path__
    if verbose >= 2:
        print("Root path: %s" % path)
    for importer, modname, ispkg in pkgutil.walk_packages(
            path=path, prefix=root.__name__+'.', onerror=lambda x: None):
        if verbose >= 2:
            print("Inspecting packages '%s'" % modname)
        if ".test." in modname:
            continue
        try:
            module = __import__(modname, fromlist="dummy")
            classes = inspect.getmembers(module, inspect.isclass)
            all_classes.extend(classes)
            if verbose >= 2:
                print("Found classes: %s" % classes)
        except ImportError:
            if verbose:
                print("Module %s is skipped due to an import error" % modname)

    all_classes = set(all_classes)

    subclasses = [c for c in all_classes
                  if (issubclass(c[1], base_class)
                      and c[0] != base_class.__name__)]
    # get rid of abstract base classes
    subclasses = [c for c in subclasses if not inspect.isabstract(c[1])]

    if exclude_classes:
        subclasses = [c for c in subclasses if not c[0] in exclude_classes]

    # drop duplicates, sort for reproducibility
    return sorted(set(subclasses))

