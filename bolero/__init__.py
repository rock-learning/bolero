__version__ = "1.1.0-dev"


__all__ = ["environment", "optimizer", "representation", "utils", "wrapper"]


def setup_module(module):
    """Monkey patch for nosetests.

    nosetests tries to import all modules and looks for the attribute 'setup'.
    In our case it finds setup.py and tries to handle it as a function which
    does not work. So we have to tell nosetests that 'setup' is not a possible
    name for a setup function.
    """
    from nose.suite import ContextSuite
    moduleSetup = list(ContextSuite.moduleSetup)
    moduleSetup.remove("setup")
    ContextSuite.moduleSetup = tuple(moduleSetup)
