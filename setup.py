#! /usr/bin/env python
# adapted from sklearn

import sys
import os
import shutil
import glob
from distutils.command.clean import clean as Clean
import bolero


if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins

class CleanCommand(Clean):
    description = "Remove build directories"#, and compiled file in the source tree"
 
    def run(self):
        Clean.run(self)
        if os.path.exists('build'):
            # setup.py creates build/lib.* and build/tmp.*
            # everything else in 'build' is created by cmake and should be
            # ignored
            for path in glob.glob("build/lib.*"):
                print("Removing '" + path +"'")
                shutil.rmtree(path)
            for path in glob.glob("build/temp.*"):
                print("Removing '" + path + "'")
                shutil.rmtree(path)


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('bolero')
    return config


def setup_package():
    metadata = dict(
        name="bolero",
        maintainer="DFKI-RIC",
        maintainer_email="behavior-learning@dfki.de",
        description="Behavior Optimization and Learning for Robots",
        license="BSD 3-clause",
        version=bolero.__version__,
        url="http://robotik.dfki-bremen.de/en/research/softwaretools/bolero.html",
        #download_url="TODO",
        cmdclass={'clean': CleanCommand},
        )

    if (len(sys.argv) >= 2
            and ('--help' in sys.argv[1:] or sys.argv[1]
                 in ('--help-commands', 'egg_info', '--version', 'clean'))):
        try:
            from setuptools import setup
            #install_requires is only available if setuptools is used
            metadata["install_requires"] = ["PyYAML", "scipy", "numpy"]
        except ImportError:
            from distutils.core import setup

    else:
        from numpy.distutils.core import setup

        metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
