[![Travis Status](https://travis-ci.org/rock-learning/bolero.svg?branch=master)](https://travis-ci.org/rock-learning/bolero)
[![CircleCI Status](https://circleci.com/gh/rock-learning/bolero/tree/master.svg?style=shield&circle-token=:circle-token)](https://circleci.com/gh/rock-learning/bolero)

<img width="60%" src="doc/source/_static/logo.png" />

# BOLeRo

BOLeRo (Behavior Learning and Optimization for Robots) provides tools to learn
behaviors for robots. It includes behavior representations as well as
reinforcement learning, black-box optimization, evolutionary algorithms, and
imitation learning. It provides a C++ and a Python interface to be efficient
where this is required and to be flexible and convenient where performance is
not an issue. Because the library provides a C++ interface, it is easy to
integrate in most robotic frameworks, e.g. the robot operating system (ROS) or
the robot construction kit (Rock).

## Installation

On Ubuntu 16.04 and similar operating systems, you can install BOLeRo with

    wget https://raw.githubusercontent.com/rock-learning/bolero/master/bootstrap_bolero.sh
    chmod +x bootstrap_bolero.sh
    ./bootstrap_bolero.sh

The installation script will create a new folder `bolero-dev` that contains
all sources and built binaries. If you want to use BOLeRo, you have to source
the file env.sh:

    source bolero-dev/env.sh

Detailed installation instructions (for other operating systems, with docker,
etc.) can be found at
[the documentation](https://rock-learning.github.io/bolero/installation.html).

## Documentation

The documentation is available [here](https://rock-learning.github.io/bolero).
It can be built in the directory `doc/` with `make`. It will be located
in `doc/build/html/index.html`. Building the documentation requires
[doxygen](http://www.stack.nl/~dimitri/doxygen/) and
[sphinx](http://sphinx-doc.org/).

## Directories

BOLeRo contains the following directories:

* benchmarks - contains benchmark scripts or scripts that reproduce results
  from scientific papers
* bolero - contains the Python library
* doc - contains the documentation
* examples - contains examples that demonstrate how to use bolero
* include - contains the header files that define the C++ interfaces
* src - contains several C++ packages

## License

BOLeRo is distributed under the
[3-clause BSD license](https://opensource.org/licenses/BSD-3-Clause).
