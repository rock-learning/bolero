.. _contributing:

============
Contributing
============

This project is mainly developed at the robotics research group at the
University of Bremen and the German Research Center for Artificial Intelligence
Robotics Innovation Center (DFKI RIC. However, everyone is welcome to
contribute.

There are several ways to contribute to BOLeRo, you could e.g.

* send a bug report (`bug tracker
  <http://github.com/rock-learning/bolero/issues>`_)
* write documentation
* add a new feature
* add tests
* add an example
* add a result from a paper as a benchmark

How to contribute code
----------------------

The preferred way to contribute to BOLeRo is to fork the `main
repository <http://github.com/rock-learning/bolero/>`__ on GitHub,
then submit a "pull request" (PR):

1. `Create an account <https://github.com/signup/free>`_ on
   GitHub if you do not already have one.

2. Fork the `project repository <http://github.com/rock-learning/bolero>`__:
   click on the 'Fork' button near the top of the page. This creates a copy of
   the code under your account on the GitHub server.

3. Clone this copy to your local disk::

       $ git clone git@github.com:YourLogin/bolero.git

4. Create a branch to hold your changes::

       $ git checkout -b my-feature

   and start making changes. Never work in the ``master`` branch!

5. Work on this copy, on your computer, using Git to do the version
   control. When you're done editing, do::

       $ git add modified_files
       $ git commit

   to record your changes in Git, then push them to GitHub with::

       $ git push -u origin my-feature

Finally, go to the web page of the your fork of the bolero repo,
and click 'Pull request' to send your changes to the maintainers for review.
request.

.. note::

  In the above setup, your ``origin`` remote repository points to
  YourLogin/bolero.git. If you wish to fetch/merge from the main
  repository instead of your forked one, you will need to add another remote
  to use instead of ``origin``. If we choose the name ``upstream`` for it, the
  command will be::

        $ git remote add upstream https://github.com/rock-learning/bolero.git

(If any of the above seems like magic to you, then look up the
`Git documentation <http://git-scm.com/documentation>`_ on the web.)

Requirements for New Features
-----------------------------

Adding a new feature to BOLeRo requires a few other changes:

* New classes or functions that are part of the public interface must be
  documented. We use `NumPy's conventions for docstrings
  <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_
  for Python and doxygen for C++.
* An entry to the API documentation must be added `here for Python
  <https://rock-learning.github.io/bolero/api_py.html>`_ or `here for C++
  <https://rock-learning.github.io/bolero/api_cpp.html>`_.
* A new section should be added to the narrative documentation. For example,
  you can add a section for a new environment `here
  <https://rock-learning.github.io/bolero/modules/environment.html>`_.
* Update NEWS.md.
* Consider writing a simple example that will be added to the example
  gallery.
* If you reimplemented an algorithm from a paper, consider adding a benchmark
  from the paper to the benchmark directory of BOLeRo.
* Tests: Unit tests for new features are mandatory. They should cover most of
  the branches. Exceptions are plotting functions, debug outputs, etc. These
  are usually hard to test and are not a fundamental part of the library. We
  use nosetests in Python and Catch in C++.

Merge Policy
------------

Usually it is not possible to push directly to the master branch of BOLeRo
for anyone. Only tiny changes, urgent bugfixes, and maintenance commits can
be pushed directly to the master branch by the maintainer without a review.
"Tiny" means backwards compatibility is mandatory and all tests must succeed.
No new feature must be added.

Developers have to submit pull requests. Those will be reviewed by at least
one other developer and merged by a maintainer. New features must be
documented and tested. Breaking changes must be discussed and announced
in advance with deprecation warnings.

Versioning
----------

Semantic versioning must be used, that is, the major version number will be
incremented when the API changes in a backwards incompatible way, the minor
version will be incremented when new functionality is added in a backwards
compatible manner, and the patch version is incremented for bugfixes,
documentation, etc.
