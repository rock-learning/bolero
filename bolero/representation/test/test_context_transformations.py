import numpy as np
from bolero.representation.context_transformations import (
    constant, linear, affine, quadratic, cubic, polynomial)
from nose.tools import assert_equal, assert_in
from numpy.testing import assert_array_equal


def test_constant():
    context = np.array([1.0, 2.0])
    assert_equal(constant(context), 1.0)
    context = np.array([3.0, 2.0])
    assert_equal(constant(context), 1.0)


def test_linear():
    context = np.array([1.0, 2.0])
    assert_array_equal(linear(context), np.array([1.0, 2.0]))
    context = np.array([3.0, 2.0])
    assert_array_equal(linear(context), np.array([3.0, 2.0]))


def test_affine():
    context = np.array([1.0, 2.0])
    assert_array_equal(affine(context), np.array([2.0, 1.0, 1.0]))
    context = np.array([3.0, 2.0])
    assert_array_equal(affine(context), np.array([2.0, 3.0, 1.0]))


def test_quadratic():
    context = np.array([1.0, 2.0])
    transformed = quadratic(context)
    assert_in(1.0, transformed)
    assert_in(2.0, transformed)
    assert_in(4.0, transformed)
    assert_equal(len(transformed), 6)
    context = np.array([3.0, 2.0])
    transformed = quadratic(context)
    assert_in(1.0, transformed)
    assert_in(2.0, transformed)
    assert_in(3.0, transformed)
    assert_in(6.0, transformed)
    assert_equal(len(transformed), 6)


def test_cubic():
    context = np.array([1.0, 2.0])
    transformed = cubic(context)
    assert_in(1.0, transformed)
    assert_in(2.0, transformed)
    assert_in(4.0, transformed)
    assert_in(8.0, transformed)
    assert_equal(len(transformed), 10)
    context = np.array([3.0, 2.0])
    transformed = cubic(context)
    assert_in(1.0, transformed)
    assert_in(2.0, transformed)
    assert_in(4.0, transformed)
    assert_in(8.0, transformed)
    assert_in(6.0, transformed)
    assert_in(9.0, transformed)
    assert_in(27.0, transformed)
    assert_equal(len(transformed), 10)


def test_polynomial():
    context = np.array([1.0, 2.0])
    assert_array_equal(quadratic(context), polynomial(context, n_degrees=2))
    assert_array_equal(cubic(context), polynomial(context, n_degrees=3))
    context = np.array([3.0, 2.0])
    assert_array_equal(quadratic(context), polynomial(context, n_degrees=2))
    assert_array_equal(cubic(context), polynomial(context, n_degrees=3))
