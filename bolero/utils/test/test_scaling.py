import numpy as np
from bolero.utils.scaling import Scaling
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_raises_regexp


params = np.linspace(0, 1, 10)


def test_no_scaling():
    s = Scaling(compute_inverse=True)
    scaled_params = s.scale(params)
    assert_array_equal(params, scaled_params)
    reconstructed_params = s.inv_scale(scaled_params)
    assert_array_equal(params, reconstructed_params)

    s = Scaling(compute_inverse=False)
    assert_raises_regexp(ValueError, "not computed", s.inv_scale,
                         s.scale(params))


def test_variance_scaling():
    s = Scaling(variance=2.35232, compute_inverse=True)
    reconstructed_params = s.inv_scale(s.scale(params))
    assert_array_almost_equal(params, reconstructed_params)

    s = Scaling(variance=2.35232, compute_inverse=False)
    assert_raises_regexp(ValueError, "not computed", s.inv_scale,
                         s.scale(params))


def test_diag_covariance_scaling():
    np.random.seed(0)
    c = np.random.rand(10)
    s = Scaling(covariance=c, compute_inverse=True)
    reconstructed_params = s.inv_scale(s.scale(params))
    assert_array_almost_equal(params, reconstructed_params)

    s = Scaling(covariance=c, compute_inverse=False)
    assert_raises_regexp(ValueError, "not computed", s.inv_scale,
                         s.scale(params))


def test_variance_diag_covariance_scaling():
    np.random.seed(0)
    c = np.random.rand(10)
    s = Scaling(variance=2.35232, covariance=c, compute_inverse=True)
    reconstructed_params = s.inv_scale(s.scale(params))
    assert_array_almost_equal(params, reconstructed_params)

    s = Scaling(variance=2.35232, covariance=c, compute_inverse=False)
    assert_raises_regexp(ValueError, "not computed", s.inv_scale,
                         s.scale(params))


def test_full_covariance_scaling():
    np.random.seed(0)
    r = np.random.rand(10, 10)
    c = r.dot(r.T)
    s = Scaling(covariance=c, compute_inverse=True)
    reconstructed_params = s.inv_scale(s.scale(params))
    assert_array_almost_equal(params, reconstructed_params)

    s = Scaling(covariance=c, compute_inverse=False)
    assert_raises_regexp(ValueError, "not computed", s.inv_scale,
                         s.scale(params))


def test_variance_full_covariance_scaling():
    np.random.seed(0)
    r = np.random.rand(10, 10)
    c = r.dot(r.T)
    s = Scaling(variance=2.35232, covariance=c, compute_inverse=True)
    reconstructed_params = s.inv_scale(s.scale(params))
    assert_array_almost_equal(params, reconstructed_params)

    s = Scaling(variance=2.35232, covariance=c, compute_inverse=False)
    assert_raises_regexp(ValueError, "not computed", s.inv_scale,
                         s.scale(params))


def test_invalid_covariance_shape():
    assert_raises_regexp(ValueError, "must have either 1 or 2 dimensions",
                         Scaling, covariance=np.array([[[]]]))
