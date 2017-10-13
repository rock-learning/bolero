from sklearn.utils.testing import assert_warns, assert_equal
from bolero.optimizer import ACMESOptimizer


def test_acmes_clip_samples():
    opt = ACMESOptimizer(n_train_max=20001)
    assert_warns(UserWarning, opt.init, 1)
    assert_equal(opt.n_train_max, 20000)
