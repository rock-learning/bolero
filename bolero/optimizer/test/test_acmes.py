from bolero.utils.validation import assert_warns
from bolero.optimizer import ACMESOptimizer
from nose.tools import assert_raises_regexp, assert_equal


def test_acmes_clip_samples():
    opt = ACMESOptimizer(n_train_max=20001)
    assert_warns(UserWarning, opt.init, 1)
    assert_equal(opt.n_train_max, 20000)


def test_acmes_no_presamples():
    opt = ACMESOptimizer(n_pre_samples_per_update=0)
    assert_raises_regexp(ValueError, "At least one sample", opt.init, 5)
