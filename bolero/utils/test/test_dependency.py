from bolero.utils.dependency import compatible_version
from nose.tools import assert_true, assert_raises_regexp


def test_compatible_version():
    assert_true(compatible_version("1", ">= 0.1.2"))
    assert_true(compatible_version("1.2", "< 2.1.3"))
    assert_true(compatible_version("sklearn", "> 0.0.0"))
    assert_true(compatible_version("0.15-git", "> 0.0.0"))
    assert_true(compatible_version("0.16.dev", "> 0.0.0"))
    assert_raises_regexp(ValueError, "Unknown comparison operator",
                         compatible_version, "0.0", "+ 1.0")
    assert_raises_regexp(ValueError, "Wrong operation syntax",
                         compatible_version, "0.16.dev", ">0.0.0")
