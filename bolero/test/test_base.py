from bolero.base import Base
from nose.tools import assert_raises_regexp, assert_in, assert_equal


def test_varargs_raises_error():
    class Test(Base):
        def __init__(self, *args):
            pass

    assert_raises_regexp(RuntimeError, "no varargs", Test._get_param_names)


def test_get_params():
    class Test(Base):
        def __init__(self, p1, p2):
            self.p1 = p1
            self.p2 = p2

    obj = Test("1", 2)
    params = obj.get_params()
    assert_in("p1", params)
    assert_in("p2", params)
    assert_equal(params["p1"], "1")
    assert_equal(params["p2"], 2)
    assert_equal(repr(obj), "Test(p1='1', p2=2)")
