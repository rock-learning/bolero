from subprocess import call
from bolero.utils.log import HideExtern
from nose.tools import assert_raises_regexp


def test_hide_extern():
    assert_raises_regexp(ValueError, "Stream 'std' not in", HideExtern, "std")
    with HideExtern("stdout"):
        call(["echo", "will never be seen"])
    with HideExtern("stderr"):
        call(["ls", "-e"])
