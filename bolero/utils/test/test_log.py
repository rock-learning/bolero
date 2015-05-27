from subprocess import call
from bolero.utils.log import HideExtern
from bolero.utils.testing import assert_raise_message


def test_hide_extern():
    assert_raise_message(ValueError, "Stream 'std' not in", HideExtern, "std")
    with HideExtern("stdout"):
        call(["echo", "will never be seen"])
    with HideExtern("stderr"):
        call(["ls", "-e"])
