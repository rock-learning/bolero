from nose.tools import assert_in


def assert_raise_message(exception, message, function, *args, **kwargs):
    """Helper function to test error messages in exceptions"""

    try:
        function(*args, **kwargs)
        raise AssertionError("Should have raised %r" % exception(message))
    except exception as e:
        error_message = str(e)
        assert_in(message, error_message)
