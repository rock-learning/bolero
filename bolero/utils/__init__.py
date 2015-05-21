class NonContextualException(Exception):
    """ Exception thrown in methods not supported in contextual scenarios."""
    pass


class ContextualException(Exception):
    """ Exception thrown in methods not supported in non-contextual scenarios."""
    pass
