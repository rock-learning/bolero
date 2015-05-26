from module_loader import from_yaml, from_dict


__all__ = ["from_yaml", "from_dict"]


class NonContextualException(Exception):
    """ Exception thrown in methods not supported in contextual scenarios."""
    pass


class ContextualException(Exception):
    """ Exception thrown in methods not supported in non-contextual scenarios."""
    pass
