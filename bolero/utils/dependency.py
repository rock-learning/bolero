def compatible_version(actual_version_info, operation):
    """Compares version strings.

    Examples
    --------
    >>> from bolero.utils.dependency import compatible_version
    >>> compatible_version("1", ">= 0.1.2")
    True
    >>> compatible_version("sklearn", "> 0.0.0")
    True

    Parameters
    ----------
    actual_version_info : string
        Either a package name of a package that provides the version string
        as variable '__version__' or a version string.

    operation : string
        A comparison operation. Operator and version string must be seperated
        by whitespace, e.g. '>= 0.12.0'.

    Returns
    -------
    comatible : bool
        Is the version compatible?
    """
    try:
        package = __import__(actual_version_info)
        actual_version = package.__version__
    except ImportError:
        # 'actual_version_info' must be a version string
        actual_version = actual_version_info

    if "git" in actual_version:
        # Handle special case '0.15-git'
        actual_version = actual_version.split("-")[0].split(".")
    elif "dev" in actual_version:
        # Handle special case '0.16.dev'
        actual_version = actual_version.split(".")[:2]
    else:
        actual_version = actual_version.split(".")

    actual_version = map(int, actual_version)

    if not " " in operation:
        raise ValueError("Wrong operation syntax, must be: OPERATOR VERSION")

    op, version = operation.split(" ")
    if not op in ["<", ">", "==", ">=", "<="]:
        raise ValueError("Unknown comparison operator '%s'" % op)
    version = map(int, version.split("."))

    return eval("%s %s %s" % (tuple(actual_version), op, tuple(version)))
