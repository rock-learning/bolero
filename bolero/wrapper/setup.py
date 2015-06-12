def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration("wrapper", parent_package, top_path)
    try:
        import build_info
    except ImportError:
        build_info = None
    if build_info is None:
        config.set_options(ignore_setup_xxx_py=True,
                           assume_default_configuration=True,
                           delegate_options_to_subpackages=True,
                           quiet=True)
        return config

    import numpy

    # CMake outputs multiple include dirs separated by ";"
    # but the setup scripts needs it as list => split it
    config.add_extension(
        '_wrapper',
        sources=["_wrapper.cpp"], #created by cython
        include_dirs=[".",
                      numpy.get_include(),
                      build_info.BOLERO_INCLUDE_DIRS.split(";"),
                      build_info.BL_LOADER_INCLUDE_DIRS.split(";"),
                      build_info.LIB_MANAGER_INCLUDE_DIRS.split(";")],
        libraries=["bl_loader", "lib_manager"],
        library_dirs=[build_info.BL_LOADER_LIBRARY_DIRS.split(";"),
                      build_info.LIB_MANAGER_LIBRARY_DIRS.split(";")],
        define_macros=[("NDEBUG",)],
        extra_compile_args=["-O3", "-Wno-unused-function",
                            "-Wno-unused-but-set-variable",])
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())


