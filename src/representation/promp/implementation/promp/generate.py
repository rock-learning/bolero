from pywrap.cython import make_cython_wrapper, write_files, run_setup
from pywrap.defaultconfig import Config
from pywrap.type_conversion import AbstractTypeConverter


class DoubleArray2dTypeConverter(AbstractTypeConverter):
    def matches(self):
        if self.context is None:
            return False
        args, index = self.context
        next_arg_is_int = (len(args) >= index + 2
                           and (args[index + 1].tipe == "int"
                                or args[index + 1].tipe == "unsigned int"))
        next_next_arg_is_int = (len(args) >= index + 3
                                and (args[index + 2].tipe == "int"
                                     or args[index + 2].tipe == "unsigned int"))
        return (self.tname == "double *" and next_arg_is_int
                and next_next_arg_is_int)

    def n_cpp_args(self):
        return 3

    def add_includes(self, includes):
        includes.add_include_for_numpy()

    def python_to_cpp(self):
        return ""

    def cpp_call_args(self):
        return ["&%s[0, 0]" % self.python_argname,
                self.python_argname + ".shape[0]",
                self.python_argname + ".shape[1]"]

    def return_output(self, copy=True):
        raise NotImplementedError("Cannot return double array")

    def python_type_decl(self):
        return "np.ndarray[double, ndim=2] %s" % self.python_argname

    def cpp_type_decl(self):
        raise NotImplementedError("Double array must provide additional size")


def main():
    config = Config()
    config.registered_converters.append(DoubleArray2dTypeConverter)

    results = make_cython_wrapper(
        "../src/promp.h", ["../src/promp.cpp"], "promp", ".", config,
        ["../src"], verbose=0)
    del results["setup.py"]
    write_files(results, ".")
    run_setup("setup.py")


if __name__ == "__main__":
    main()
