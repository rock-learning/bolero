from pywrap.cython import make_cython_wrapper, write_files, run_setup
from pywrap.defaultconfig import Config


def main():
    config = Config()

    results = make_cython_wrapper(
        "../src/Dmp.h", ["../src/Dmp.cpp"], "dmp", ".", config,
        ["../src"], verbose=0)
    del results["setup.py"]
    write_files(results, ".")
    run_setup("setup.py")


if __name__ == "__main__":
    main()
