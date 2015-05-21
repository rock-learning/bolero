import os
import shutil


INDEXHEADER = """
=========
Notebooks
=========

.. toctree::
   :maxdepth: 1

"""


class bcolors:  # From http://stackoverflow.com/a/287944/915743
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


if __name__ == "__main__":
    scriptdir = os.sep.join(["."] + __file__.split(os.sep)[:-1])
    root = os.sep.join(scriptdir.split(os.sep) + [".."]) + os.sep + "examples"
    outdir = scriptdir + os.sep + "source" + os.sep + "notebooks"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print(bcolors.OKBLUE + ("Notebook directory: %s" % outdir) + bcolors.ENDC)

    notebooks = []
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".ipynb"):
                notebooks.append(dirpath + os.sep + filename)

    names = []
    for notebook in notebooks:
        name = notebook.split(os.sep)[-1].replace(".ipynb", "")
        output = outdir + os.sep + name
        code = os.system("ipython nbconvert %s --to=rst" % notebook)
        for filename in [name + ".rst", name + "_files"]:
            target_file = outdir + os.sep + filename
            if os.path.exists(target_file):
                try:
                    shutil.rmtree(target_file)
                except:
                    os.remove(target_file)
            shutil.move(filename, outdir)
        names.append(name)

    with open(outdir + os.sep + "index.rst", "w") as index:
        index.write(INDEXHEADER)
        for name in names:
            index.write("   " + name + os.linesep)
