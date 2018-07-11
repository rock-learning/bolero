import numpy as np
import os
import scipy.io
import zipfile
import io
import urllib2

def load_lasa(shape_idx):
    """Load demonstrations from LASA dataset.

    The LASA dataset contains 2D handwriting motions recorded from a
    Tablet-PC. It can be found `here
    <https://bitbucket.org/khansari/lasahandwritingdataset>`_
    Take a look at the `detailed explanation
    <http://cs.stanford.edu/people/khansari/DSMotions#SEDS_Benchmark_Dataset>`_
    for more information.

    The following plot shows multiple demonstrations for the same shape.

    .. plot::

        import matplotlib.pyplot as plt
        from datasets import load_lasa
        X, Xd, Xdd, dt, shape_name = load_lasa(0)
        plt.figure()
        plt.title(shape_name)
        plt.plot(X[0], X[1])
        plt.show()

    Parameters
    ----------
    shape_idx : int
        Choose demonstrated shape, must be within range(30).

    Returns
    -------
    X : array-like, shape (n_task_dims, n_steps, n_demos)
        Positions

    Xd : array-like, shape (n_task_dims, n_steps, n_demos)
        Velocities

    Xdd : array-like, shape (n_task_dims, n_steps, n_demos)
        Accelerations

    dt : float
        Time between steps

    shape_name : string
        Name of the Matlab file from which we load the demonstrations
        (without suffix)
    """

    dataset_path = os.path.expanduser("~")
    dataset_path += os.sep + "bolero_data" + os.sep

    if not os.path.isdir(dataset_path):
        url = urllib2.urlopen("http://bitbucket.org/khansari/lasahandwritingdataset/get/38304f7c0ac4.zip")
        z = zipfile.ZipFile(io.BytesIO(url.read()))
        z.extractall(dataset_path)
        os.rename(dataset_path+z.namelist()[0], dataset_path+"lasa_data"+os.sep)

    dataset_path += "lasa_data" + os.sep + "DataSet" + os.sep
    demos, shape_name = _load_from_matlab_file(dataset_path, shape_idx)
    X, Xd, Xdd, dt = _convert_demonstrations(demos)
    return X, Xd, Xdd, dt, shape_name


def _load_from_matlab_file(dataset_path, shape_idx):
    """Load demonstrations from Matlab files."""
    file_name = sorted(os.listdir(dataset_path))[shape_idx]
    return (scipy.io.loadmat(dataset_path + file_name)["demos"][0],
            file_name[:-4])


def _convert_demonstrations(demos):
    """Convert Matlab struct to numpy arrays."""
    tmp = []
    for demo_idx in range(demos.shape[0]):
        # The Matlab format is strange...
        demo = demos[demo_idx][0, 0]
        # Positions, velocities and accelerations
        tmp.append((demo[0], demo[2], demo[3]))

    X = np.transpose([P for P, _, _ in tmp], [1, 2, 0])
    Xd = np.transpose([V for _, V, _ in tmp], [1, 2, 0])
    Xdd = np.transpose([A for _, _, A in tmp], [1, 2, 0])

    dt = float(demos[0][0, 0][4])

    return X, Xd, Xdd, dt
