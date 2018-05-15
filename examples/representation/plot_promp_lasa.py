"""
================================
LASA Handwriting with ProMPs
================================

The LASA Handwriting dataset learned with ProMPs. 
"""
print(__doc__)

import numpy as np
import os
import scipy.io
from git import Repo
import numpy as np
from bolero.representation import ProMPBehavior
import matplotlib.pyplot as plt

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

    if not os.path.isdir("lasa_data"):
        Repo.clone_from("https://bitbucket.org/khansari/lasahandwritingdataset.git", "lasa_data")

    
    dataset_path = os.sep
        
    dataset_path = "lasa_data" + os.sep + "DataSet" + os.sep
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

def load(idx):    
    X, Xd, Xdd, dt, shape_name = load_lasa(idx)
    y = X.transpose(2,1,0)	    
    x = np.linspace(0,1,1000)
    return (x,y)

def learn(x,y):
    traj = ProMPBehavior(1.0, 1.0/999.0, numWeights,learnCovariance=True,useCovar=True)
    traj.init(4,4)
    traj.imitate(y.transpose(2,1,0))  
    return traj
	
def draw(x,y,traj,idx,axs):
    h = int(idx/width)
    w = int(idx%width)*2
    axs[h,w].plot(y.transpose(2,1,0)[0], y.transpose(2,1,0)[1])
    
    mean, _ , covar = traj.trajectory()
    axs[h,w+1].plot(mean[:,0],mean[:,1])
    traj.plotCovariance(axs[h,w+1],mean,covar.reshape(-1,4,4))
        
    axs[h,w+1].set_xlim(axs[h,w].get_xlim())
    axs[h,w+1].set_ylim(axs[h,w].get_ylim())
    axs[h,w].get_yaxis().set_visible(False)
    axs[h,w].get_xaxis().set_visible(False)
    axs[h,w+1].get_yaxis().set_visible(False)
    axs[h,w+1].get_xaxis().set_visible(False)

numWeights = 30 #how many weights shall be used 
numShapes = 10
width = 2
height = 5

fig, axs = plt.subplots(int(height),int(width*2))

for i in range(numShapes):
    x,y = load(i)
    traj = learn(x,y)
    draw(x,y,traj,i,axs)
plt.show()
