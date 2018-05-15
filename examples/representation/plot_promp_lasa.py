"""
================================
LASA Handwriting with ProMPs
================================

The LASA Handwriting dataset learned with ProMPs. 
"""
print(__doc__)

import numpy as np
from bolero.datasets import load_lasa
from bolero.representation import ProMPBehavior
import matplotlib.pyplot as plt

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
