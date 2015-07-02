# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#          Alexander Fabisch <afabisch@informatik.uni-bremen.de>

import numpy as np
from .behavior import BlackBoxBehavior
from dmp import DMP


PERMITTED_METAPARAMETERS = ["x0", "g", "gd", "execution_time"]


class DMPBehavior(BlackBoxBehavior):
    """Dynamical Movement Primitive.

    Can be used to optimize the weights of a DMP with a black box optimizer.

    Parameters
    ----------
    dmp : string or DMP, optional (default: None)
        A DMP object or the name of a configuration file
    """
    def __init__(self, dmp=None):
        if dmp is None:
            self.dmp = DMP()
        elif isinstance(dmp, str):
            self.dmp = DMP.from_file(dmp)
        elif hasattr(dmp, "execute_step"):
            self.dmp = dmp
        else:
            raise ValueError("Unknown DMP type: %r" % dmp)

    def init(self, n_inputs, n_outputs):
        """Initialize the behavior.

        Parameters
        ----------
        n_inputs : int
            number of inputs

        n_outputs : int
            number of outputs
        """
        if n_inputs != n_outputs:
            raise ValueError("Input and output dimensions must match, got "
                             "%d inputs and %d outputs" % (n_inputs, n_outputs))

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.n_task_dims = self.n_inputs / 3

        if self.dmp.get_weights() is None:
            weights = np.zeros((self.n_task_dims, self.dmp.get_num_features()))
            self.dmp.set_weights(weights)

        self.x0 = None
        self.g = np.zeros(self.n_task_dims)
        self.gd = None

        self.x = np.empty(self.n_task_dims)
        self.v = np.empty(self.n_task_dims)
        self.a = np.empty(self.n_task_dims)


    def set_meta_parameters(self, keys, meta_parameters):
        """Set DMP meta parameters.

        Permitted meta-parameters:

        x0 : array
            Initial position

        g : array
            Goal

        gd : array
            Velocity at the goal

        execution_time : float
            New execution time

        Parameters
        ----------
        keys : list of string
            names of meta-parameters

        meta_parameters : list of float
            values of meta-parameters
        """
        for key, meta_parameter in zip(keys, meta_parameters):
            if key not in PERMITTED_METAPARAMETERS:
                raise ValueError("Meta parameter '%s' is not allowed, use "
                                 "one of %r" % (key, PERMITTED_METAPARAMETERS))
            setattr(self, key, meta_parameter)
        self.dmp.set_metaparameters(keys, meta_parameters)

    def set_inputs(self, inputs):
        n_task_dims = len(inputs) / 3
        self.x[:] = inputs[:n_task_dims]
        self.v[:] = inputs[n_task_dims:-n_task_dims]
        self.a[:] = inputs[-n_task_dims:]

        if self.x0 is None:
            self.x0 = self.x.copy()

    def get_outputs(self, outputs):
        n_task_dims = len(outputs) / 3
        outputs[:n_task_dims] = self.x[:]
        outputs[n_task_dims:-n_task_dims] = self.v[:]
        outputs[-n_task_dims:] = self.a[:]

    def step(self):
        if self.n_task_dims == 0:
            return

        if self.dmp.can_step():
            self.x, self.v, self.a = self.dmp.execute_step(
                self.x, self.v, self.a, self.x0, self.g, self.gd)
        else:
            self.v[:] = 0.0
            self.a[:] = 0.0

    def can_step(self):
        return self.dmp.can_step()

    def get_n_params(self):
        return self.dmp.get_weights().size

    def get_params(self):
        return self.dmp.get_weights().ravel()

    def set_params(self, params):
        self.dmp.set_weights(params)

    def reset(self):
        self.dmp.reset()
        self.x0 = None
