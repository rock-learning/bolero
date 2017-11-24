# Author: Hans Hohenfeld <hans.hohenfeld@dfki.de>

import numpy as np

from .optimizer import Optimizer


class MOREOptimizer(Optimizer)
    def __init__(self, parameters)
        """ TODO: Implement """
       
    def init(self, n_params):
        """Initialize the behavior search.

        Parameters
        ----------
        n_params : int
            dimension of the parameter vector
        """

    def get_next_parameters(self, params):
        """Get next individual/parameter vector for evaluation.

        Parameters
        ----------
        params : array_like, shape (n_params,)
            Parameter vector, will be modified
        """

    def set_evaluation_feedback(self, rewards):
        """Set feedbacks for the parameter vector.

        Parameters
        ----------
        rewards : list of float
            feedbacks for each step or for the episode, depends on the problem
        """

    def is_behavior_learning_done(self):
        """Check if the optimization is finished.

        Returns
        -------
        finished : bool
            Is the learning of a behavior finished?
        """

    def get_best_parameters(self):
        """Get best individual/parameter vector so far.

        Returns
        -------
        p : array_like, shape (n_params,)
            Best parameter vector so far
        """
