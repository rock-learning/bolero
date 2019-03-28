import numpy as np
from.controller import Controller
from ..utils.validation import check_feedback


class StepBasedController(Controller):
    def __init__(self, config=None, environment=None, behavior_search=None,
                 **kwargs):
        super(StepBasedController, self).__init__(
            config, environment, behavior_search, **kwargs)

    def learn(self, meta_parameter_keys=(), meta_parameters=()):
        """Learn the behavior.

        Parameters
        ----------
        meta_parameter_keys : list
            Meta parameter keys

        meta_parameters : list
            Meta parameter values

        Returns
        -------
        feedback_history : array, shape (n_episodes or less, dim_feedback)
            Feedbacks for each episode. If is_behavior_learning_done is True
            before the n_episodes is reached, the length of feedback_history
            is shorter than n_episodes.
        """
        feedback_history = []
        for _ in range(self.n_episodes):
            feedbacks = self.episode(meta_parameter_keys, meta_parameters)
            feedback_history.append(feedbacks)
            if (self.finish_after_convergence and
                    (self.behavior_search.is_behavior_learning_done() or
                     self.environment.is_behavior_learning_done())):
                break
        if self.verbose >= 2:
            print("[Controller] Terminated because of:\nbehavior_search: %s, "
                  "environment: %s"
                  % (self.behavior_search.is_behavior_learning_done(),
                     self.environment.is_behavior_learning_done()))
        return np.array(feedback_history)

    def episode(self, meta_parameter_keys=(), meta_parameters=()):
        """Execute one learning episode.

        Parameters
        ----------
        meta_parameter_keys : array-like, shape = (n_meta_parameters,)
            Meta parameter keys

        meta_parameters : array-like, shape = (n_meta_parameters,)
            Meta parameter values

        Returns
        -------
        accumulated_feedback : float or array-like, shape = (n_feedbacks,)
            Feedback(s) of the episode
        """
        if self.behavior_search is None:
            raise ValueError("A BehaviorSearch is required to execute an "
                             "episode without specifying a behavior.")

        if self.verbose >= 1:
            print("[Controller] Episode: #%d" % (self.episode_cnt + 1))

        behavior = self.behavior_search.get_next_behavior()
        feedbacks = self.episode_with(behavior, meta_parameter_keys,
                                      meta_parameters, learn=True)
        self.behavior_search.set_evaluation_feedback(feedbacks)

        if self.verbose >= 2:
            if self.accumulate_feedbacks:
                print("[Controller] Accumulated feedback: %g"
                      % np.sum(feedbacks))
            else:
                print("[Controller] Feedbacks: %s"
                      % np.array_str(feedbacks, precision=4))

        self.episode_cnt += 1

        if self.do_test and self.episode_cnt % self.n_episodes_before_test == 0:
            self.test_results_.append(
                self._perform_test(meta_parameter_keys, meta_parameters))

        feedbacks = check_feedback(
            feedbacks, compute_sum=self.accumulate_feedbacks)

        return feedbacks

    def episode_with(self, behavior, meta_parameter_keys=[],
                     meta_parameters=[], record=True, learn=False):
        """Execute a behavior in the environment.

        Parameters
        ----------
        behavior : Behavior
            Fix behavior

        meta_parameter_keys : list, optional (default: [])
            Meta parameter keys

        meta_parameters : list, optional (default: [])
            Meta parameter values

        record : bool, optional (default: True)
            Record feedbacks or trajectories if activated

        learn : bool, optional (default: False)
            Use rewards to improve behavior

        Returns
        -------
        feedbacks : array, shape (n_steps,)
            Feedback for each step in the environment
        """
        behavior.set_meta_parameters(meta_parameter_keys, meta_parameters)
        self.environment.reset()

        if self.record_inputs:
            inputs = []
        if self.record_outputs:
            outputs = []

        feedbacks = []

        # Sense initial state
        self.environment.get_outputs(self.outputs)
        while not self.environment.is_evaluation_done():
            behavior.set_inputs(self.outputs)
            if behavior.can_step():
                behavior.step()
                behavior.get_outputs(self.inputs)
            # Act
            self.environment.set_inputs(self.inputs)
            self.environment.step_action()
            # Sense
            self.environment.get_outputs(self.outputs)
            reward = self.environment.get_feedback()
            if learn:
                self.behavior_search.set_evaluation_feedback(reward)
            feedbacks.extend(list(reward))

            if record:
                if self.record_inputs:
                    inputs.append(self.inputs.copy())
                if self.record_outputs:
                    outputs.append(self.outputs.copy())

        if record:
            if self.record_inputs:
                self.inputs_.append(inputs)
            if self.record_outputs:
                self.outputs_.append(outputs)
            if self.record_feedbacks:
                self.feedbacks_.append(feedbacks)
        return feedbacks