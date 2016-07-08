import numpy as np
from bolero.utils.validation import check_feedback
from nose.tools import assert_true, assert_raises


def test_check_feedback_inf():
    feedbacks = [0, 1, np.inf]
    assert_true(np.isinf(check_feedback(feedbacks, compute_sum=True,
                                        check_inf=False)))
    assert_raises(ValueError, check_feedback, feedbacks)

def test_check_feedback_nan():
    feedbacks = [0, 1, np.nan]
    assert_true(np.isnan(check_feedback(feedbacks, compute_sum=True,
                                        check_nan=False)))
    assert_raises(ValueError, check_feedback, feedbacks)
