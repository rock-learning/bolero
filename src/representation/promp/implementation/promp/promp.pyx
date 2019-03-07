from libcpp cimport bool
from libcpp.vector cimport vector
cimport numpy as np
import numpy as np
from cython.operator cimport dereference as deref
cimport _declarations as cpp


cdef class TrajectoryData:
    cdef cpp.TrajectoryData * thisptr
    cdef bool delete_thisptr

    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = True

    def __dealloc__(self):
        if self.delete_thisptr and self.thisptr != NULL:
            del self.thisptr

    def __init__(TrajectoryData self, int numBF, int numDim, bool isStroke, double overlap):
        cdef int cpp_numBF = numBF
        cdef int cpp_numDim = numDim
        cdef bool cpp_isStroke = isStroke
        cdef double cpp_overlap = overlap
        self.thisptr = new cpp.TrajectoryData(cpp_numBF, cpp_numDim, cpp_isStroke, cpp_overlap)

    mean_ = property(__get_mean_, __set_mean_)

    cpdef __get_mean_(TrajectoryData self):
        cdef vector[double] result = self.thisptr.mean_
        return result


    cpdef __set_mean_(TrajectoryData self, object mean_):
        cdef vector[double] cpp_mean_ = mean_
        self.thisptr.mean_ = cpp_mean_

    covariance_ = property(__get_covariance_, __set_covariance_)

    cpdef __get_covariance_(TrajectoryData self):
        cdef vector[double] result = self.thisptr.covariance_
        return result


    cpdef __set_covariance_(TrajectoryData self, object covariance_):
        cdef vector[double] cpp_covariance_ = covariance_
        self.thisptr.covariance_ = cpp_covariance_

    conditions_ = property(__get_conditions_, __set_conditions_)

    cpdef __get_conditions_(TrajectoryData self):
        cdef vector[double] result = self.thisptr.conditions_
        return result


    cpdef __set_conditions_(TrajectoryData self, object conditions_):
        cdef vector[double] cpp_conditions_ = conditions_
        self.thisptr.conditions_ = cpp_conditions_

    iteration_limit_ = property(__get_iteration_limit_, __set_iteration_limit_)

    cpdef __get_iteration_limit_(TrajectoryData self):
        cdef int result = self.thisptr.iterationLimit_
        return result


    cpdef __set_iteration_limit_(TrajectoryData self, int iterationLimit_):
        cdef int cpp_iterationLimit_ = iterationLimit_
        self.thisptr.iterationLimit_ = cpp_iterationLimit_

    num_b_f_ = property(__get_num_b_f_, __set_num_b_f_)

    cpdef __get_num_b_f_(TrajectoryData self):
        cdef int result = self.thisptr.numBF_
        return result


    cpdef __set_num_b_f_(TrajectoryData self, int numBF_):
        cdef int cpp_numBF_ = numBF_
        self.thisptr.numBF_ = cpp_numBF_

    num_dim_ = property(__get_num_dim_, __set_num_dim_)

    cpdef __get_num_dim_(TrajectoryData self):
        cdef int result = self.thisptr.numDim_
        return result


    cpdef __set_num_dim_(TrajectoryData self, int numDim_):
        cdef int cpp_numDim_ = numDim_
        self.thisptr.numDim_ = cpp_numDim_

    is_stroke_ = property(__get_is_stroke_, __set_is_stroke_)

    cpdef __get_is_stroke_(TrajectoryData self):
        cdef bool result = self.thisptr.isStroke_
        return result


    cpdef __set_is_stroke_(TrajectoryData self, bool isStroke_):
        cdef bool cpp_isStroke_ = isStroke_
        self.thisptr.isStroke_ = cpp_isStroke_

    overlap_ = property(__get_overlap_, __set_overlap_)

    cpdef __get_overlap_(TrajectoryData self):
        cdef double result = self.thisptr.overlap_
        return result


    cpdef __set_overlap_(TrajectoryData self, double overlap_):
        cdef double cpp_overlap_ = overlap_
        self.thisptr.overlap_ = cpp_overlap_

    random_state_ = property(__get_random_state_, __set_random_state_)

    cpdef __get_random_state_(TrajectoryData self):
        cdef unsigned int result = self.thisptr.randomState_
        return result


    cpdef __set_random_state_(TrajectoryData self, unsigned int randomState_):
        cdef unsigned int cpp_randomState_ = randomState_
        self.thisptr.randomState_ = cpp_randomState_


    cpdef sample_trajectory_data(TrajectoryData self, TrajectoryData traj):
        cdef cpp.TrajectoryData * cpp_traj = traj.thisptr
        self.thisptr.sampleTrajectoryData(deref(cpp_traj))

    cpdef step_cov(TrajectoryData self, double timestamp, np.ndarray[double, ndim=1] covs):
        cdef double cpp_timestamp = timestamp
        self.thisptr.stepCov(cpp_timestamp, &covs[0], covs.shape[0])

    cpdef step(TrajectoryData self, double timestamp, np.ndarray[double, ndim=1] values):
        cdef double cpp_timestamp = timestamp
        self.thisptr.step(cpp_timestamp, &values[0], values.shape[0])

    cpdef imitate(TrajectoryData self, np.ndarray[double, ndim=1] sizes, np.ndarray[double, ndim=1] timestamps, np.ndarray[double, ndim=1] values):
        self.thisptr.imitate(&sizes[0], sizes.shape[0], &timestamps[0], timestamps.shape[0], &values[0], values.shape[0])

    cpdef get_values(TrajectoryData self, np.ndarray[double, ndim=1] timestamps, np.ndarray[double, ndim=1] means, np.ndarray[double, ndim=1] covars):
        self.thisptr.getValues(&timestamps[0], timestamps.shape[0], &means[0], means.shape[0], &covars[0], covars.shape[0])

    cpdef condition(TrajectoryData self, int count, np.ndarray[double, ndim=1] points):
        cdef int cpp_count = count
        self.thisptr.condition(cpp_count, &points[0], points.shape[0])

