"""."""
import numpy as np


class OptimizationResult(object):

    """Container for results of an optimization process."""

    def __init__(self, x=None, fx=None, runtime=None):
        """OptimizationResult constructor."""
        self.x = x
        self.fx = fx
        self.runtime = runtime


# TODO: add distinction between continuous, binary or integer variables.
class SearchDomain(object):

    """Search domain description.

    Object that defines the desired search domain size
    for an optimization problem.
    """

    def __init__(self, lower_bound, upper_bound):
        """SearchDomain constructor.

        Parameters
        ----------
        upper_bound : float or array-like of float
            Upper boundaries for each dimension.

        lower_bound : float or array-like of float
            Lower boundaries for each dimension.
        """
        lower_bound = np.asarray(lower_bound)
        upper_bound = np.asarray(upper_bound)

        if upper_bound.size != lower_bound.size:
            raise ValueError('Uneven dimensions of given boundaries.')

        if np.any(lower_bound >= upper_bound):
            raise ValueError('Not all lower boundaries are lesser '
                             'than upper boundaries')

        self.lb = lower_bound[:]
        self.ub = upper_bound[:]
        self.ndim = upper_bound.size
