"""."""
import numbers

import numpy as np

from ..basic_types import SearchDomain


def check_random_state(seed):
    """."""
    if seed is None:
        return np.random.RandomState()
    if isinstance(seed, np.random.RandomState):
        return seed
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState '
                     'instance' % seed)


def check_domain(domain):
    """."""
    if isinstance(domain, SearchDomain):
        return domain
    raise ValueError('%r is not a correct SearchDomain instance' % domain)


def check_int(number):
    """."""
    if isinstance(number, numbers.Integral):
        return number
    raise ValueError('%r is not an integer number' % number)
