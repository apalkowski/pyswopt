"""
Copyright (c) 2015 Aleksander Palkowski.

CS
"""
import time

import numpy as np

from scipy.stats import levy_stable

from .base import BaseAgent, BaseSwarmAlgorithm
from ..basic_types import OptimizationResult
from ..utils.validation import check_domain


class Nest(BaseAgent):

    """Swarm agent in the Cuckoo Search algorithm."""

    def __init__(self):
        """."""
        super(Nest, self).__init__()


class CuckooSearch(BaseSwarmAlgorithm):

    """Cuckoo Search algorithm optimization.

    The algorithm performs a continuous space search using Levy flights
    mechanism to find new solutions.

    References
    ----------
    .. [1] Yang, X.-S. and Deb, S. (2009)
           'Cuckoo Search via Lévy flights',
           in 2009 World Congress on Nature & Biologically
           Inspired Computing (NaBIC). IEEE, pp. 210-214.

    .. [2] Yang, X.-S. and Deb, S. (2014)
           'Cuckoo search: recent advances and applications',
           Neural Computing and Applications, 24(1), pp. 169-174.
    """

    def __init__(self, pop_size=25, drate=0.75, random_state=None,
                 step_size=1, alpha=1.5, beta=0):
        """CuckooSearch algorithm constructor.

        Parameters
        ----------
        pop_size : int, optional
            Population size of the swarm of agents.

        disc_rate : float, optional
            Percentage discovery rate for the nests to be discarded.
            Should be in range of [0, 1).

        random_state : RandomState or int, optional
            Pseudo-random number generator to use for random sampling.
            When not provided, a local numpy.random.RandomState
            instance is created.

        step_size : float, optional
            Step multiplier for tuning agent's local walk step length.

        alpha : float, optional
            The alpha parameter for Levy distribution sampling.
            Should not be changed, unless other than a Levy flight
            behavior is desired.

        beta : float, optional
            The beta parameter for Levy distribution sampling.
            Should not be changed, unless other than a Levy flight
            behavior is desired.
        """
        super(CuckooSearch, self).__init__(pop_size, random_state)

        if drate < 0 or drate >= 1:
            raise ValueError('Incorrect discovery rate. '
                             'Should be in range of [0, 1).')
        self.discovery_rate = drate
        self.step_size = step_size
        self.alpha = alpha
        self.beta = beta
        self._name = 'Cuckoo Search'

    # TODO: add tolerance criterium
    def minimize(self, obj_fun, domain, max_iter=100, tol=None,
                 verbose=False):
        """Return the minimum of the given objective function."""
        def _global_walk():
            new_step = levy_stable.rvs(self.alpha, self.beta,
                                       size=(self.pop_size, domain.ndim),
                                       scale=step_scale,
                                       random_state=self.random_state)
            for n, nest in enumerate(self.agents):
                nest.custom_step(obj_fun, domain, new_step[n, :])

        def _local_walk():
            discard = \
                self.random_state.rand(self.pop_size) < self.discovery_rate
            perm_nests_1 = \
                [self.agents[i].solution
                 for i in self.random_state.permutation(self.pop_size)]
            perm_nests_2 = \
                [self.agents[i].solution
                 for i in self.random_state.permutation(self.pop_size)]
            new_step = np.subtract(perm_nests_1, perm_nests_2) * self.step_size
            for n, nest in enumerate(self.agents):
                if discard[n]:
                    nest.custom_step(obj_fun, domain, new_step[n, :])

        domain = check_domain(domain)
        self._init_population(obj_fun, domain)
        best_solution = []
        best_fitness = np.inf
        best_history = []
        iter_n = 1
        step_scale = 0.01 * (domain.ub - domain.lb)
        start_time = time.perf_counter()
        while iter_n <= max_iter:
            _global_walk()
            _local_walk()
            best_solution, best_fitness = \
                self._get_best_solution(best_solution, best_fitness)
            best_history.append(best_fitness)
            if verbose:
                self._print_status_msg(iter_n, best_fitness)
            iter_n += 1
        perf_time = time.perf_counter() - start_time
        return OptimizationResult(x=best_solution, fx=best_fitness,
                                  runtime=perf_time)

    def _gen_random_agent(self, fobj, domain):
        new_nest = Nest()
        new_nest.solution = domain.lb + (domain.ub - domain.lb) * \
            self.random_state.rand(domain.ndim)
        new_nest.fitness = fobj(new_nest.solution)
        return new_nest
