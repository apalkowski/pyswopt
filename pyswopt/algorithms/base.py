"""
Copyright (c) 2015 Aleksander Pałkowski.

@author: AP
"""
import numbers
import time
from abc import ABCMeta, abstractmethod
from operator import attrgetter

import numpy as np

from ..utils.validation import check_domain, check_int, check_random_state


class BaseAgent(metaclass=ABCMeta):

    """Abstract base class for all swarm agents."""

    def __init__(self):
        """BaseAgent constructor."""
        self.solution = []
        self.fitness = np.inf

    def __eq__(self, value):
        if isinstance(value, BaseAgent):
            return self.fitness == value.fitness
        if isinstance(value, numbers.Number):
            return self.fitness == value

    def __ne__(self, value):
        if isinstance(value, BaseAgent):
            return self.fitness != value.fitness
        if isinstance(value, numbers.Number):
            return self.fitness != value

    def __gt__(self, value):
        if isinstance(value, BaseAgent):
            return self.fitness > value.fitness
        if isinstance(value, numbers.Number):
            return self.fitness > value

    def __lt__(self, value):
        if isinstance(value, BaseAgent):
            return self.fitness < value.fitness
        if isinstance(value, numbers.Number):
            return self.fitness < value

    def __ge__(self, value):
        if isinstance(value, BaseAgent):
            return self.fitness >= value.fitness
        if isinstance(value, numbers.Number):
            return self.fitness >= value

    def __le__(self, value):
        if isinstance(value, BaseAgent):
            return self.fitness <= value.fitness
        if isinstance(value, numbers.Number):
            return self.fitness <= value

    def custom_step(self, obj_fun, domain, step_length):
        """Change agent's solution by moving him by a custom step."""
        new_solution = self.solution[:] + step_length
        is_lesser = new_solution < domain.lb
        new_solution[is_lesser] = domain.lb[is_lesser]
        is_greater = new_solution > domain.ub
        new_solution[is_greater] = domain.ub[is_greater]
        new_fitness = obj_fun(new_solution)
        if new_fitness < self.fitness:
            self.fitness = new_fitness
            self.solution = new_solution[:]


class BaseSwarmAlgorithm(metaclass=ABCMeta):

    """Abstract base class for all swarm algorithms."""

    def __init__(self, pop_size, random_state):
        """BaseSwarmAlgorithm constructor."""
        self.random_state = check_random_state(random_state)
        self.pop_size = check_int(pop_size)
        self.agents = []
        self._name = ''

    @abstractmethod
    def minimize(self, obj_fun, domain, max_iter=100, tol=None,
                 verbose=False):
        """Return the minimum of the given objective function."""
        pass

    def _sort_agents(self, descending=False):
        self.agents.sort(key=attrgetter('fitness'), reverse=descending)

    def _get_best_solution(self, best_solution=[], best_fitness=np.inf):
        fitness = [agent.fitness for agent in self.agents]
        if np.amin(fitness) < best_fitness:
            best_fitness = np.amin(fitness)
            best_solution = self.agents[np.argmin(fitness)].solution[:]
        return best_solution, best_fitness

    @abstractmethod
    def _gen_random_agent(self, *args):
        """Return a randomly generated swarm agent."""
        pass

    def _init_population(self, *args):
        self.agents = []
        for _ in range(self.pop_size):
            new_agent = self._gen_random_agent(*args)
            self.agents.append(new_agent)

    def _print_status_msg(self, iter_n, best_fitness):
        print(self._name, ':')
        print('After iteration no. ', iter_n, ': fitness = ', best_fitness)
