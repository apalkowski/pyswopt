u"""
Copyright (c) 2015 Aleksander Pałkowski.

A collection of optimization testing functions.

References
----------
.. [1] Surjanovic, S. and Bingham, D. (2013).
       Virtual Library of Simulation Experiments: Test Functions
       and Datasets. Retrieved July 10, 2015,
       from http://www.sfu.ca/~ssurjano.
"""
import numpy as np
from numpy import (abs, cos, e, exp, pi, sin, sqrt)


def ackley(xx, a=20.0, b=0.2, c=2*pi):
    """Ackley function.

    Single-objective d-dimensional optimization problem
    with many local minima.

    The function computed is:
    f(xx) = -a * exp(-b * sqrt(1/d * sum(x_i**2)))
            - exp(1/d * sum(cos(c*x_i))) + a + exp(1)

    Global minimum:
    f(xx*) = 0, at xx* = (0, ..., 0) and a = 20, b = 0.2, c = 2*PI

    Domain:
    The function is usually evaluated on a hypercube
    x_i in [-32.768, 32.768], for all i = 1, ..., d.
    However it may be restricted to a smaller domain.
    """
    xx = np.asarray(xx)
    d = xx.size
    f = -a*exp(-b*sqrt(1.0/d*np.sum(xx**2.0, axis=0))) - \
        exp(1.0/d*np.sum(cos(c*xx), axis=0)) + a + e
    return f


def eggholder(xx):
    """Eggholder function.

    Single-objective 2-dimensional optimization problem
    with many local minima.

    The function computed is:
    f(xx) = -(x_2+47)*sin(sqrt(|x_2+x_1/2+47|))
            -x_1*sin(sqrt(|x_1-(x_2+47)|))

    Global minimum:
    f(xx*) = -959.6407, at xx* = (512, 404.2319)

    Domain:
    The function is usually evaluated on a square
    x_i in [-512, 512], for all i = 1, 2.
    """
    xx = np.asarray(xx)
    f = -(xx[1]+47.0) * sin(sqrt(abs(xx[1]+xx[0]/2.0+47.0))) - \
        xx[0] * sin(sqrt(abs(xx[0]-(xx[1]+47.0))))
    return f


def rosenbrock(xx):
    """Rosenbrock function.

    Single-objective d-dimensional optimization problem
    with a valley-shaped surface.
    The function is unimodal with the global minimum lying in a narrow,
    parabolic valley.

    The function computed is:
    f(xx) = sum{from i=1 to d-1}[100*(x_{i+1} - x_i**2)**2
                                 + (x_i - 1)**2]

    Global minimum:
    f(xx*) = 0, at xx* = (1, ..., 1)

    Domain:
    The function is usually evaluated on a hypercube
    x_i in [-5, 10], for all i = 1, ..., d.
    However it may be restricted to a hypercube
    x_i in [-2.048, 2.048], for all i = 1, ..., d.
    """
    xx = np.asarray(xx)
    f = np.sum(100.0 * (xx[1:] - xx[:-1]**2.0)**2.0 + (xx[:-1] - 1)**2.0,
               axis=0)
    return f


def sphere(xx):
    """Sphere function.

    Single-objective d-dimensional optimization problem
    with a bowl-shaped surface.
    The function is continuous, convex and unimodal with d local minima.

    The function computed is:
    f(xx) = sum(x_i**2)

    Global minimum:
    f(xx*) = 0, at xx* = (0, ..., 0)

    Domain:
    The function is usually evaluated on a hypercube
    x_i in [-5.12, 5.12], for all i = 1, ..., d.
    """
    xx = np.asarray(xx)
    f = np.sum(xx**2.0, axis=0)
    return f


def stybtang(xx):
    """Styblinski-Tang function.

    Single-objective d-dimensional optimization problem.

    The function computed is:
    f(xx) = 1/2 * sum(x_i**4 - 16*x_i**2 + 5*x_i)

    Global minimum:
    f(xx*) = -39.16599*d, at xx* = (-2.903534, ..., -2.903534)

    Domain:
    The function is usually evaluated on a hypercube
    x_i in [-5, 5], for all i = 1, ..., d.
    """
    xx = np.asarray(xx)
    f = 0.5 * np.sum(xx**4.0 - 16.0*xx**2.0 + 5.0*xx, axis=0)
    return f
