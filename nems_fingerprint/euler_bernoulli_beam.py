"""
Resonant Modes of Euler-Bernoulli Beams

Module contains functions returning the displacements of Euler-Bernoulli beam
modes. We assume that the cantilever is over the interval [0, 1]. All quantities
can be scaled appropriately from this fundamental dimensionless result. The
modes are numbered 1, 2, 3, ...

Author: Alex Nunn
Date: 27/10/22
"""

import functools
import numpy as np

from scipy.optimize import root_scalar
from scipy.integrate import quadrature


boundary_type_map = dict()


def register_boundary_type(key):
    def register_decorator(func):
        boundary_type_map[key] = func
        return func
    return register_decorator


def displacement(boundary_type, mode, x):
    r"""Return displacement of resonant mode at position x

    The normalisation of the resonant modes is chosen so that,

        \int_0^1 \phi_n(x)^2 dx = 1,

    where n \in \{1, 2, \dots \}

    Parameters
    ----------
    boundary_type: enum('clamped-free', 'clamped-clamped')
        type of boundary conditions satisfied by resonant mode
    mode : int | iterable
        index of resonant mode 1, 2, 3, ...
    x : float | ndarray(N,)
        position to evaluate displacement

    Returns
    -------
    displacement : ndarray(N,)
        displacement of resonant mode
    """
    try:
        single_mode_flag = False
        iter(mode)
    except TypeError:
        # Single mode
        mode = [mode]
        single_mode_flag = True

    params = np.array(
        [_mode_parameters(boundary_type, m) for m in mode]
    )
    beta = params[:, 0]
    a_values = params[:, 1:]
    y = beta[:, np.newaxis] * np.asanyarray(x)

    result = np.sum(a_values.T[..., np.newaxis] * np.stack(
        [np.sin(y), np.cos(y), np.sinh(y), np.cosh(y)]
    ), axis=0)

    # Ouput configuration
    single_position_flag = np.isscalar(x)
    mode_select = 0 if single_mode_flag else slice(None)
    position_select = 0 if single_position_flag else slice(None)
    selection = (mode_select, position_select)

    output = result[selection]
    return output


def displacement_deriv(boundary_type, mode, x):
    r"""Return derivative of the displacement of resonant mode at position x

    The normalisation of the resonant modes is chosen so that,

        \int_0^1 \phi_n(x)^2 dx = 1,

    where n \in \{1, 2, \dots \}

    Parameters
    ----------
    boundary_type: enum('clamped-free', 'clamped-clamped')
        type of boundary conditions satisfied by resonant mode
    mode : int | iterable
        index of resonant mode 1, 2, 3, ...
    x : float | ndarray(N,)
        position to evaluate displacement

    Returns
    -------
    displacement : ndarray(N,)
        displacement of resonant mode
    """
    try:
        single_mode_flag = False
        iter(mode)
    except TypeError:
        # Single mode
        mode = [mode]
        single_mode_flag = True

    params = np.array(
        [_mode_parameters(boundary_type, m) for m in mode]
    )
    beta = params[:, 0]
    a_values = params[:, 1:]
    y = beta[:, np.newaxis] * np.asanyarray(x)

    result = np.sum(a_values.T[..., np.newaxis] * np.stack(
        [np.cos(y), -np.sin(y), np.cosh(y), np.sinh(y)]
    ), axis=0)

    # Ouput configuration
    single_position_flag = np.isscalar(x)
    mode_select = 0 if single_mode_flag else slice(None)
    position_select = 0 if single_position_flag else slice(None)
    selection = (mode_select, position_select)

    output = beta[:, np.newaxis] * result[selection]
    return output


def resonant_frequency(boundary_type, mode):
    """Return resonant frequency of Euler-Bernoulli beam

    Parameters
    ----------
    boundary_type: enum('clamped-free', 'clamped-clamped')
        type of boundary conditions satisfied by resonant mode
    mode : int
        index of resonant mode 1, 2, 3, ...

    Returns
    -------
    float
        resonant frequency
    """
    beta, *_ = _mode_parameters(boundary_type, mode)
    return beta


@functools.lru_cache()
def _mode_parameters(boundary_type, mode):
    """Return parameter tupe for boundary type and mode

    The modes of an Euler-Bernoulli beam can be expressed uniquely as

    u(x) = a1 sin(beta x) + a2 cos(beta x) + a3 sinh(beta x) + a4 cosh(beta x)

    where x in [0, 1] and the constantst beta, a1, ..., a4 are determined by the
    mode number and the boundary conditions. This function returns these
    fundamental parameters for a given mode.

    Parameters
    ----------
    boundary_type: enum('clamped-free', 'clamped-clamped')
        type of boundary conditions satisfied by resonant mode
    mode : int
        index of resonant mode 1, 2, 3, ...

    Returns
    -------
    tuple
        (beta, a1, a2, a3, a4)
        fundamental parameters of resonant mode
    """
    if type(mode) is not int or mode < 1:
        raise ValueError(
            'mode parameter must be an int greater than or equal to 1'
        )

    try:
        return boundary_type_map[boundary_type](mode)
    except KeyError as e:
        raise ValueError(f"Unknown boundary type '{boundary_type}' encountered.")


@register_boundary_type('clamped-free')
def _mode_parameters_clamped_free(mode):
    """Return fundamental parameters for clamped-free configuration"""

    # Function and derivative for beta root finding
    f = lambda x: np.cos(x) + 1 / np.cosh(x)
    fp = lambda x: -np.sin(x) - np.sinh(x) / np.cosh(x) ** 2

    beta0 = np.pi * (mode - 0.5)
    root_result = root_scalar(f, x0=beta0, fprime=fp)
    beta = root_result.root

    trans_coeff = (np.sin(beta) + np.sinh(beta)) / (np.cos(beta) + np.cosh(beta))
    g = lambda x: (
        np.sin(beta * x) - np.sinh(beta * x)
        - trans_coeff * (np.cos(beta * x) - np.cosh(beta * x))
    ) ** 2

    quad_result, _ = quadrature(g, a=0, b=1)

    a1 = np.sqrt(1 / quad_result)
    a2 = -trans_coeff * a1
    a3 = -a1
    a4 = -a2

    return (beta, a1, a2, a3, a4)


@register_boundary_type('clamped-clamped')
def _mode_parameters_clamped_clamped(mode):

    # Function and derivative for beta root finding
    f = lambda x: np.cos(x) - 1 / np.cosh(x)
    fp = lambda x: -np.sin(x) + np.sinh(x) / np.cosh(x) ** 2

    beta0 = np.pi * (mode + 0.5)
    root_result = root_scalar(f, x0=beta0, fprime=fp)
    beta = root_result.root

    trans_coeff = (np.cos(beta) - np.cosh(beta)) / (np.sin(beta) + np.sinh(beta))
    g = lambda x: (
        np.sin(beta * x) - np.sinh(beta * x)
        + trans_coeff * (np.cos(beta * x) - np.cosh(beta * x))
    ) ** 2

    quad_result, _ = quadrature(g, a=0, b=1)

    a1 = np.sqrt(1 / quad_result)
    a2 = trans_coeff * a1
    a3 = -a1
    a4 = -a2

    return (beta, a1, a2, a3, a4)
