from energies import (compute_edges_length,
                      compute_equilibrium_energy, compute_shape_energy,
                      compute_equilibrium_energy_gradient, compute_shape_energy_gradient)

import numpy as np
import time
import igl


def compute_optimization_objective(V, F, E, x_csl, L0, w):
    """
    Compute the objective function of make-it-stand problem.
    E = E_equilibrium + w * E_shape

    Input:
    - V : np.array (#V, 3)
        The array of vertices positions.
        Contains the coordinates of the i-th vertex in i-th row
    - F : np.array (#F, 3)
        The array of triangle faces.
    - E : np.array (#edges, 2)
        The array of mesh edges.
    - x_csl : float
        The x coordinate of the center of the support line.
    - l0 : np.array (#edges,)
        The rest lengths of mesh edges.
    - w : float
        The weight for shape preservation energy.
    Output:
    - obj : float
        The value of the objective function.
    """

    E_equilibrium = compute_equilibrium_energy(V, F, x_csl)
    E_shape = compute_shape_energy(V, E, L0)

    return E_equilibrium + w * E_shape


def compute_optimization_objective_gradient(V, F, E, x_csl, L0, w):
    """
    Compute the gradient of the objective function of make-it-stand problem.
    D_E = D_E_equilibrium + w * D_E_shape

    Input:
    - V : np.array (#V, 3)
        The array of vertices positions.
        Contains the coordinates of the i-th vertex in i-th row
    - F : np.array (#F, 3)
        The array of triangle faces.
    - E : np.array (#edges, 2)
        The array of mesh edges.
    - x_csl : float
        The x coordinate of the center of the support line.
    - l0 : np.array (#edges,)
        The rest lengths of mesh edges.
    - w : float
        The weight for shape preservation energy.
    Output:
    - grad_obj : np.array (#V, 2)
        The gradient of objective function.
    """

    D_E_equilibrium = compute_equilibrium_energy_gradient(V, F, x_csl)
    D_E_shape = compute_shape_energy_gradient(V, E, L0)

    return D_E_equilibrium + w * D_E_shape


def fixed_step_gradient_descent(V, F, x_csl, w, theta, iters):
    """
    Find equilibrium shape by using fixed step gradient descent method

    Input:
    - V : np.array (#V, 3)
        The array of vertices positions.
        Contains the coordinates of the i-th vertex in i-th row
    - F : np.array (#F, 3)
        The array of triangle faces.
    - x_csl : float
        The x coordinate of the center of the support line.
    - w : float
        The weight for shape preservation energy.
    - theta : float
        The optimization step.
    - iters : int
        The number of iteration for gradient descent.

    Output:
    - V1 : np.array (#V, 3)
        The optimized mesh's vertices
    - F : np.array (#F, 3)
        The array of triangle faces.
    - energy: np.array(iters, 1)
        The objective function energy curve with respect to the number of iterations.
    - running_time: float
        The tot running time of the optimization
    """

    V1 = V.copy()

    # this function of libigl returns an array (#edges, 2) where i-th row
    # contains the indices of the two vertices of i-th edge.
    E = igl.edges(F)

    fix = np.where(V1[:, 1] < 1e-3)[0]

    L0 = compute_edges_length(V1, E)

    t0 = time.time()

    energy = []

    for i in range(iters):
        grad = compute_optimization_objective_gradient(V1, F, E, x_csl, L0, w)  # shape = (|V|, 2)

        obj = compute_optimization_objective(V1, F, E, x_csl, L0, w)  # shape = scalar

        energy.append(obj)

        grad[fix] = 0

        V1 = V1 - theta * grad

    running_time = time.time() - t0

    return [V1, F, energy, running_time]