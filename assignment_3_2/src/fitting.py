import numpy as np
import scipy.sparse

from scipy import sparse

from scipy.sparse import linalg

import igl

import smooth_surfaces

from utils import *


# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------

def compute_orthogonal_frames(N):
    """Computes an orthonormal frame {e1, e2, e3} at vertices x_i.

    Parameters:
    - N : np.array (|n|, 3)
        The i-th row contains a vector in direction e3 at x_i.

    Returns:
    - e1: np.array (|n|, 3)
        The i-th row contains the axis e1 at vertex x_i.
    - e2: np.array (|n|, 3)
        The i-th row contains the axis e2 at vertex x_i.
    - e3: np.array (|n|, 3)
        The i-th row contains the axis e3 at vertex x_i.
    """
    e3 = N / np.linalg.norm(N, axis=1, keepdims=True)
    e1 = np.zeros(e3.shape)
    e1[:, 0] = - e3[:, 1]
    e1[:, 1] = e3[:, 0]
    e1[np.where((e1[:, 0] == 0) & (e1[:, 1] == 0))[0], 1] = 1
    e1 = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(e3, e1)
    return e1, e2, e3


def vertex_double_rings(F):
    """Computes double rings of mesh vertices.

    Parameters:
    - F : np.array (|F|, 3)
        The array of triangle faces.

    Returns:
    - v_i : np.array (n, )
        The indices of the central vertices i.
    - v_j : np.array (n, )
        The indices of the vertices j connected to vertex i by at most two
        edges, such that v_j[k] belongs to the double ring of v_i[k].
    """
    M = igl.adjacency_matrix(F)
    vi, vj = M.nonzero()
    N = M[vj]
    vii, vjj = N.nonzero()
    L = sparse.coo_matrix((np.ones(len(vii)), (vii, np.arange(len(vii)))))
    k = np.array(L.sum(axis=1)).flatten().astype('i')
    vii = np.repeat(vi, k)
    M = sparse.coo_matrix((np.ones(len(vii)), (vii, vjj)), shape=M.shape)
    M = M.tolil()
    M.setdiag(0)
    return M.nonzero()


# -----------------------------------------------------------------------------
# OSCULATING PARABOLOID
# -----------------------------------------------------------------------------

def compute_osculating_paraboloids(V, F, e1, e2, e3):
    """
    Computes the coefficients of the osculating paraboloid at vertices x_i
    in local orthonormal coordinates with base {x_i; e1, e2, e3}, with
    eta(x,y) = a x^2 + b y^2 + c xy + d x + e y
    through least squares fitting. Try to vectorize this function.

    Parameters:
    - V : np.array (|V|, 3)
        The array of vertices positions.
        Contains the global coordinates of the vertex x_i in i-th row
    - F : np.array (|F|, 3)
        The array of triangle faces.
    - e1: np.array (|V|, 3)
        The i-th row contains the axis e1 at vertex x_i.
    - e2: np.array (|V|, 3)
        The i-th row contains the axis e2 at vertex x_i.
    - e3: np.array (|V|, 3)
        The i-th row contains the axis e3 at vertex x_i.

    Returns:
    - a : np.array (|V|, 5)
        The paraboloid coefficients. i-th row contains the coefficients
        [a, b, c, d, e] of the paraboloid at x_i.
    """
    # we compute the indices for the double ring vj at each vertex vi
    vi, vj = vertex_double_rings(F)

    Pj = V[vj] - V[vi]

    x = np.einsum('ij, ij -> i', Pj, e1[vi])
    y = np.einsum('ij, ij -> i', Pj, e2[vi])
    z = np.einsum('ij, ij -> i', Pj, e3[vi])

    n = vj.shape[0]

    i = np.arange(n)
    i = np.hstack((i, i, i, i, i))

    j = 5 * vi
    j = np.hstack((j, j+1, j+2, j+3, j+4))

    data = np.hstack((x**2, y**2, x * y, x, y))

    X = sparse.coo_matrix((data, (i, j)), shape=(n, 5 * len(V)))

    a = linalg.spsolve(X.T @ X, X.T @ z)

    return a.reshape((len(np.unique(vi)), 5))


def compute_osculating_paraboloid_first_derivatives(a):
    """Computes the first derivatives of the osculating paraboloid at vertices x_i
    in local orthonormal coordinates with base {x_i; e1, e2, e3}, with
    eta(x,y) = a x^2 + b y^2 + c xy + d x + e y,
    evaluated at the point x_i. Try to vectorize this function.

    Parameters:
    - a : np.array (|V|, 5)
        The paraboloid coefficients. i-th row contains the coefficients
        [a, b, c, d, e] of the paraboloid at x_i.

    Returns:
    - x_x : np.array (|V|, 3)
        The first derivatives x_x, where the i-th row contains the local (x,y,z)
        coordinates of the vector x_x(x_i).
    - x_y : np.array (|V|, 3)
        The second derivatives x_y, where the i-th row contains the local (x,y,z)
        coordinates of the vector x_y(x_i).
    """
    d, e = a[:, 3], a[:, 4]

    zeros = np.zeros(a.shape[0])
    ones = np.ones(a.shape[0])

    x_x = np.stack((ones, zeros, d)).T
    x_y = np.stack((zeros, ones, e)).T

    return x_x, x_y


def compute_osculating_paraboloid_second_derivatives(a):
    """Computes the second derivatives of the osculating paraboloid at vertices x_i
    in local orthonormal coordinates with base {x_i; e1, e2, e3}, with
    eta(x,y) = a x^2 + b y^2 + c xy + d x + e y,
    evaluated at the point x_i. Try to vectorize this function.

    Parameters:
    - a : np.array (|V|, 5)
        The paraboloid coefficients. i-th row contains the coefficients
        [a, b, c, d, e] of the paraboloid at x_i.

    Returns:
    - x_xx : np.array (|V|, 3)
        The second derivatives x_xx, where the i-th row contains the local (x,y,z)
        coordinates of the vector x_xx(x_i).
    - x_xy : np.array (|V|, 3)
        The second derivatives x_xy, where the i-th row contains the local (x,y,z)
        coordinates of the vector x_xy(x_i).
    - x_yy : np.array (|V|, 3)
        The second derivatives x_yy, where the i-th row contains the local (x,y,z)
        coordinates of the vector x_yy(x_i).
    """
    zeros = np.zeros(a.shape[0])

    a_, b, c = a[:, 0], a[:, 1], a[:, 2]
    x_xx = np.stack((zeros, zeros, 2 * a_)).T
    x_xy = np.stack((zeros, zeros, c * np.ones(a.shape[0]))).T
    x_yy = np.stack((zeros, zeros, 2 * b)).T

    return x_xx, x_xy, x_yy


def compute_mesh_principal_curvatures(V, F):
    """Computes the principal curvatures at mesh vertices v_i through quadratic
    fitting.

    Parameters:
    - V : np.array (|V|, 3)
        The array of vertices positions.
        Contains the global coordinates of the vertex x_i in i-th row
    - F : np.array (|F|, 3)
        The array of triangle faces.

    Returns:
    - k_1 : np.array (n)
        The min principal curvature. i-th element contains the curvature
        at vertex x_i.
    - k_2 : np.array (n)
        The max principal curvature. i-th element contains the curvature
        at vertex x_i.
    - d_1 : np.array (n, 3)
        The unitized principal curvature direction corresponding to k_1.
        The i-th row contains the global coordinates of d_1(x_i).
    - d_2 : np.array (n, 3)
        The unitized principal curvature direction corresponding to k_2.
        The i-th row contains the global coordinates of d_2(x_i).
    """
    # we compute a vertex normal with libigl and use it as local axis e3:
    N = igl.per_vertex_normals(V, F)
    # then we compute the local axes:
    e1, e2, e3 = compute_orthogonal_frames(N)

    a = compute_osculating_paraboloids(V, F, e1, e2, e3)

    x_x, x_y = compute_osculating_paraboloid_first_derivatives(a)
    x_xx, x_xy, x_yy = compute_osculating_paraboloid_second_derivatives(a)

    n = smooth_surfaces.compute_surface_normal(x_x, x_y)

    I = smooth_surfaces.compute_first_fundamental_form(x_x, x_y)
    II = smooth_surfaces.compute_second_fundamental_form(x_xx, x_xy, x_yy, n)
    S = smooth_surfaces.compute_shape_operator(I, II)
    k_1, k_2, d_1, d_2 = smooth_surfaces.compute_principal_curvatures(S, x_x, x_y)

    d1 = (np.einsum('i, ij -> ij', d_1[:, 0], e1) +
          np.einsum('i, ij -> ij', d_1[:, 1], e2) +
          np.einsum('i, ij -> ij', d_1[:, 2], e3))

    d2 = (np.einsum('i, ij -> ij', d_2[:, 0], e1) +
          np.einsum('i, ij -> ij', d_2[:, 1], e2) +
          np.einsum('i, ij -> ij', d_2[:, 2], e3))

    return k_1, k_2, d1, d2
