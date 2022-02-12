import igl
import numpy as np
import scipy.sparse.linalg
from scipy import sparse
from scipy.sparse import csr_matrix
import numpy.linalg as la
import geometry_helper


def compute_mass_matrix(V, F):
    '''
    Assemble the Mass matrix by computing quantities per face.

    Parameters:
    - V : np.array (#v, 3)
    - F : np.array (#f, 3)

    Returns:
    - M : scipy sparse diagonal matrix (#v, #v)
        Mass matrix
    '''

    # compute face area per face
    faces_area = geometry_helper.compute_faces_area(V, F)

    # compute the sum of adjacent face areas per vertex * 1/3
    vertex_masses = 1/3 * geometry_helper.vertex_cells_sum(faces_area, F)

    n_v = vertex_masses.shape[0]

    M = sparse.diags(vertex_masses, shape=(n_v, n_v), format='csr')
    return M


def compute_cotangent(a, b, c, A):
    '''
    Compute the cotangent of an angle in a triangle by using the triangle edge lengths and area only.
    The input parameters are defined in the handout figure.
    The purpose of this function is to check that your formula is correct.
    You should not directly use this in the `compute_laplacian_matrix` function.

    Parameters:
    - a : float
    - b : float
    - c : float
    - A : float
    '''

    return (a ** 2 + b ** 2 - c ** 2) / (4 * A)


def compute_laplacian_matrix(V, F):
    '''
    Assemble the Laplacian matrix by computing quantities per face.

    Parameters:
    - V : np.array (#v, 3)
    - F : np.array (#f, 3)

    Returns:
    - L : scipy sparse matrix (#v, #v)
        Laplacian matrix
    '''

    cot = np.zeros(F.shape)
    double_area = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]],
                                          V[F[:, 2]] - V[F[:, 1]]), axis=1)
    E1 = np.linalg.norm(V[F[:, 1]] - V[F[:, 0]], axis=1)
    E2 = np.linalg.norm(V[F[:, 2]] - V[F[:, 1]], axis=1)
    E3 = np.linalg.norm(V[F[:, 0]] - V[F[:, 2]], axis=1)

    cot[:, 0] = (E1**2 + E3**2 - E2**2) / (4 * double_area)
    cot[:, 1] = (E2**2 + E1**2 - E3**2) / (4 * double_area)
    cot[:, 2] = (E3**2 + E2**2 - E1**2) / (4 * double_area)

    i = np.hstack((F.flatten(), F.flatten()))
    j = np.hstack((np.roll(F, 1, axis=1).flatten(),
                   np.roll(F, -1, axis=1).flatten()))
    d = np.hstack((np.roll(cot, -1, axis=1).flatten(),
                   np.roll(cot, 1, axis=1).flatten()))
    L = sparse.coo_matrix((d, (i, j)), (len(V), len(V)))
    D = sparse.diags(np.array(L.sum(axis=1)).flatten(), shape=(len(V), len(V)))
    return L - D
