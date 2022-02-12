import numpy as np
from scipy import sparse

# -----------------------------------------------------------------------------
#                       Functions from assignment_1_1
# -----------------------------------------------------------------------------

from geometry import compute_mesh_centroid
from geometry import compute_faces_centroid
from geometry import compute_faces_area
from geometry import compute_mesh_area

# -----------------------------------------------------------------------------
#                            Provided functions
# -----------------------------------------------------------------------------

def vertex_cells_sum(values, cells):
    """
    Sums values at vertices from each incident n-cell.

    Input:
    - values : np.array (#cells,) or (#cells, n)
        The cell values to be summed at vertices.
        If shape (#cells,): The value is per-cell,
        If shape (#cells, n): The value refers to the corresponding cell vertex.
    - cells : np.array (#cells, n)
        The array of cells.
    Output:
    - v_sum : np.array (#vertices,)
        A vector with the sum at each i-th vertex in i-th position.
    Note:
        If n = 2 the cell is an edge, if n = 3 the cell is a triangle,
        if n = 4 the cell can be a tetrahedron or a quadrilateral, depending on
        the data structure.
    """
    i = cells.flatten('F')  # i.shape = (|cells| * n) // example: [[1, 2][3, 4]] -> [1, 3, 2, 4]
    j = np.arange(len(cells))   # j.shape = (|cells|, )
    j = np.tile(j, cells.shape[1])  # j.shape = cells.shape[0] * cells.shape[1]  // if cells.shape = (2, 3) -> j = [0, 1, 0, 1, 0, 1]
    v = values.flatten('F')  # v.shape = (|cells| * max(n, 1))
    # if values came in one dimensional
    if len(v) == len(cells):
        v = v[j]

    # i/j - row/column indexes of final matrix to fill (if duplicate indexes then values are summed) with data from v
    v_sum = sparse.coo_matrix((v, (i, j)), (np.max(cells) + 1, len(cells)))  # v_sum.shape = (np.max(cells) + 1, len(cells)
    return np.array(v_sum.sum(axis=1)).flatten()


def compute_edges_length(V, E):
    """
    Computes the edge length of each mesh edge.

    Input:
    - V : np.array (#V, 3)
        The array of vertices positions.
        Contains the coordinates of the i-th vertex in i-th row
    - E : np.array (#edges, 2)
        The array of mesh edges.
        generated from the function E = igl.edges(F)
        returns an array (#edges, 2) where i-th row contains the indices of the two vertices of i-th edge.
    Output:
    - l : np.array (#edges,)
        The edge lengths.
    """

    l = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    return l


# -----------------------------------------------------------------------------
#                           2.5.1  Target energies
# -----------------------------------------------------------------------------


def compute_equilibrium_energy(V, F, x_csl):
    """
    Computes the equilibrium energy E_eq = 1/2*(x_cm - x_csl)^2.

    Input:
    -- V : np.array (#V, 3)
        The array of vertices positions.
        Contains the coordinates of the i-th vertex in i-th row
    - F : np.array (#F, 3)
        The array of triangle faces.
    - x_csl : float
        The x coordinate of the center of the support line.
    Output:
    - E_eq : float
        the equilibrium energy of the mesh with respect to the target centroid
        x_csl.
    """

    # we take the x coordinate of the mesh centroid
    x_cm = compute_mesh_centroid(V, F)[0]

    E_eq = 1/2 * np.square(x_cm - x_csl)
    return E_eq


def compute_shape_energy(V, E, L):
    """
    Computes the energy E_sh = 1/2 sum_e (l_e - L_e)^2 in the current
    configuration V, where l_e is the length of mesh edges, and L_e the
    corresponding length in the undeformed configuration.

    Input:
    - V : np.array (#V, 3)
        The array of vertices positions.
        Contains the coordinates of the i-th vertex in i-th row
    - E : np.array (#edges, 2)
        The array of mesh edges.
    - L : np.array (#edges,)
        The rest lengths of mesh edges.
    Output:
    - E_sh : float
        The energy E_sh in the current configuration V.
    """

    E_sh = 1/2 * np.sum(np.square(L - compute_edges_length(V, E)))
    return E_sh


# -----------------------------------------------------------------------------
#                         2.5.2  Faces area gradient
# -----------------------------------------------------------------------------


def compute_faces_area_gradient(V, F):
    """
    Computes the gradient of faces area.

    Input:
    - V : np.array (#V, 3)
        The array of vertices positions.
        Contains the coordinates of the i-th vertex in i-th row
    - F : np.array (#F, 3)
        The array of triangle faces.
    Output:
    - dA_x : np.array (#F, 3)
        The gradient of faces areas A_i with respect to the x coordinate of each
        face vertex x_1, x_2, and x_3, with (i,0) = dA_i/dx_1, (i,1) = dA_i/dx_2,
        and (i,2) = dA_i/dx_3.
    - dA_y : np.array (#F, 3)
        The gradient of faces areas A_i with respect to the y coordinate of each
        face vertex y_1, y_2, and y_3, with (i,0) = dA_i/dy_1, (i,1) = dA_i/dy_2,
        and (i,2) = dA_i/dy_3.
    """

    # define a rotation matrix for 90 degree rotation
    M_rot = np.array([[0, -1],
                      [1, 0]])

    # take only x and y coordinates and rotate them by 90 degrees
    V_rot_xy = V[:, :2] @ M_rot

    # get according vectors by subtracting the coordinates of the start point from the coordinates of the end point
    # rotated_edge_vectors array is of shape (#F, 2)
    rotated_edge_vectors = np.stack((V_rot_xy[F[:, 1]] - V_rot_xy[F[:, 2]],
                                     V_rot_xy[F[:, 2]] - V_rot_xy[F[:, 0]],
                                     V_rot_xy[F[:, 0]] - V_rot_xy[F[:, 1]]))

    # compute the gradients w.r.t. both x and y
    dA_xy = 1/2 * rotated_edge_vectors.T

    dA_x, dA_y = dA_xy

    return dA_x, dA_y


# -----------------------------------------------------------------------------
#                      2.5.3  Equilibrium energy gradient
# -----------------------------------------------------------------------------


def compute_equilibrium_energy_gradient(V, F, x_csl):
    """
    Computes the gradient of the energy E_eq = 1/2*(x_cm - x_csl)^2 with respect
    to the x and y coordinates of each vertex, where x_cm is the x coordinate
    of the area centroid and x_csl x coordinate of the center of the support
    line.

    Input:
    - V : np.array (#V, 3)
        The array of vertices positions.
        Contains the coordinates of the i-th vertex in i-th row
    - F : np.array (#F, 3)
        The array of triangle faces.
    - x_csl : float
        The x coordinate of the center of the support line.
    Output:
    1- grad_E_eq : np.array (#V, 2)
        The gradient of the energy E_eq with respect to vertices v_i,
        with (i, 0) = dE_eq/dx_i and (i, 1) = dE_eq/dy_i
    """

    def compute_coo_matrix(values, F):
        i = F.flatten('F')  # i.shape = (|cells| * n) // example: [[1, 2][3, 4]] -> [1, 3, 2, 4]
        j = np.arange(len(F))  # j.shape = (|cells|, )
        j = np.tile(j, F.shape[1])  # j.shape = cells.shape[0] * cells.shape[1]  // if cells.shape = (2, 3) -> j = [0, 1, 0, 1, 0, 1]
        v = values.flatten('F')
        return sparse.coo_matrix((v, (i, j)), (np.max(F) + 1, len(F)))

    grad_E_eq = np.zeros((V.shape[0], 2))

    x_cm = compute_mesh_centroid(V, F)[0]
    mesh_area = compute_mesh_area(V, F)

    x_faces_centroids = compute_faces_centroid(V, F)[:, 0]  # scalar 
    x_centroid_diff = x_faces_centroids - x_cm  # shape (|F|, )

    x_faces_ares = compute_faces_area(V, F)  # shape (|F|)

    x_faces_area_gradients, y_faces_area_gradients = compute_faces_area_gradient(V, F)  # shape = (|F|, 3), (|F|, 3)

    # co-occurrence matrices of shape (|V|, |F|) with: M[i][j] = grad if V[i] in F[j],
    #                                                  M[i][j] = 0 otherwise
    x_faces_area_grad_coo_m: sparse.coo_matrix = compute_coo_matrix(x_faces_area_gradients, F)
    y_faces_area_grad_coo_m: sparse.coo_matrix = compute_coo_matrix(y_faces_area_gradients, F)

    # shape of each of first terms is (|V|, (1?))
    x_first_term = x_faces_area_grad_coo_m @ x_centroid_diff
    y_first_term = y_faces_area_grad_coo_m @ x_centroid_diff

    # co-occurrence matrices of shape (|V|, |F|) with: M[i][j] = 1 if V[i] in F[j],
    #                                                  M[i][j] = 0 otherwise
    x_faces_centroid_grad: sparse.coo_matrix = 1/3 * compute_coo_matrix(np.ones_like(x_faces_area_gradients), F)  # shape = (|V|, |F|)

    # we only do this for x, since d(x_f)/d(V) would have a column of zeros for y coordinate
    x_second_term = x_faces_centroid_grad @ x_faces_ares  # shape (|V|, (1?))

    scaling_constant = (x_cm - x_csl) / mesh_area

    grad_E_eq[:, 0] = x_first_term + x_second_term
    grad_E_eq[:, 1] = y_first_term

    return scaling_constant * grad_E_eq

# -----------------------------------------------------------------------------
#                          2.5.4 Shape energy gradient
# -----------------------------------------------------------------------------


def compute_shape_energy_gradient(V, E, L):
    """
    Computes the gradient of the energy E_sh = 1/2 sum_e (l_e - L_e)^2 with
    respect to the x and y coordinates of each vertex, where l_e is the length
    of mesh edges, and L_e the corresponding length in the undeformed
    configuration.

    Input:
    - V : np.array (#V, 3)
        The array of vertices positions.
        Contains the coordinates of the i-th vertex in i-th row
    - E : np.array (#edges, 2)
        The array of mesh edges.
    - L : np.array (#edges,)
        The rest lengths of mesh edges.
    Output:
    - grad_E_sh : np.array (#V, 2)
        The gradient of the energy E_sh with respect to vertices v_i,
        with (i, 0) = dE_sh/dx_i, (i, 1) = dE_sh/dy_i
    """
    grad_E_sh = np.zeros((len(V), 2))
    return grad_E_sh
