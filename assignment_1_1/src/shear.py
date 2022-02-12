import numpy as np
from geometry import compute_mesh_centroid
# -----------------------------------------------------------------------------
#                               Mesh geometry
# -----------------------------------------------------------------------------

def shear_transformation(V, nu):
    """
    Computes vertices' position after the shear transformation.

    Input:
    - V : np.array (|V|, 3)
        The array of vertices positions.
        Contains the coordinates of the i-th vertex in i-th row
    - nu : the shear paramter
    
    Output:
    - V1 : np.array (|V|, 3)
        The array of vertices positions after transformation.
    """
    V1 = V.copy()

    # HW1 1.3.6
    shearing_transformation_matrix = np.array([[1, nu],
                                               [0, 1]])

    # we multiply the x, y coordinates by the transformation matrix
    # such that V_new = (x_i + nu * y_i, y_i)
    V1[:, :2] = np.dot(shearing_transformation_matrix, V1[:, :2].T).T

    return V1


def shear_equilibrium(V, F, x_csl):
    """
    Shear the input mesh to make it equilibrium.

    Input:
    - V : np.array (|V|, 3)
        The array of vertices positions.
        Contains the coordinates of the i-th vertex in i-th row
    - F : np.array (|F|, 3)
        The array of triangle faces.
    - x_csl: np.array (3, )
        The x coordinate of the target centroid
    Output:
    - V1 : np.array (|V|, 3)
        The array of vertices positions that are equilibrium.
    """
    # HW1 1.3.7

    # we need:                        x_new_mesh_centroid = x_csl  (1)
    # we can get that by shearing:    x_new_mesh_centroid = x_mesh_centroid + y_mesh_centroid * nu  (2)
    # rewriting (1) using (2):        x_mesh_centroid + y_mesh_centroid * nu = x_csl
    # which is equivalent to:         nu = (x_csl - x_mesh_centroid) / y_mesh_centroid

    mesh_centroid = compute_mesh_centroid(V, F)
    nu = (x_csl - mesh_centroid[0]) / mesh_centroid[1]

    return shear_transformation(V, nu)
