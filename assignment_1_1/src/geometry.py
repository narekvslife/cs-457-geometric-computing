import numpy as np
# -----------------------------------------------------------------------------
#                               Mesh geometry
# -----------------------------------------------------------------------------

def compute_faces_area(V, F):
    """
    Computes the area of the faces of a given triangle mesh (V, F).

    Input:
    - V : np.array (|V|, 3)
        The array of vertices positions.
        Contains the coordinates of the i-th vertex in i-th row
    - F : np.array (|F|, 3)
        The array of triangle faces.
    Output:
    - area : np.array (|F|,)
        The area of the faces. The i-th position contains the area of the i-th
        face.
    """
    area = np.zeros(F.shape[0])

    def get_area(v_1, v_2, v_3):
        return 0.5 * ((v_1[0] * (v_2[1] - v_3[1])) +
                      (v_2[0] * (v_3[1] - v_1[1])) +
                      (v_3[0] * (v_1[1] - v_2[1])))
    
    # HW1 1.3.3
    # enter your code here

    for face_n in range(F.shape[0]):
        # getting indices of vertices
        v1_ind, v2_ind, v3_ind = F[face_n]

        # getting coordinates of according vertices
        v1, v2, v3 = V[v1_ind],  V[v2_ind],  V[v3_ind]

        area[face_n] = get_area(v1, v2, v3)

    return area


def compute_mesh_area(V, F):
    """
    Computes the area of a given triangle mesh (V, F).

    Input:
    - V : np.array (|V|, 3)
        The array of vertices positions.
        Contains the coordinates of the i-th vertex in i-th row
    - F : np.array (|F|, 3)
        The array of triangle faces.
    Output:
    - area : float
        The area of the mesh.
    """
    # HW1 1.3.3

    return np.sum(compute_faces_area(V, F))


def compute_faces_centroid(V, F):
    """
    Computes the area centroid of each face of a given triangle mesh (V, F).

    Input:
    - V : np.array (|V|, 3)
        The array of vertices positions.
        Contains the coordinates of the i-th vertex in i-th row
    - F : np.array (|F|, 3)
        The array of triangle faces.
    Output:
    - cf : np.array (|F|, 3)
        The area centroid of the faces.
    """
    cf = np.zeros((F.shape[0], 3))

    # HW1 1.3.4
    # enter your code here

    for face_n in range(F.shape[0]):
        # getting indices of vertices
        v1_ind, v2_ind, v3_ind = F[face_n]

        # getting coordinates of according vertices
        v1, v2, v3 = V[v1_ind], V[v2_ind], V[v3_ind]

        # face centroid has the mean coordinates of it's vertices
        cf[face_n] = np.mean([v1, v2, v3], axis=0)
    return cf


def compute_mesh_centroid(V, F):
    """
    Computes the area centroid of a given triangle mesh (V, F).

    Input:
    - V : np.array (|V|, 3)
        The array of vertices positions.
        Contains the coordinates of the i-th vertex in i-th row
    - F : np.array (|F|, 3)
        The array of triangle faces.
    Output:
    - centroid : np.array (3,)
        The area centroid of the mesh.
    """

    # HW1 1.3.4

    mesh_area = compute_mesh_area(V, F)
    face_areas = compute_faces_area(V, F)
    face_centroids = compute_faces_centroid(V, F)

    # we take weighted sum of each face's centroid coordinates
    weighted_coordinate_sum = np.dot(face_areas,  face_centroids)

    # then we divide the weighted_coordinate_sum by the total mesh area to get the weighted average
    mc = weighted_coordinate_sum / mesh_area

    return mc


def compute_center_support_line(V):
    """
    Computes the x coordinate of the center of the support line

    Input:
    - V : np.array (|V|, 3)
        The array of vertices positions.
        Contains the coordinates of the i-th vertex in i-th row

    Output:
    - x_csl : float
        the x coordinate of the center of the support line
    """
    # HW1 1.3.5

    # vertices which lie on the ground have their y = 0  # TODO: maybe add some tolerance(?)
    ground_vertices = V[V[:, 1] == 0]

    # we take x coordinates of those vertices
    ground_vertices_x = ground_vertices[:, 0]

    x_min = np.min(ground_vertices_x)
    x_max = np.max(ground_vertices_x)

    return x_min + np.abs(x_max - x_min) / 2
