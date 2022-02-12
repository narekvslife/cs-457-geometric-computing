import numpy as np
from scipy import sparse


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

    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]

    area = 0.5 * np.linalg.norm(np.cross(e1, e2, axis=1), axis=1)
    return area


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
        If n = 2 the cell is an edge
        If n = 3 the cell is a triangle
        if n = 4 the cell can be a tetrahedron or a quadrilateral, depending on the data structure.
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
