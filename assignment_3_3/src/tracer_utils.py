import igl
import numpy as np
import math
import numpy.linalg as la
from tracer_helper import rotate_vector


def asymptotic_path(idx, mesh, num_steps, step_size, first_principal_direction, num_neighbors, sampling_dist=0):
    """
    Computes both tracing direction (backward and forward) following an asymptotic path.
    Try to compute an asymptotic path with both tracing directions 

    Inputs:
    - idx : int
        The index of the vertex on the mesh to start tracing.
    - mesh : Mesh
        The mesh for tracing.
    - num_steps : int
        The number of tracing steps.
    - step_size : int
        The size of the projection at each tracing step.
    - first_principal_direction : bool
        Indicator for using the first principal curvature to do the tracing.
    - num_neighbors : int
        Number of closest vertices to consider for avering principal curvatures.
    - sampling_distance : float
        The distance to sample points on the path (For the design interface - Don't need to be define).
            
    Outputs:
    - P : np.array (n, 3)
        The ordered set of unique points representing a full asymptotic path.
    - A : np.array (n,)
        The ordered set of deviated angles in degrees calculated for the asymptotic directions.
    - PP : np.array (m,3)
        The ordered set of points laying on the path spaced with a given distance (For the design interface).
    """
        
    P_f, A_f, PP_f = trace(idx, mesh, num_steps, step_size, first_principal_direction,
                           False, num_neighbors, sampling_dist)

    P_b, A_b, PP_b = trace(idx, mesh, num_steps, step_size, first_principal_direction,
                           True, num_neighbors, sampling_dist)
    P_f = P_f[::-1][:-1]
    A_f = A_f[::-1][:-1]
    PP_f = PP_f[::-1][:-1]

    P = np.append(P_f, P_b, axis=0)
    A = np.append(A_f, A_b, axis=0)
    PP = np.append(PP_f, PP_b, axis=0)
    return P, A, PP


def trace(idx, mesh, num_steps, step_size, first_principal_direction, trace_backwards, num_neighbors, sampling_dist=0):
    """
    Computes one tracing direction following an asymptotic path.
    Try to compute the points on the asymptotic path.

    Inputs:
    - idx : int
        The index of the vertex on the mesh to start tracing.
    - mesh : Mesh
        The mesh for tracing.
    - num_steps : int
        The number of tracing steps.
    - step_size : int
        The size of the projection at each tracing step.
    - first_principal_direction : bool
        Indicator for using the first principal curvature to do the tracing.
    - trace_backwards : bool
        Indicator for mirroring the deviated angle
    - num_neighbors : int
        Number of closest vertices to consider for averaging principal curvatures.
    - sampling_distance : float
        The distance to sample points on the path (For the design interface - Don't need to be define).
            
    Outputs:
    - P : np.array (n, 3)
        The ordered set of points representing one tracing direction.
    - A : np.array (n,)
        The ordered set of deviated angles calculated for the asymptotic directions.
    - PP : np.array (m,3)
        The ordered set of points laying on the path spaced with a given distance (For the design interface).
    """

    P = np.empty((0, 3), float)
    PP = np.empty((0, 3), float)
    A = np.array([], float)

    # Get the data of the first vertex in the path
    pt = mesh.V[idx]

    # Store partial distance (For the design interface)
    partial_dist = 0

    while len(P) < num_steps:
        # Add the current point to the path

        P = np.append(P, np.array([pt]), axis=0)

        # Get the averaged principal curvature directions & values
        k1_aver, k2_aver, v1_aver, v2_aver, n_aver = averaged_principal_curvatures(pt, mesh, num_neighbors)

        # Calculate deviation angle (theta) based on principal curvature values
        theta = 2 * np.arctan(
            np.sqrt(
                (2 * np.sqrt(k2_aver * (k2_aver - k1_aver)) + k1_aver - 2 * k2_aver) / k1_aver
                )
        )
            
        # Store theta
        A = np.append(A, np.array([theta]), axis=0)

        # Mirror the angle for tracing backwards. Use trace_backwards indicator
        if trace_backwards:
            theta = np.pi + theta  # todo change it in A as well?

        # Rotate principal curvature direction to get asymptotic direction.
        # Use first_principal_direction indicator
        if not first_principal_direction:
            theta = -theta

        a_dir = rotate_vector(v1_aver, theta, v1_aver, v2_aver, n_aver)

        # Check for anticlastic surface-regions
        if k1_aver * k2_aver > 0:
            break

        # Check for valid asymptotic direction and unitize
        direction_norm = np.linalg.norm(a_dir)

        if direction_norm == 0:
            break

        a_dir /= direction_norm

        # Prevent the tracer to go in the opposite direction
        if len(P) > 1:
            previous_direction = pt - P[-2]
            if a_dir.T @ previous_direction < 0:
                a_dir = rotate_vector(a_dir, np.pi, v1_aver, v2_aver, n_aver)

        # Scale the asymptotic direction to the given step-size
        a_dir *= step_size

        # Compute edge-point
        edge_point, is_boundary_edge = find_edge_point(mesh, pt, a_dir)

        # Check for boundaries
        if is_boundary_edge:
            P = np.append(P, np.array([edge_point]), axis=0)
            break

        # Check for duplicated points
        if np.linalg.norm(P[-1] - edge_point) == 0:
            break

        # Store sampling points (For the design interface)
        if sampling_dist > 0:
            partial_dist += la.norm(edge_point-pt)
            if partial_dist >= sampling_dist:
                partial_dist = 0
                PP = np.append(PP, np.array([edge_point]), axis=0)

        pt = edge_point

    return P, A, PP


def averaged_principal_curvatures(pt, mesh, num_neighbors=2, eps=1e-6):
    """
    Computes inverse weighted distance average of principal curvatures of a given mesh-point
       on the basis of the two closest vertices.
    Try to compute values, directions and normal at the given query point.

    Inputs:
    - pt : np.array (3,)
        The query point position.
    - mesh : Mesh
        The mesh for searching nearest vertices.
    - num_neighbors : int
        Number of closest vertices to consider for avering.
    - eps : float
        The distance tolerance to consider whether the given point and a mesh-vertex are coincident.
            
    Outputs:
    - k_1 : np.array (n)
        The min principal curvature average at the given query point.
    - k_2 : np.array (n)
        The max principal curvature average at the given query point.
    - v1_aver : np.array (3,)
        The unitized min principal curvature direction average at the given query point.
    - v2_aver : np.array (3,)
        The unitized max principal curvature direction average at the given query point.
    - n_aver : np.array (3,)
        The unitized normal average at the given query point.
    """

    # Get the closest vertices and distances to the query point
    # Use these data to compute principal curvature weighted averages.
    dist, neighbors = mesh.get_closest_vertices(pt, num_neighbors)

    # get the distance to the closest neighbor vertex
    min_dist = np.min(dist)
    # get the number of the closest neighbor vertex
    closest_vertex_id = neighbors[np.argmin(dist)]
    # todo ?
    weights = 1 - dist / sum(dist)
    weights /= (num_neighbors - 1)

    if min_dist > eps:
        # this is the case, where none of the vertices are close enough to be considered coincident
        k1_aver = weights @ mesh.K1[neighbors]
        k2_aver = weights @ mesh.K2[neighbors]

        v1_aver = np.einsum('i, ij, i ->j', weights, mesh.V1[neighbors],
                            np.sign(np.dot(mesh.V1[neighbors], mesh.V1[neighbors[0]])))
        v1_aver /= np.linalg.norm(v1_aver)

        v2_aver = np.einsum('i, ij, i->j', weights, mesh.V2[neighbors],
                            np.sign(np.dot(mesh.V2[neighbors], mesh.V2[neighbors[0]])))
        v2_aver /= np.linalg.norm(v2_aver)

        n_aver = weights @ mesh.N[neighbors]
        n_aver /= np.linalg.norm(n_aver)
    else:
        # this is the case where the closest vertex is close enough to be coincident
        k1_aver = mesh.K1[closest_vertex_id]
        k2_aver = mesh.K2[closest_vertex_id]

        v1_aver = mesh.V1[closest_vertex_id]
        v2_aver = mesh.V2[closest_vertex_id]

        n_aver = mesh.N[closest_vertex_id]

    return k1_aver, k2_aver, v1_aver, v2_aver, n_aver


def find_edge_point(mesh, a_orig, a_dir):
    """
    Computes the point where a mesh-edge intersects with the asymptotic projection.
    Try to compute the edge-point resulting from this intersection.

    Inputs:
    - mesh : Mesh
        The mesh for searching edge intersections.
    - a_orig : np.array (3,)
        The start position of the asymptotic projection.
    - a_dic : np.array (3,)
        The direction of the asymptotic projection.
            
    Outputs:
    - edge_point : np.array (3,)
        The position of the edge-point.
    - is_boundary_point : bool
        Indicator for whether the edge-point is at the boundary of the mesh.
    """

    # Get the closest face-index and mesh-point (point laying on the mesh)
    proj_pt = a_orig + a_dir
    face_index, mesh_point = mesh.get_closest_mesh_point(proj_pt)

    # Update the projection vector with the position of the mesh-point
    a_dir = mesh_point - a_orig

    # If the mesh-point is equal to the starting point, return flag for boundary vertex.
    if la.norm(a_dir) == 0:
        return mesh_point, True

    # Unitize projection vector
    a_dir /= la.norm(a_dir)

    # Initialize variables
    edge_point = mesh_point
    is_boundary_point = False
    prev_projection_param = 0
 
    # Find the required edge-point
    # by computing intersections between the edge-segments of the face and the asymptotic-segment.
    # Different intersection events need to be considered. 
    edges = mesh.face_edges[face_index]
    for e_idx in edges:
        e = mesh.edge_vertices[e_idx]
        e_orig = mesh.V[e[0]]
        e_dir = mesh.V[e[1]] - e_orig
        is_boundary_edge = np.any(mesh.edge_faces[e_idx] == -1)

        edge_param, projection_param, intersection = intersection_event(e_orig, e_dir, a_orig, a_dir)

        if (not edge_param) or (not projection_param):
            continue

        if intersection == 0 and prev_projection_param < projection_param and prev_projection_param < projection_param:
            is_boundary_point = is_boundary_edge
            prev_projection_param = projection_param

            edge_point = e_orig + edge_param * e_dir

    return edge_point, is_boundary_point


def intersection_event(a_orig, a_dir, b_orig, b_dir, eps=1e-6):
    """
    Computes the intersection event between segments A and B.
    Try to compute the intersection event.

    Inputs:
    - a_orig : np.array (3,)
        The start position of segment A.
    - a_dic : np.array (3,)
        The direction of the segment A.
    - b_orig : np.array (3,)
        The start position of segment B.
    - b_dic : np.array (3,)
        The direction of the segment B.
    - eps : float
        The tolerance for determining intersections.
            
    Outputs:
    - t : float
        The parameter on segment A where the intersection occurred.
    - u : float
        The parameter on segment B where the intersection occurred.
    - E  : int
        Indicator for the type of intersection event. 
        Returns 0 for all intersection events. 
        Returns 1 for collinearity.
    """
    # todo: ?
    A = np.vstack((a_dir, b_dir)).T
    b = b_orig - a_orig

    lhs = A.T @ A
    rhs = A.T @ b

    if np.linalg.det(lhs) == 0 or np.linalg.norm(np.cross(a_dir,  b_dir)) < eps:
        return None, None, 1

    t, u = np.linalg.solve(lhs, rhs)
    u *= -1

    E = (0 <= u <= 1 and
         0 <= t <= 1)

    return t, u, not E
