import numpy as np


# -----------------------------------------------------------------------------
#  DERIVATIVES OF PARABOLOID
# -----------------------------------------------------------------------------

def compute_paraboloid_points(P, a, b, c, d, e):
    """Computes the points of the paraboloid x(u,v) = (u, v, z(u,v)) with
    z(u,v) = a*u^2 + b*v^2 + c*u*v + d*u + e*v.
    Try to vectorize this function.

    Parameters:
    - P : np.array (n, 2)
        Contains in i-th row the (u, v) coordinates of the i-th parameter point p_i.
    - a, b, c, d, e : float
        The parameters of the paraboloid.

    Returns:
    - x : np.array (n, 3)
        The points x(P), where the i-th row contains the (x,y,z) coordinates
        of the point x(p_i)
    """

    x = u = P[:, 0]
    y = v = P[:, 1]
    z = a * u*u + b * v*v + c*u*v + d * u + e * v
    return np.vstack((x, y, z)).T


def compute_paraboloid_first_derivatives(P, a, b, c, d, e):
    """Computes the first derivatives of the paraboloid x(u,v) = (u, v, z(u,v))
    with z(u,v) = a*u^2 + b*v^2 + c*u*v + d*u + e*v.
    Try to vectorize this function.

    Parameters:
    - P : np.array (n, 2)
        Contains in i-th row the (u, v) coordinates of the i-th parameter point p_i.
    - a, b, c, d, e : float
        The parameters of the paraboloid.

    Returns:
    - x_u : np.array (n, 3)
        The vectors x_u(P), where the i-th row contains the (x,y,z) coordinates
        of the vector x_u(p_i).
    - x_v : np.array (n, 3)
        The vectors x_v(P), where the i-th row contains the (x,y,z) coordinates
        of the vector x_v(p_i).
    """
    u = P[:, 0]
    v = P[:, 1]

    x_xu = np.ones_like(P[:, 0])
    x_yu = np.zeros_like(P[:, 0])
    x_zu = 2 * a * u + c * v + d

    x_u = np.vstack((x_xu, x_yu, x_zu)).T

    x_xv = np.zeros_like(P[:, 0])
    x_yv = np.ones_like(P[:, 0])
    x_zv = 2 * b * v + c * u + e

    x_v = np.vstack((x_xv, x_yv, x_zv)).T

    return x_u, x_v


def compute_paraboloid_second_derivatives(P, a, b, c, d, e):
    """Computes the second derivatives of the paraboloid x(u,v) = (u, v, z(u,v))
    with z(u,v) = a*u^2 + b*v^2 + c*u*v + d*u + e*v.
    Try to vectorize this function.

    Parameters:
    - P : np.array (n, 2)
        Contains in i-th row the (u, v) coordinates of the i-th parameter point p_i.
    - a, b, c, d, e : float
        The parameters of the paraboloid.

    Returns:
    - x_uu : np.array (n, 3)
        The vectors x_uu(P), where the i-th row contains the (x,y,z) coordinates
        of the vector x_uu(p_i).
    - x_uv : np.array (n, 3)
        The vectors x_uv(P), where the i-th row contains the (x,y,z) coordinates
        of the vector x_uv(p_i).
    - x_vv : np.array (n, 3)
        The vectors x_vv(P), where the i-th row contains the (x,y,z) coordinates
        of the vector x_vv(p_i).
    """
    x_x_uu_uv_vv = np.zeros_like(P[:, 0])
    x_y_uu_uv_vv = np.zeros_like(P[:, 0])

    x_zuu = 2 * a * np.ones_like(P[:, 0])
    x_zuv = c * np.ones_like(P[:, 0])
    x_zvv = 2 * b * np.ones_like(P[:, 0])

    x_uu = np.vstack((x_x_uu_uv_vv, x_y_uu_uv_vv, x_zuu)).T
    x_uv = np.vstack((x_x_uu_uv_vv, x_y_uu_uv_vv, x_zuv)).T
    x_vv = np.vstack((x_x_uu_uv_vv, x_y_uu_uv_vv, x_zvv)).T
    return x_uu, x_uv, x_vv


# -----------------------------------------------------------------------------
#  DERIVATIVES OF TORUS
# -----------------------------------------------------------------------------

def compute_torus_points(P, R, r):
    """Computes the second derivatives of a torus.
    Try to vectorize this function.

    Parameters:
    - P : np.array (n, 2)
        Contains in i-th row the (u, v) coordinates of the i-th parameter point p_i.
    - R : float
        The radius of revolution.
    - r : float
        The radius of the cross section.
    Returns:
    - x : np.array (n, 3)
        The points x(P), where the i-th row contains the (x,y,z) coordinates
        of the point x(p_i)
    """
    u = P[:, 0]
    v = P[:, 1]

    rcosu_R = r * np.cos(u) + R

    x_x = rcosu_R * np.cos(v)
    x_y = rcosu_R * np.sin(v)
    x_z = r * np.sin(u)

    x = np.vstack((x_x, x_y, x_z)).T
    return x


def compute_torus_first_derivatives(P, R, r):
    """Computes the second derivatives of a torus.
    Try to vectorize this function.

    Parameters:
    - P : np.array (n, 2)
        Contains in i-th row the (u, v) coordinates of the i-th parameter point p_i.
    - R : float
        The radius of revolution.
    - r : float
        The radius of the cross section.
    Returns:
    - x_u : np.array (n, 3)
        The vectors x_u(P), where the i-th row contains the (x,y,z) coordinates
        of the vector x_u(p_i).
    - x_v : np.array (n, 3)
        The vectors x_v(P), where the i-th row contains the (x,y,z) coordinates
        of the vector x_v(p_i).
    """
    u = P[:, 0]
    v = P[:, 1]

    sinu = np.sin(u)
    sinv = np.sin(v)

    cosu = np.cos(u)
    cosv = np.cos(v)

    x_xu = -r * sinu * cosv
    x_yu = -r * sinu * sinv
    x_zu = r * np.cos(u)

    x_u = np.vstack((x_xu, x_yu, x_zu)).T

    x_xv = - r * cosu * sinv - R * sinv
    x_yv = r * cosu * cosv + R * cosv
    x_zv = np.zeros_like(v)

    x_v = np.vstack((x_xv, x_yv, x_zv)).T

    return x_u, x_v


def compute_torus_second_derivatives(P, R, r):
    """Computes the second derivatives of a torus.
    Try to vectorize this function.

    Parameters:
    - P : np.array (n, 2)
        Contains in i-th row the (u, v) coordinates of the i-th parameter point p_i.
    - R : float
        The radius of revolution.
    - r : float
        The radius of the cross section.

    Returns:
    - x_uu : np.array (n, 3)
        The vectors x_uu(P), where the i-th row contains the (x,y,z) coordinates
        of the vector x_uu(p_i).
    - x_uv : np.array (n, 3)
        The vectors x_uv(P), where the i-th row contains the (x,y,z) coordinates
        of the vector x_uv(p_i).
    - x_vv : np.array (n, 3)
        The vectors x_vv(P), where the i-th row contains the (x,y,z) coordinates
        of the vector x_vv(p_i).
    """
    u = P[:, 0]
    v = P[:, 1]

    sinu = np.sin(u)
    sinv = np.sin(v)

    cosu = np.cos(u)
    cosv = np.cos(v)

    # x_uu
    x_x_uu = - r * cosu * cosv
    x_y_uu = - r * cosu * sinv
    x_z_uu = - r * sinu
    x_uu = np.vstack((x_x_uu, x_y_uu, x_z_uu)).T

    # x_uv
    x_x_uv = r * sinu * sinv
    x_y_uv = - r * sinu * cosv
    x_z_uv = np.zeros_like(v)
    x_uv = np.vstack((x_x_uv, x_y_uv, x_z_uv)).T

    # x_vv
    x_x_vv = - r * cosu * cosv - R * cosv
    x_y_vv = - r * cosu * sinv - R * sinv
    x_z_vv = np.zeros_like(v)
    x_vv = np.vstack((x_x_vv, x_y_vv, x_z_vv)).T

    return x_uu, x_uv, x_vv


# -----------------------------------------------------------------------------
#  SHAPE OPERATOR
# -----------------------------------------------------------------------------

def compute_first_fundamental_form(x_u, x_v):
    """Computes the first fundamental form I.
    Try to vectorize this function.

    Parameters:
    - x_u : np.array (n, 3)
        The i-th row contains the (x,y,z) coordinates of the vector x_u(p_i).
    - x_v : np.array (n, 3)
        The i-th row contains the (x,y,z) coordinates of the vector x_v(p_i).

    Returns:
    - I : np.array (n, 2, 2)
        The first fundamental forms.
        The (i, j, k) position contains the (j, k) element of the first
        fundamental form I(p_i).
    """

    first = np.expand_dims(np.einsum('ik, ik -> i', x_u, x_u), 1)
    symmetric = np.expand_dims(np.einsum('ik, ik -> i', x_v, x_u), 1)
    second = np.expand_dims(np.einsum('ik, ik -> i', x_v, x_v), 1)

    first_row = np.hstack((first, symmetric))
    second_row = np.hstack((symmetric, second))
    final = np.stack((first_row, second_row), axis=1)

    return final


def compute_surface_normal(x_u, x_v):
    """Computes the surface normal n.
    Try to vectorize this function.

    Parameters:
    - x_u : np.array (n, 3)
        The i-th row contains the (x,y,z) coordinates of the vector x_u(p_i).
    - x_v : np.array (n, 3)
        The i-th row contains the (x,y,z) coordinates of the vector x_v(p_i).

    Returns:
    - n : np.array (n, 3)
        The surface normals.
        The i-th row contains the (x,y,z) coordinates of the vector n(p_i).
    """
    a1 = x_u[:, 0]
    a2 = x_u[:, 1]
    a3 = x_u[:, 2]

    b1 = x_v[:, 0]
    b2 = x_v[:, 1]
    b3 = x_v[:, 2]

    s1 = a2 * b3 - a3 * b2
    s2 = a3 * b1 - a1 * b3
    s3 = a1 * b2 - a2 * b1

    cross_product = np.zeros_like(x_u)
    cross_product[:, 0] = s1
    cross_product[:, 1] = s2
    cross_product[:, 2] = s3

    cross_norm = np.linalg.norm(cross_product, axis=1)

    normal = cross_product / np.expand_dims(cross_norm, 1)
    return normal


def compute_second_fundamental_form(x_uu, x_uv, x_vv, n):
    """Computes the second fundamental form II.
    Try to vectorize this function.

    Parameters:
    - x_uu : np.array (n, 3)
         The i-th row contains the (x,y,z) coordinates of the vector x_uu(p_i).
    - x_uv : np.array (n, 3)
         The i-th row contains the (x,y,z) coordinates of the vector x_uv(p_i).
    - x_vv : np.array (n, 3)
         The i-th row contains the (x,y,z) coordinates of the vector x_vv(p_i).
    - n : np.array (n, 3)
        The surface normals.
        The i-th row contains the (x,y,z) coordinates of the vector n(p_i).

    Returns:
    - II : np.array (n, 2, 2)
        The second fundamental forms.
        The (i, j, k) position contains the (j, k) element of the second
        fundamental form II(p_i).
    """
    first = np.expand_dims(np.einsum('ik, ik -> i', n, x_uu), 1)
    symmetric = np.expand_dims(np.einsum('ik, ik -> i', n, x_uv), 1)
    second = np.expand_dims(np.einsum('ik, ik -> i', n, x_vv), 1)

    first_row = np.hstack((first, symmetric))
    second_row = np.hstack((symmetric, second))
    II = np.stack((first_row, second_row), axis=1)

    return II


def compute_shape_operator(I, II):
    """Computes the shape operator S.
    Try to vectorize this function.

    Parameters:
    - I : np.array (n, 2, 2)
        The first fundamental forms.
        The (i, j, k) position contains the (j, k) element of the first
        fundamental form I(p_i).
    - II : np.array (n, 2, 2)
        The second fundamental forms.
        The (i, j, k) position contains the (j, k) element of the second
        fundamental form II(p_i).

    Returns:
    - S : np.array (n, 2, 2)
        The shape operators.
        The (i, j, k) position contains the (j, k) element of the shape
        operator S(p_i).
    """
    S = np.einsum('ijk, ijl -> ikl', np.linalg.inv(I), II)
    return S


# -----------------------------------------------------------------------------
#  PRINCIPAL CURVATURES
# -----------------------------------------------------------------------------

def compute_principal_curvatures(S, x_u, x_v):
    """Computes principal curvatures and corresponding principal directions.
    Try to vectorize this function.

    Parameters:
    - S : np.array (n, 2, 2)
        The shape operators.
        The (i, j, k) position contains the (j, k) element of the shape
        operator S(p_i).
    - x_u : np.array (n, 3)
        The i-th row contains the (x,y,z) coordinates of the vector x_u(p_i).
    - x_v : np.array (n, 3)
        The i-th row contains the (x,y,z) coordinates of the vector x_v(p_i).

    Returns:
    - k_1 : np.array (n)
        The min principal curvature. i-th element contains the curvature k_1(p_i).
    - k_2 : np.array (n)
        The max principal curvature. i-th element contains the curvature k_2(p_i).
    - e_1 : np.array (n, 3)
        The unitized principal curvature direction corresponding to k_1.
        The i-th row contains the (x,y,z) coordinates of e_1(p_i).
    - e_2 : np.array (n, 3)
        The unitized principal curvature direction corresponding to k_2.
        The i-th row contains the (x,y,z) coordinates of e_2(p_i).
    """
    # this section computes the ordered eigenvalues and eigenvectors of S where
    # k_1[i] = min eigenvalue at p_i, k_2[i] = max eigenvalue at p_i,
    # bar_e_1[i] = [u, v] components of the eigenvector of k_1,
    # bar_e_2[i] = [u, v] components of the eigenvector of k_2
    eig = np.linalg.eig(S)
    index = np.argsort(eig[0], axis=1)
    k_1 = eig[0][np.arange(len(S)), index[:, 0]]
    k_2 = eig[0][np.arange(len(S)), index[:, 1]]
    bar_e_1 = eig[1][np.arange(len(S)), :, index[:, 0]]
    bar_e_2 = eig[1][np.arange(len(S)), :, index[:, 1]]

    # TODO: compute the normalized 3D vectors e_1, e_2
    J = np.stack((x_u, x_v), 2)
    e1 = np.einsum('ijk, ik -> ij', J, bar_e_1)
    e2 = np.einsum('ijk, ik -> ij', J, bar_e_2)

    e_1_norm = e1 / np.expand_dims(np.linalg.norm(e1, axis=1), 1)
    e_2_norm = e2 / np.expand_dims(np.linalg.norm(e2, axis=1), 1)
    return k_1, k_2, e_1_norm, e_2_norm


# -----------------------------------------------------------------------------
#  ASYMPTOTIC DIRECTIONS
# -----------------------------------------------------------------------------

def compute_asymptotic_directions(k_1, k_2, e_1, e_2):
    """Computes principal curvatures and corresponding principal directions.
    Try to vectorize this function.

    Parameters:
    - k_1 : np.array (n)
        The min principal curvature. i-th element contains the curvature k_1(p_i).
    - k_2 : np.array (n)
        The max principal curvature. i-th element contains the curvature k_2(p_i).
    - e_1 : np.array (n, 3)
        The unitized principal curvature direction corresponding to k_1.
        The i-th row contains the (x,y,z) coordinates of e_1(p_i).
    - e_2 : np.array (n, 3)
        The unitized principal curvature direction corresponding to k_2.
        The i-th row contains the (x,y,z) coordinates of e_2(p_i).

    Returns:
    - a_1 : np.array (n, 3)
        The first unitized asymptotic direction. The i-th row contains the
        (x,y,z) coordinates of a_2(p_i) if it exists, (0, 0, 0) otherwise.
    - a_2 : np.array (n, 3)
        The second unitized asymptotic direction. The i-th row contains the
        (x,y,z) coordinates of a_2(p_i) if it exists, (0, 0, 0) otherwise.
    """

    K = k_1 * k_2

    exist_both_directions = np.expand_dims(K < 0, 1)
    exist_exactly_one_direction = np.expand_dims(K != 0, 1)

    cos_theta = np.expand_dims(np.sqrt(1 - k_1 / (k_1 - k_2)), 1)
    sin_theta = np.expand_dims(np.sqrt(k_1 / (k_1 - k_2)), 1)

    a_1 = cos_theta * e_1 + sin_theta * e_2
    a_2 = cos_theta * e_1 - sin_theta * e_2

    a_1 = a_1 * exist_both_directions
    a_2 = a_2 * exist_both_directions * exist_exactly_one_direction

    return np.nan_to_num(a_1), np.nan_to_num(a_2)
