import scipy.sparse

import laplacian_utils
import scipy as sp
from scipy.sparse.linalg import spsolve
import igl

from utils import normalize_area, has_zero_area_triangle
import copy 
import numpy as np
import numpy.linalg as la


class MCF():
    def __init__(self, num_bdry_vx, num_intr_vx):
        '''
        Inputs:
        - num_bdry_vx : int
            The first num_bdry_vx vertices in v are boundary vertices in the mesh.
        - num_intr_vx : int
            The number of interior vertices in the mesh.
        '''
        self.num_bdry_vx = num_bdry_vx
        self.num_intr_vx = num_intr_vx

        self.L = None  # Laplacian matrix.
        self.M = None  # Mass matrix.
        self.average_mean_curvature = 0  # The average mean curvature value of the mesh.

    def update_system(self, v, f):
        '''
        Update the member variables in the class, including the mass matrix,
        the Laplacian matrix, and the average mean curvature value of the mesh.
        '''
        self.M = laplacian_utils.compute_mass_matrix(v, f)
        self.L = laplacian_utils.compute_laplacian_matrix(v, f)

        M_inv= sp.sparse.diags(1 / self.M.diagonal())
        hn = - 0.5 * M_inv @ self.L @ v

        H = np.linalg.norm(hn, axis=1)
        H[:self.num_bdry_vx] *= 0
        self.average_mean_curvature = np.sum(H) / self.num_intr_vx / igl.bounding_box_diagonal(v)

    def solve_laplace_equation(self, v, f):
        '''
        Solve the Laplace equation for the current mesh. Update the vertex positions with the solution.
        '''

        L = laplacian_utils.compute_laplacian_matrix(v, f)

        # taking xyz coordinates of boundary vertices
        boundary_vertices_x = v[:self.num_bdry_vx, 0]
        boundary_vertices_y = v[:self.num_bdry_vx, 1]
        boundary_vertices_z = v[:self.num_bdry_vx, 2]

        lhs = L[self.num_bdry_vx:, self.num_bdry_vx:]
        rhs = -L[self.num_bdry_vx:, :self.num_bdry_vx]

        v[self.num_bdry_vx:, 0] = spsolve(lhs, rhs @ boundary_vertices_x)
        v[self.num_bdry_vx:, 1] = spsolve(lhs, rhs @ boundary_vertices_y)
        v[self.num_bdry_vx:, 2] = spsolve(lhs, rhs @ boundary_vertices_z)

    def meet_stopping_criteria(self, mean_curvature_list, epsilon1, epsilon2):
        '''
        Stopping criteria for mean curvature flow.
        '''
        if len(mean_curvature_list) < 2:
            return False
        # If the changes in the iteration is smaller than epsilon1, terminate the flow. 
        if la.norm(mean_curvature_list[-1] - mean_curvature_list[-2]) < epsilon1:
            print("Insufficient improvement from the previous iteration!")
            return True
        # If the average mean curvature value of the mesh is sufficiently small, terminate the flow.
        if np.abs(mean_curvature_list[-1]) < epsilon2:
            print("Sufficiently small average mean curvature value!")
            return True
        return False

    def run_mean_curvature_flow(self, v, f, max_iter, epsilon1, epsilon2):
        '''
        Running mean curvature flow by iteratively solving the Laplace equation.
        '''

        self.update_system(v, f)

        vs = [copy.deepcopy(v)]
        average_mean_curvature_list = []
        i = 0
        self.update_system(v, f)
        average_mean_curvature_list.append(self.average_mean_curvature)
        while (i < max_iter and not has_zero_area_triangle(v, f) and
               not self.meet_stopping_criteria(average_mean_curvature_list, epsilon1, epsilon2)):

            self.solve_laplace_equation(v, f)
            self.update_system(v, f)

            if self.num_bdry_vx == 0:
                normalize_area(v, f)
            
            vs.append(copy.deepcopy(v))
            average_mean_curvature_list.append(self.average_mean_curvature)
            i += 1

        return vs, average_mean_curvature_list
