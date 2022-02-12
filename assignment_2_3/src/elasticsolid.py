import numpy as np
from numpy import linalg
from scipy import sparse
from Utils import *
import igl


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                             ELASTIC SOLID CLASS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class ElasticSolid(object):

    def __init__(self, v_rest, t, ee, rho=1, pin_idx=[], f_mass=None):
        '''
        Input:
        - v_rest      : position of the vertices of the mesh (#v, 3)
        - t           : indices of the element's vertices (#t, 4)
        - ee          : elastic energy object that can be found in elasticenergy.py
        - rho         : mass per unit volume [kg.m-3]
        - pin_idx     : list of vertex indices to pin
        - f_mass      : external force per unit mass (3,) [N.kg-1]
        '''

        self.v_rest = v_rest.copy()
        self.v_def = v_rest.copy()
        self.t = t
        self.ee = ee
        self.rho = rho
        self.pin_idx = pin_idx
        self.f_mass = f_mass.copy()
        self.free_idx = None
        self.pin_mask = None

        self.W0 = None
        self.Dm = None
        self.Bm = None
        self.rest_barycenters = None

        self.W = None
        self.Ds = None
        self.F = None
        self.def_barycenters = None

        self.f = None
        self.f_vol = None
        self.f_ext = None

        self.make_free_indices_and_pin_mask()
        self.update_rest_shape(self.v_rest)
        self.update_def_shape(self.v_def)

    # Utils #

    def vertex_tet_sum(self, data):
        '''
        Distributes data specified at each tetrahedron to the neighboring vertices.
        All neighboring vertices will receive the value indicated at the corresponding tet position in data.

        //
        The input array should first contain the data for all the tetrahedron and their first vertices,
        then the data for all the tetrahedron and their second vertices, and so on...

        as opposed to the data for the 4 vertices of the first tetr., then the 4 vertices of the second tetrahedron, ...
        //

        Input:
        - data : np array of shape (#t,) or (4*#t,)

        Output:
        - data_sum : np array of shape (#v,), containing the summed data
        '''

        i = self.t.flatten('F')  # (4*#t,)
        j = np.arange(len(self.t))  # (#t,)
        j = np.tile(j, 4)  # (4*#t,)

        if len(data) == len(self.t):
            data = data[j]

        # Has shape (#v, #t)
        m = sparse.coo_matrix((data, (i, j)), (len(self.v_rest), len(self.t)))
        return np.array(m.sum(axis=1)).flatten()

    # Precomputation #

    def make_free_indices_and_pin_mask(self):
        '''
        Should list all the free indices and the pin mask.

        Updated attributes:
        - free_index : np array of shape (#free_vertices,) containing the list of unpinned vertices
        - pin_mask   : np array of shape (#v, 1) containing 1 at free vertex indices and 0 at pinned vertex indices
        '''

        v = self.v_rest.shape[0]
        vi = np.arange(v)
        pin_filter = np.invert(np.in1d(vi, self.pin_idx))
        self.free_idx = vi[pin_filter]

        self.pin_mask = np.array([idx not in self.pin_idx
                                  for idx in range(v)]).reshape(-1, 1)

    # Methods related to rest quantities #

    def make_rest_barycenters(self):
        '''
        Construct the barycenters of the undeformed configuration

        Updated attributes:
        - rest_barycenters : np array of shape (#t, 3) containing the position of each tet's barycenter
        '''
        self.rest_barycenters = np.einsum('ijk -> ik', self.v_rest[self.t]) / 4

    def make_rest_shape_matrices(self):
        '''
        Construct Dm that has shape (#t, 3, 3), and its inverse Bm

        Updated attributes:
        - Dm : np array of shape (#t, 3, 3) containing the shape matrix of each tet
        - Bm : np array of shape (#t, 3, 3) containing the inverse shape matrix of each tet
        '''

        X1_3 = self.v_rest[self.t[:, :3]]
        X4 = self.v_rest[self.t[:, 3]]

        # we transpose X_3, because we need values of each coordinate to be on each row, but for now they are in columns
        self.Dm = np.einsum('ijk -> ikj', X1_3) - X4[:, :, np.newaxis]
        self.Bm = np.linalg.inv(self.Dm)

    def update_rest_shape(self, v_rest):
        '''
        Updates the vertex position, the shape matrices Dm and Bm, the volumes W0,
        and the mass matrix at rest

        Input:
        - v_rest : position of the vertices of the mesh at rest state (#v, 3)

        Updated attributes:
        - v_rest : np array of shape (#v, 3) containing the position of each vertex at rest
        - W0     : np array of shape (#t,) containing the signed volume of each tet
        '''

        self.v_rest = v_rest
        self.make_rest_barycenters()
        self.make_rest_shape_matrices()
        self.W0 = -1 / 6 * np.linalg.det(self.Dm)

        self.update_def_shape(self.v_def)
        self.make_volumetric_and_external_forces()

    # Methods related to deformed quantities ##

    def make_def_barycenters(self):
        '''
        Construct the barycenters of the deformed configuration

        Updated attributes:
        - def_barycenters : np array of shape (#t, 3) containing the position of each tet's barycenter
        '''

        self.def_barycenters = np.einsum('ijk -> ik', self.v_def[self.t]) / 4

    def make_def_shape_matrices(self):
        '''
        Construct Ds that has shape (#t, 3, 3)

        Updated attributes:
        - Ds : np array of shape (#t, 3, 3) containing the shape matrix of each tet
        '''

        x1_3 = self.v_def[self.t[:, :3]]
        x4 = self.v_def[self.t[:, 3]]

        # we transpose X_3, because we need values of each coordinate to be on each row, but for now they are in columns
        self.Ds = np.einsum('ijk -> ikj', x1_3) - x4[:, :, np.newaxis]

    def make_jacobians(self):
        '''
        Compute the current Jacobian of the deformation

        Updated attributes:
        - F : np array of shape (#t, 3, 3) containing Jacobian of the deformation in each tet
        '''
        self.F = np.einsum('ijk, ikn -> ijn', self.Ds, self.Bm)

    def update_def_shape(self, v_def):
        '''
        Updates the vertex position, the Jacobian of the deformation, and the
        resulting elastic forces.

        Input:
        - v_def : position of the vertices of the mesh (#v, 3)

        Updated attributes:
        - v_def : np array of shape (#v, 3) containing the position of each vertex after deforming the solid
        - W     : np array of shape (#t,) containing the signed volume of each tet
        '''
        # this way only update the free vertices
        self.v_def = (1 - self.pin_mask) * self.v_rest + self.pin_mask * v_def
        # self.v_def[self.pin_mask == 1, :] = self.v_rest[self.pin_mask == 1, :]

        self.make_def_barycenters()
        self.make_def_shape_matrices()
        self.make_jacobians()
        self.W = -1 / 6 * np.linalg.det(self.Ds)

        self.make_elastic_energy()
        self.make_elastic_forces()
        self.make_external_energy()

    def displace(self, v_disp):
        '''
        Displace the whole mesh so that v_def += v_disp

        Input:
        - v_disp : displacement of the vertices of the mesh (#v, 3)
        '''

        self.update_def_shape(self.v_def + v_disp)

    # Energies #

    def make_elastic_energy(self):
        '''
        This updates the elastic energy

        Updated attributes:
        - energy_el  : elastic energy of the system [J]
        '''
        self.ee.make_strain_tensor(self.F)
        self.ee.make_energy_density(self.F)
        self.energy_el = np.sum(self.W0 * self.ee.psi)

    def make_external_energy(self):
        '''
        This computes the external energy potential

        Updated attributes:
        - energy_ext : postential energy due to external forces [J]
        '''

        f_vol_tet = np.einsum('i, j -> ij', self.W0 * self.rho, self.f_mass)  # force per tetrahedron
        self.energy_ext = np.sum(f_vol_tet * (self.rest_barycenters - self.def_barycenters))

    # Forces #

    def make_elastic_forces(self):
        '''
        This method updates the elastic forces stored in self.f (#v, 3)

        Updated attributes:
        - f  : elastic forces per vertex (#v, 3)
        - ee : elastic energy, some attributes should be updated
        '''

        self.ee.make_piola_kirchhoff_stress_tensor(self.F)

        PD = np.einsum('hij, hkj -> hik', self.ee.P, self.Bm)

        H = np.zeros((self.W0.shape[0], 3, 4))
        H[:, :, :3] = np.einsum('i, ijk -> ijk', -self.W0, PD)
        H[:, :, 3] = - H[:, :, 0] - H[:, :, 1] - H[:, :, 2]

        self.f = np.array([self.vertex_tet_sum(H[:, i].flatten('F'))
                           for i in range(3)]).T

    def make_volumetric_and_external_forces(self):
        '''
        Convert force per unit mass to volumetric forces, then distribute
        the forces to the vertices of the mesh.

        Updated attributes:
        - f_vol : np array of shape (#t, 3) net external volumetric force acting on the tets
        - f_ext : np array of shape (#v, 3) external force acting on the vertices
        '''

        # Since we are apparently expected to give back force per unit volume, not mass of each tet
        w0 = np.ones_like(self.W0)
        self.f_vol = np.einsum('i, j -> ij', w0 * self.rho, self.f_mass)  # force per unit volume in each tet
        f_vol_tet = np.einsum('i, j -> ij', self.W0 * self.rho, self.f_mass)  # force per tetrahedron

        # force per vertex
        self.f_ext = np.array([self.vertex_tet_sum(f_vol_tet[:, i]) / 4 for i in range(3)]).T

    # Force Differentials

    def compute_force_differentials(self, v_disp):
        '''
        This computes the differential of the force given a displacement dx,
        where df = df/dx|x . dx = - K(x).dx. Where K(x) is the stiffness matrix (or Hessian)
        of the solid. Note that the implementation doesn't need to construct the stiffness matrix explicitly.

        Input:
        - v_disp : displacement of the vertices of the mesh (#v, 3)

        Output:
        - df : force differentials at the vertices of the mesh (#v, 3)

        Updated attributes:
        - ee : elastic energy, some attributes should be updated
        '''

        # First compute the differential of the Jacobian
        x1_3 = v_disp[self.t[:, :3]]
        x4 = v_disp[self.t[:, 3]]
        dDs = np.einsum('ijk -> ikj', x1_3) - x4[:, :, np.newaxis]
        dJ = np.einsum('ijk, ikn -> ijn', dDs, self.Bm)

        # Then update differential quantities in self.ee
        self.ee.make_differential_strain_tensor(jac=self.F, dJac=dJ)
        self.ee.make_differential_piola_kirchhoff_stress_tensor(jac=self.F, dJac=dJ)

        # Compute the differential of the forces
        PD = np.einsum('hij, hkj -> hik', self.ee.dP, self.Bm)

        H = np.zeros((self.W0.shape[0], 3, 4))
        H[:, :, :3] = np.einsum('i, ijk -> ijk', -self.W0, PD)
        H[:, :, 3] = - H[:, :, 0] - H[:, :, 1] - H[:, :, 2]

        return np.array([self.vertex_tet_sum(H[:, i].flatten('F'))
                         for i in range(3)]).T

    def equilibrium_step(self, step_size_init=2, max_l_iter=20, c1=1e-4):
        '''
        This function displaces the whole solid to the next deformed configuration
        using a Newton-CG step.

        Updated attributes:
        - LHS : The hessian vector product
        - RHS : Right hand side for the conjugate gradient linear solve
        Other than them, only attributes updated by displace(self, v_disp) should be changed
        '''

        dx0s = np.zeros_like(self.v_rest)

        # Define LHS
        def LHS(dx):
            '''
            Should implement the Hessian-Vector Product L(dx), and take care of pinning constraints
            as described in the handout.
            '''
            dx0s[self.free_idx] = dx.reshape(-1, 3)
            df0s = - self.compute_force_differentials(dx0s)
            return df0s[self.free_idx, :].reshape(-1, )

        self.LHS = LHS  # Save to class for testing

        # Define RHS
        ft = self.f + self.f_ext
        RHS = ft[self.free_idx, :].reshape(-1, )
        self.RHS = RHS  # Save to class for testing

        dx = conjugate_gradient(LHS, RHS)
        dx0s[self.free_idx] = dx.reshape(-1, 3)

        # Run line search on the direction
        step_size = 2
        ft_free = RHS
        g_old = np.linalg.norm(ft_free)
        for l_iter in range(max_l_iter):
            step_size *= 0.5
            dx_search = dx0s * step_size
            energy_tot_prev = self.energy_el + self.energy_ext
            self.displace(dx_search)
            ft_new = (self.f_ext + self.f)[self.free_idx].reshape(-1, )
            g = np.linalg.norm(ft_new)

            energy_tot_tmp = self.energy_el + self.energy_ext
            print(self.energy_el, self.energy_ext)
            armijo = energy_tot_tmp < energy_tot_prev - c1 * step_size * np.sum(dx.reshape(-1, ) * ft_free)

            if armijo or l_iter == max_l_iter - 1:
                print("Energy: " + str(energy_tot_tmp) + " Force residual norm: " + str(g) + " Line search Iters: " + str(l_iter))
                break
            else:
                self.displace(-dx_search)