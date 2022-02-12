import numpy as np
from numpy import linalg
from scipy import sparse
import igl


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                             ELASTIC SOLID CLASS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class ElasticSolid(object):

    def __init__(self, v_rest, t, rho=1, pin_idx=[]):
        '''
        Input:
        - v_rest      : position of the vertices of the mesh (#v, 3)
        - t           : indices of the element's vertices (#t, 4)
        - rho         : mass per unit volume [kg.m-3]
        - pin_idx     : list of vertex indices to pin
        '''

        self.v_rest   = v_rest.copy()
        self.v_def    = v_rest.copy()
        self.t        = t
        self.rho      = rho
        self.pin_idx  = pin_idx
        self.free_idx = None
        self.pin_mask = None
        
        self.W0 = None
        self.Dm = None
        self.Bm = None
        self.rest_barycenters = None

        self.W  = None
        self.Ds = None
        self.F  = None
        self.def_barycenters = None

        self.make_free_indices_and_pin_mask()
        self.update_rest_shape(self.v_rest)
        self.update_def_shape(self.v_def)

    ## Precomputation ##

    def make_free_indices_and_pin_mask(self):
        '''
        Should list all the free indices and the pin mask.

        Updated attributes:
        - free_index : np array of shape (#free_vertices,) containing the list of unpinned vertices
        - pin_mask   : np array of shape (#v, 1) containing 1 at free vertex indices and 0 at pinned vertex indices
        '''

        N_v = self.v_rest.shape[0]

        self.free_idx = np.delete(np.arange(N_v), self.pin_idx)

        self.pin_mask = np.zeros((N_v, 1))
        self.pin_mask[self.free_idx] = 1

    ## Methods related to rest quantities ##

    def make_rest_barycenters(self):
        '''
        Construct the barycenters of the undeformed configuration

        Updated attributes:
        - rest_barycenters : np array of shape (#t, 3) containing the position of each tet's barycenter
        '''

        w = 1/4 * np.ones(len(self.t))
        self.rest_barycenters = np.einsum('ijk, i -> ik', self.v_rest[self.t], w)

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
        - W0     : np array of shape (#t, 3) containing the volume of each tet
        '''

        self.v_rest = v_rest
        self.make_rest_barycenters()
        self.make_rest_shape_matrices()
        self.W0 = -1/6 * np.linalg.det(self.Dm)

        self.make_def_shape_matrices()
        self.make_jacobians()

    # Methods related to deformed quantities

    def make_def_barycenters(self):
        '''
        Construct the barycenters of the deformed configuration

        Updated attributes:
        - def_barycenters : np array of shape (#t, 3) containing the position of each tet's barycenter
        '''
        w = 1/4 * np.ones(len(self.t))
        self.def_barycenters = np.einsum('ijk, i -> ik', self.v_def[self.t], w)

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
        - W     : np array of shape (#t, 3) containing the volume of each tet
        '''
        # this way only update the free vertices
        self.v_def[self.free_idx] = v_def[self.free_idx]
        # self.v_def[self.pin_mask == 1, :] = self.v_rest[self.pin_mask == 1, :]

        self.make_def_barycenters()
        self.make_def_shape_matrices()
        self.make_jacobians()
        self.W = -1/6 * np.linalg.det(self.Ds)

    def displace(self, v_disp):
        '''
        Displace the whole mesh so that v_def += v_disp

        Input:
        - v_disp : displacement of the vertices of the mesh (#v, 3)
        '''
        self.update_def_shape(self.v_def + v_disp)
