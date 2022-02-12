import torch
from utils import linear_solve

torch.set_default_dtype(torch.float64)


def compute_adjoint(solid ,dJdx_U):
    '''
    This assumes that S is at equilibrium when called
    
    Input:
    - solid : an elastic solid at equilibrium
    - dJdx_U  : array of shape (#unpinned, 3)
    
    Output:
    - adjoint : array of shape (3*#v,)
    '''

    dx0s = torch.zeros_like(solid.v_rest)
    def LHS(dx):
        '''
        Should implement the Hessian-Vector product (taking into account pinning constraints) as described in the handout.
        '''
        dx0s[solid.free_idx] = dx.reshape(-1, 3)
        dx = -solid.compute_force_differentials(dx0s)
        return dx[solid.free_idx].reshape(-1, )

    RHS = dJdx_U.flatten()

    # here we only pass unpinned vertices
    y_hat = dx0s
    y_hat[solid.free_idx] = linear_solve(LHS, RHS).reshape(-1, 3)

    return y_hat.flatten()
