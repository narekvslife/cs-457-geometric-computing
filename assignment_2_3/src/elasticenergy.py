import numpy as np
from numpy.core.einsumfunc import einsum
from numpy.core.fromnumeric import swapaxes


class ElasticEnergy:
    def __init__(self, young, poisson):
        '''
        Input:
        - young   : Young's modulus [Pa]
        - poisson : Poisson ratio
        '''
        self.young   = young
        self.poisson = poisson
        self.lbda = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
        self.mu   = young / (2 * (1 + poisson))

        self.psi = None
        self.E   = None
        self.P   = None

        self.dE = None
        self.dP = None

    def make_energy_density(self, jac):
        '''
        This method computes the energy density at each tetrahedron (#t,),
        and stores the result in self.psi

        Input:
        - jac  : jacobian of the deformation (#t, 3, 3)
        
        Updated attributes:
        - psi : energy density per tet (#t,)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError
    
    def make_strain_tensor(self, jac):
        '''
        This method computes the strain tensor (#t, 3, 3), and stores it in self.E

        Input:
        - jac : jacobian of the deformation (#t, 3, 3)
        
        Updated attributes:
        - E : strain induced by the deformation (#t, 3, 3)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError

    def make_piola_kirchhoff_stress_tensor(self, jac):
        '''
        This method computes the stress tensor (#t, 3, 3), and stores it in self.P

        Input:
        - jac : jacobian of the deformation (#t, 3, 3)
        
        Updated attributes:
        - P : stress tensor induced by the deformation (#t, 3, 3)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError

    def make_differential_strain_tensor(self, jac, dJac):
        '''
        This method computes the differential of strain tensor (#t, 3, 3), 
        and stores it in self.dE

        Input:
        - jac  : jacobian of the deformation (#t, 3, 3)
        - dJac : differential of the jacobian of the deformation (#t, 3, 3)
        
        Updated attributes:
        - dE : differential of the strain tensor (#t, 3, 3)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError

    def make_differential_piola_kirchhoff_stress_tensor(self, jac, dJac):
        '''
        This method computes the differential of the stress tensor (#t, 3, 3), and stores it in self.dP

        Input:
        - jac  : jacobian of the deformation (#t, 3, 3)
        - dJac : differential of the jacobian of the deformation (#t, 3, 3)
        
        Updated attributes:
        - dP : differential of the stress tensor (#t, 3, 3)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError


class LinearElasticEnergy(ElasticEnergy):
    def __init__(self, young, poisson):
        super().__init__(young, poisson)

    def make_energy_density(self, jac):
        # self.make_strain_tensor(jac)  # I would prefer this to be inside the function, actually
        self.psi = self.mu * np.einsum('ijk, ikj -> i', self.E, self.E) + 0.5 * self.lbda * np.einsum('ijj -> i', self.E) ** 2

    def make_strain_tensor(self, jac):
        I = np.identity(3)
        self.E = 0.5 * (jac + np.einsum('ijk -> ikj', jac)) - I

    def make_piola_kirchhoff_stress_tensor(self, jac):
        # self.make_strain_tensor(jac)  # I would prefer this to be inside the function, actually

        strain_trace = np.einsum('ijj', self.E)

        I = np.identity(3)
        self.P = 2 * self.mu * self.E + self.lbda * np.einsum('i, jk -> ijk', strain_trace, I)

    def make_differential_strain_tensor(self, jac, dJac):
        self.dE = 0.5 * (dJac + np.einsum('ijk -> ikj', dJac))

    def make_differential_piola_kirchhoff_stress_tensor(self, jac, dJac):
        # self.make_strain_tensor(jac)  # I would prefer this to be inside the function, actually

        strain_trace = np.einsum('ijj', self.dE)

        I = np.identity(3)
        self.dP = 2 * self.mu * self.dE + self.lbda * np.einsum('i, jk -> ijk', strain_trace, I)


class NeoHookeanElasticEnergy(ElasticEnergy):
    def __init__(self, young, poisson):
        super().__init__(young, poisson)
        self.logJ = None
        self.Finv = None

    def make_energy_density(self, jac):
        FTF = np.einsum('ikj, ikn -> ijn', jac, jac)
        I1 = np.einsum('ijj', FTF)

        J = np.linalg.det(jac)
        self.psi = 0.5 * self.mu * (I1 - 3) - self.mu * np.log(J) + 0.5 * self.lbda * np.log(J) ** 2

    def make_strain_tensor(self, jac):
        pass

    def make_piola_kirchhoff_stress_tensor(self, jac):
        '''
        Additional updated attributes:
        - logJ ; log of the determinant of the jacobians (#t,)
        - Finv : inverse of the jacobians (#t, 3, 3)
        '''
        J = np.linalg.det(jac)
        self.logJ = np.log(J)
        self.Finv = np.linalg.inv(jac)

        FinvT = np.einsum('ijk -> ikj', self.Finv)

        self.P = self.mu * (jac - FinvT) + self.lbda * np.einsum('i, ijk -> ijk', self.logJ, FinvT)

    def make_differential_strain_tensor(self, jac, dJac):
        pass

    def make_differential_piola_kirchhoff_stress_tensor(self, jac, dJac):
        J = np.linalg.det(jac)
        self.logJ = np.log(J)
        self.Finv = np.linalg.inv(jac)

        FinvT = np.einsum('ijk -> ikj', self.Finv)

        FinvT_dFT = np.einsum('ijk, ink -> ijn', FinvT, dJac)
        FinvT_dFT_FinvT = np.einsum('ijk, ikn -> ijn', FinvT_dFT, FinvT)

        trace = np.einsum('ijj', FinvT_dFT)
        final = np.einsum('i, ijk -> ijk', trace, FinvT)

        self.dP = self.mu * dJac
        self.dP += np.einsum('i, ijk -> ijk', (self.mu - self.lbda * self.logJ), FinvT_dFT_FinvT)
        self.dP += self.lbda * final
