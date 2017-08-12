import numpy as np
import math

class potential():
    def __init__(self):
        self.forcefield = ""
        self.numdims = 0
        self.energy = 0.0
        self.forces = np.zeros(self.get_numdims())

    def eval_forcefield(self,r):
        exec("self.eval_forcefield_" + self.forcefield + "(r)")

    def set_forcefield(self,ff):
        self.forcefield = ff

    def get_forcefield(self,ff):
        return ff

    def set_numdims(self,ndims):
        self.numdims = ndims
        self.forces = np.zeros(ndims)
    
    def get_numdims(self):
        return self.numdims

    def set_energy(self,en):
        self.energy = en

    def get_energy(self):
        return self.energy

    def set_forces(self,f):
        self.forces = f.copy()

    def get_forces(self):
        return self.forces.copy()

###########################################
# chain forcefield
###########################################

    def set_chain_length(self,n):
        self.chain_length = n
        self.set_numdims(3*n)

    def get_chain_length(self):
        return self.chain_length

    def set_chain_bond_r(self,r):
        self.chain_bond_r = r

    def get_chain_bond_r(self):
        return self.chain_bond_r

    def set_chain_bond_k(self,k):
        self.chain_bond_k =  k

    def get_chain_bond_k(self):
        return self.chain_bond_k

    def set_chain_lj_eps(self,eps):
        self.chain_lj_eps = eps

    def get_chain_lj_eps(self):
        return self.chain_lj_eps
    
    def set_chain_lj_sigma(self,sigma):
        self.chain_lj_sigma = sigma

    def get_chain_lj_sigma(self):
        return self.chain_lj_sigma
    
    def init_chain(self,n,r,k,sigma,eps):
        self.set_forcefield("chain")
        self.set_chain_length(n)
        self.set_chain_bond_r(r)
        self.set_chain_bond_k(k)
        self.set_chain_lj_sigma(sigma)
        self.set_chain_lj_eps(eps)

    def eval_forcefield_chain(self,r):
        e = 0.0
        f = np.zeros(self.get_numdims())

        # compute bonding terms
        for iatom in range(self.get_chain_length()-1):
            ix = 3*iatom
            iy = 3*iatom + 1
            iz = 3*iatom + 2
            jx = 3*iatom + 3
            jy = 3*iatom + 4
            jz = 3*iatom + 5
            
            x_ab = r[ix] - r[jx]
            y_ab = r[iy] - r[jy]
            z_ab = r[iz] - r[jz]
            
            r_ab = math.sqrt(x_ab*x_ab + y_ab*y_ab + z_ab*z_ab)
            disp_ab = r_ab - self.get_chain_bond_r()
            
            e += self.get_chain_bond_k() * disp_ab * disp_ab
            
            ftmp = 2.0 * self.get_chain_bond_k() * disp_ab
            
            x_ab_norm = x_ab / r_ab
            y_ab_norm = y_ab / r_ab
            z_ab_norm = z_ab / r_ab
            
            f[ix] -= x_ab_norm * ftmp
            f[iy] -= y_ab_norm * ftmp
            f[iz] -= z_ab_norm * ftmp
            f[jx] += x_ab_norm * ftmp
            f[jy] += y_ab_norm * ftmp
            f[jz] += z_ab_norm * ftmp

            # compute lj terms
            for jatom in range(iatom + 2, self.get_chain_length()):
                jx = 3*jatom 
                jy = 3*jatom + 1
                jz = 3*jatom + 2

                x_ab = r[ix] - r[jx]
                y_ab = r[iy] - r[jy]
                z_ab = r[iz] - r[jz]
                
                r_ab = math.sqrt(x_ab*x_ab + y_ab*y_ab + z_ab*z_ab)

                one_d_r = 1.0 / r_ab

                sigma_d_r = self.get_chain_lj_sigma() * one_d_r

                tmp3 = sigma_d_r * sigma_d_r * sigma_d_r
                tmp6 = tmp3 * tmp3
                tmp12 = tmp6 * tmp6

                e += 4.0 * self.get_chain_lj_eps() * (tmp12 - tmp6)
                print iatom, jatom, 4.0 * self.get_chain_lj_eps() * (tmp12 - tmp6)
                
                ftmp = 4.0 * self.get_chain_lj_eps()
                ftmp *= 6.0 * tmp6 * one_d_r - 12.0 * tmp12 * one_d_r
                
                x_ab_norm = x_ab / r_ab
                y_ab_norm = y_ab / r_ab
                z_ab_norm = z_ab / r_ab
                
                f[ix] -= x_ab_norm * ftmp
                f[iy] -= y_ab_norm * ftmp
                f[iz] -= z_ab_norm * ftmp
                f[jx] += x_ab_norm * ftmp
                f[jy] += y_ab_norm * ftmp
                f[jz] += z_ab_norm * ftmp

        self.set_energy(e)
        self.set_forces(f)
        
                

