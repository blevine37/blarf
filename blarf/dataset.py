import numpy as np
import math
import h5py

class dataset():
    def __init__(self,nd):
        self.numdims = nd
        self.numpoints = 0
        self.positions = np.zeros((self.get_numpoints(),self.get_numdims()))
        self.energies_exact = np.zeros(self.get_numpoints())
        self.energies_approx = np.zeros(self.get_numpoints())
        self.internal_type = ""
        self.numinternals = 0
        self.internals = np.zeros((self.get_numpoints(),self.get_numinternals()))

    def allocate_arrays(self):
        self.positions = np.resize(self.positions,(self.get_numpoints(),self.get_numdims()))
        self.energies_exact = np.resize(self.energies_exact,self.get_numpoints())
        self.energies_approx = np.resize(self.energies_approx,self.get_numpoints())
        self.internals = np.resize(self.positions,(self.get_numpoints(),self.get_numinternals()))

    def set_numdims(self,n):
        self.numdims = n
        self.allocate_arrays()

    def get_numdims(self):
        return self.numdims

    def set_numpoints(self,n):
        self.numpoints = n
        self.allocate_arrays()

    def get_numpoints(self):
        return self.numpoints

    def set_positions(self,pos):
        self.positions = pos.copy()

    def get_positions(self):
        return self.positions.copy()

    def set_energies_exact(self,e):
        self.energies_exact = e.copy()

    def get_energies_exact(self):
        return self.energies_exact.copy()

    def set_energies_approx(self,e):
        self.energies_approx = e.copy()

    def get_energies_approx(self):
        return self.energies_approx.copy()

    def set_internals(self,pos):
        self.internals = pos.copy()

    def get_internals(self):
        return self.internals.copy()

    def add_point(self,pos,e_ex,e_ap):
        l = self.get_numpoints()+1
        self.set_numpoints(l)
        self.positions[l-1,:] = pos[:]
        self.energies_exact[l-1] = e_ex
        self.energies_approx[l-1] = e_ap
        
    def thin_dataset(self,stride):
        thinned_set = dataset(self.get_numdims())
        for ipoint in range(0,self.get_numpoints(),stride):
            pos = self.positions[ipoint,:].copy()
            e_ex = self.energies_exact[ipoint]
            e_ap = self.energies_approx[ipoint]
            thinned_set.add_point(pos,e_ex,e_ap)
        return thinned_set
        
    def get_position_element(self, i):
        return self.positions[i,:].copy()

    def get_internal_element(self, i):
        return self.internals[i,:].copy()

    def set_internal_type(self, typ):
        self.internal_type = typ

    def get_internal_type(self):
        return self.internal_type

    def get_mean_residual(self):
        return self.mean_residual

    def get_mean_unsigned_residual(self):
        return self.mean_unsigned_residual

    def compute_numinternals(self):
        self.numinternals = eval("self.compute_numinternals_" + self.get_internal_type() + "()")
        self.allocate_arrays()

    def compute_residual(self):
        self.residual = (self.energies_approx - self.energies_exact)
        self.mean_residual = (np.sum(self.residual) / self.get_numpoints())
        self.mean_unsigned_residual = (np.sum(np.absolute(self.residual)) / self.get_numpoints())

    def get_numinternals(self):
        return self.numinternals

    def h5_output(self,filename):
        h5f = h5py.File(filename, "w")
        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        for key in members:
            dset = h5f.create_dataset(key, data=eval("self." + key))

#############################################
# code for internal types
#############################################

# reciprical_bond

    def compute_numinternals_reciprical_bond(self):
        natoms = self.get_numdims() / 3
        return ((natoms * (natoms - 1)) / 2)
    
    def compute_internals(self):
        self.compute_numinternals()
        natoms = self.get_numdims() / 3
        for ipoint in range(self.get_numpoints()):
            r = self.get_position_element(ipoint)
            i_internal = 0
            for iatom in range(natoms-1):
                ix = 3*iatom
                iy = 3*iatom + 1
                iz = 3*iatom + 2
                
                for jatom in range(iatom+1,natoms):
                    jx = 3*jatom 
                    jy = 3*jatom + 1
                    jz = 3*jatom + 2
                    
                    x_ab = r[ix] - r[jx]
                    y_ab = r[iy] - r[jy]
                    z_ab = r[iz] - r[jz]
                    
                    r_ab = math.sqrt(x_ab*x_ab + y_ab*y_ab + z_ab*z_ab)
                    
                    one_d_r = 1.0 / r_ab
                    
                    self.internals[ipoint,i_internal] = one_d_r
                    
                    i_internal+=1
    

                    
