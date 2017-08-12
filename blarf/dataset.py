import numpy as np

class dataset():
    def __init__(self,nd):
        self.numdims = nd
        self.numpoints = 0
        self.positions = np.zeros((self.get_numpoints(),self.get_numdims()))
        self.energies_exact = np.zeros(self.get_numpoints())
        self.energies_approx = np.zeros(self.get_numpoints())

    def allocate_arrays(self):
        self.positions = np.resize(self.positions,(self.get_numpoints(),self.get_numdims()))
        self.energies_exact = np.resize(self.energies_exact,self.get_numpoints())
        self.energies_approx = np.resize(self.energies_approx,self.get_numpoints())

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

    def add_point(self,pos,e_ex,e_ap):
        l = self.get_numpoints()+1
        self.set_numpoints(l)
        self.positions[l-1,:] = pos[:]
        self.energies_exact[l-1] = e_ex
        self.energies_approx[l-1] = e_ap
        
