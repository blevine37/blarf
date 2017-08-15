import numpy as np
import math
from blarf.dataset import dataset
from blarf.cluster import cluster

class rbfn_center():
    def __init__(self,nd):
        self.numdims = nd
        self.positions = np.zeros(nd)
        self.numbf = 0
        self.bf_icoords = np.zeros((0,nd),dtype=np.int8)
        self.bf_widths = np.zeros((0,nd))

    def get_numdims(self):
        return self.numdims

    def set_positions(self,pos):
        self.positions = pos.copy()

    def get_positions(self):
        return self.positions.copy()

    def set_numbf(self,nbf):
        self.numbf = nbf
        nd = self.get_numdims()
        self.bf_icoords = np.resize(self.bf_icoords,(nbf,nd))
        self.bf_widths = np.resize(self.bf_widths,(nbf,nd))

    def get_numbf(self):
        return self.numbf

    def set_bf_icoords(self,ic):
        self.bf_icoords = ic.copy()

    def get_bf_icoords(self):
        return self.bf_icoords.copy()

    def add_bf(self,ic,w):
        nbf = self.get_numbf() + 1
        self.set_numbf(nbf)
        self.bf_icoords[nbf-1,:] = ic
        self.bf_widths[nbf-1,:] = w
        
    def set_bf_widths(self,w):
        self.bf_widths = w.copy()

    def get_bf_widths(self):
        return self.bf_widths.copy()

    def h5_output(self,centgrp):
        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        for key in members:
            dset = centgrp.create_dataset(key, data=eval("self." + key))

#################################
# code for internal types
#################################

# reciprical_bonds

    def init_rbfn_center_reciprical_bonds_traditionalrbf(self,width):
        ic = np.ones(self.get_numdims(),dtype=np.int8)
        w = width*np.ones(self.get_numdims())
        self.add_bf(ic,w)
        
    def init_rbfn_center_reciprical_bonds_onedimensional(self,width):
        for idim in range(self.get_numdims()):
            ic = np.zeros(self.get_numdims(),dtype=np.int8)
            ic[idim] = 1
            w = np.zeros(self.get_numdims())
            w[idim] = width
            self.add_bf(ic,w)
        
