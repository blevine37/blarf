import numpy as np
import math
from blarf.dataset import dataset
from blarf.cluster import cluster
from blarf.rbfn_center import rbfn_center
import h5py

class rbfn():
    def __init__(self):
        self.numcenters = 0
        self.numdims = 0
        self.centers = []
        self.numbf = 0
        self.width_factor = 2.0

    def init_from_cluster(self,clust):
        self.numcenters = clust.get_k()
        self.numdims = clust.get_numdims()

    def get_numcenters(self):
        return self.numcenters

    def get_numdims(self):
        return self.numdims

    def set_width_factor(self,wf):
        self.width_factor = wf

    def get_width_factor(self):
        return self.width_factor

    def add_center(self,cent):
        self.numcenters = self.get_numcenters()+1
        self.centers.append(cent)

    def get_centers(self):
        return self.centers[:]

    def get_numbf(self):
        return self.numbf

    def get_alpha(self):
        return self.alpha.copy()

    def compute_numbf(self):
        nbf = 0
        for icenter in range(self.get_numcenters()):
            nbf += self.centers[icenter].get_numbf()
        self.numbf = nbf

    def build_alpha(self):
        nbf = self.get_numbf()
        ncent = self.get_numcenters()
        self.alpha = np.zeros((nbf,self.get_numdims()))
        ibf = 0
        for icent in range(ncent):
            nbfcent = self.get_centers()[icent].get_numbf()
            self.alpha[ibf:(ibf+nbfcent),:] = self.get_centers()[icent].get_bf_widths()
            ibf += nbfcent
        print self.alpha
        
    def build_G(self,data):
        ndims = self.get_numdims()
        npoints = data.get_numpoints()
        print "npoints ", npoints
        ncent = self.get_numcenters()
        nbf = self.get_numbf()
        self.G = np.ones((npoints,nbf+1))

        alpha = self.get_alpha()
        r = data.get_internals()
        rprime = np.zeros_like(r)
        ibf = 1
        for icent in range(ncent):
            rcent = self.get_centers()[icent].get_positions()
            for ipoint in range(npoints):
                rprime[ipoint,:] = r[ipoint,:] - rcent
                rprime2 = rprime * rprime
            prods = np.matmul(rprime2,alpha.T)
            Gcol = np.exp(np.sum(prods,axis = 1))
            nbfcent = self.get_centers()[icent].get_numbf()
            for jbf in range(nbfcent):
                self.G[:,ibf] = Gcol
                ibf += 1
        print "G"
        print self.G

    def solve_weights(self,data):
        self.build_alpha()
        self.build_G(data)
        self.Ginv = np.linalg.pinv(self.G)
        print "Ginv"
        print self.Ginv
        e_exact = data.get_energies_exact()
        self.weights = np.matmul(self.Ginv,e_exact)
        print "weights"
        print self.weights

        data.set_energies_approx(np.matmul(self.G,self.weights))
        residual = data.compute_residual()
        print "residual"
        print residual
        meanresidual = np.sum(residual) / data.get_numpoints()
        meanunsignedresidual = np.sum(np.absolute(residual)) / data.get_numpoints()
        print "meanresidual, meanunsignedresidual ",meanresidual, meanunsignedresidual
        


    def h5_output(self,filename):
        h5f = h5py.File(filename, "w")
        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        for key in members:
            print eval("type(self." + key + ").__name__")
            if eval("type(self." + key + ").__name__") == "list":
                print eval("self." + key + "[0].__class__.__name__")
                if eval("self." + key + "[0].__class__.__name__") == "rbfn_center":
                    for key2 in eval("range(len(self." + key + "))"):
                        centgrp = h5f.create_group(key + "__" + str(key2))
                        eval("self." + key + "[" + str(key2) + "].h5_output(centgrp)")
            else:
                dset = h5f.create_dataset(key, data=eval("self." + key))
                        
                    
            #dset = h5f.create_dataset(key, data=eval("self." + key))

    
        
        

        
        
#################################
# code for internal types
#################################

# reciprical_bonds

    def init_from_cluster_reciprical_bonds_traditionalrbf(self,clust):
        ndims = clust.get_numdims()
        self.numdims = ndims

        for icenter in range(clust.get_k()):
            cent = rbfn_center(ndims)
            
            pos = clust.get_mean_element(icenter)

            cent.set_positions(pos)
            
            rmin = clust.compute_second_shortest_distance(pos)
            # width is in reciprical length^2 and is negative to avoid
            # the need for subsequent negation
            width = (0.5 / rmin)
            width = -1.0 * width * width

            cent.init_rbfn_center_reciprical_bonds_traditionalrbf(width)
            
            self.add_center(cent)
            print "ncenters ", self.get_numcenters()
            print "positions ", cent.get_positions()
            print "bf_icoords ", cent.get_bf_icoords()
            print "bf_widths ", cent.get_bf_widths()

        self.compute_numbf()
        print "numbf ", self.get_numbf()

    def init_from_cluster_reciprical_bonds_onedimensional(self,clust):
        ndims = clust.get_numdims()
        self.numdims = ndims

        for icenter in range(clust.get_k()):
            cent = rbfn_center(ndims)
            
            pos = clust.get_mean_element(icenter)

            cent.set_positions(pos)
            
            rmin = clust.compute_second_shortest_distance(pos)
            # width is in reciprical length^2 and is negative to avoid
            # the need for subsequent negation
            width = math.sqrt(float(ndims)) / (rmin * self.get_width_factor())
            width = -1.0 * width * width

            cent.init_rbfn_center_reciprical_bonds_onedimensional(width)
            
            self.add_center(cent)
            print "ncenters ", self.get_numcenters()
            print "positions ", cent.get_positions()
            print "bf_icoords ", cent.get_bf_icoords()
            print "bf_widths ", cent.get_bf_widths()

        self.compute_numbf()
        print "numbf ", self.get_numbf()