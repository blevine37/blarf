import numpy as np
import random
import math

class cluster():
    def __init__(self,k,nd):
        self.k = k
        self.numdims = nd
        self.means = np.zeros((self.get_k(),self.get_numdims()))
        self.num_fixed_means = 0

    def allocate_arrays():
        np.resize(self.means,(self.get_k(),self.get_numdims()))

    def set_numdims(self,n):
        self.numdims = n
        self.allocate_arrays()

    def get_numdims(self):
        return self.numdims

    def set_nummeans(self,n):
        self.nummeans = n

    def get_nummeans(self):
        return self.nummeans

    def set_num_fixed_means(self,n):
        self.num_fixed_means = n

    def get_num_fixed_means(self):
        return self.num_fixed_means

    def set_k(self,k):
        self.k = k
        self.allocate_arrays()

    def get_k(self):
        return self.k

    def add_mean(self, mean, nummeans):
        self.means[nummeans,:] = mean[:]
        nummeans += 1
        return nummeans

    def get_means(self):
        return self.means.copy()

    def set_means(self,m):
        self.means = m.copy()

    def get_mean_element(self,imean):
        return self.means[imean,:].copy()

    def compute_distance(self,pos1,pos2):
        numdims = self.get_numdims()
        r = 0.0
        for i in range(numdims):
            diff = pos1[i] - pos2[i]
            r += diff * diff
        r = math.sqrt(r)
        return r

    def compute_shortest_distance(self,pos,nummeans=-1):
        if nummeans < 0:
            nummeans = self.get_k()
        rmin = 1.0e20
        for imean in range(nummeans):
            mean = self.get_mean_element(imean)
            r = self.compute_distance(mean,pos)
            if r < rmin:
                rmin = r
        return rmin        

    def compute_second_shortest_distance(self,pos,nummeans=-1):
        if nummeans < 0:
            nummeans = self.get_k()
        rmin = 1.0e20
        r2min = 1.0e19
        for imean in range(nummeans):
            mean = self.get_mean_element(imean)
            r = self.compute_distance(mean,pos)
            if r < rmin:
                r2min = rmin
                rmin = r
            else:
                if r < r2min:
                    r2min = r
        return r2min        

    def k_means_plus_plus(self,data):
        # randomly select first mean
        nummeans = self.get_num_fixed_means()
        numpoints = data.get_numpoints()
        ipoint = random.randrange(numpoints)
        mean = data.get_internal_element(ipoint)
        nummeans = self.add_mean(mean,nummeans)

        while nummeans < self.get_k():
            distances = np.zeros(numpoints)
            for ipoint in range(numpoints):
                pos = data.get_internal_element(ipoint)
                distances[ipoint] = self.compute_shortest_distance(pos,nummeans=nummeans)
            distmax = np.amax(distances)
            denom = distmax * distmax
            probabilities = (distances * distances) / denom
            print "prob ", probabilities
            ipoint = -1
            while ipoint == -1:
                jpoint = random.randrange(numpoints)
                draw  = random.random()
                if draw < probabilities[jpoint]:
                    ipoint = jpoint
            mean = data.get_internal_element(ipoint)
            nummeans = self.add_mean(mean,nummeans)

    def k_means_optimization(self,data):
        k = self.get_k()
        nfixed = self.get_num_fixed_means()
        ndims = self.get_numdims()
        
        self.k_means_plus_plus(data)

        iter = 0
        for iter in range(10000):
            prev_means = self.get_means()
            assignment = self.assign_to_means(data)
            current_means = self.compute_means(data,assignment)
            zdone = True
            for imean in range(nfixed,k):
                for idim in range(ndims):
                    if abs(prev_means[imean,idim]-current_means[imean,idim]) > 1.0e-8:
                        zdone=False
            if zdone:
                print "k-means converged"
                return
            current_means[0:nfixed,:] = prev_means[0:nfixed,:]
            self.set_means(current_means)
        
            iter += 1
        print "k-means maximum iterations reached???"

    def assign_to_means(self,data):
        npoints = data.get_numpoints()
        assignment = np.zeros(npoints,dtype=np.int8)
        
        for ipoint in range(npoints):
            pos = data.get_internal_element(ipoint)            
            assignment[ipoint] = self.find_closest_mean(pos)
        return assignment
            
    def find_closest_mean(self,pos):
        k = self.get_k()
        rmin = 1.0e20
        for imean in range(k):
            mean = self.get_mean_element(imean)
            r = self.compute_distance(mean,pos)
            if r < rmin:
                rmin = r
                i_closest_mean = imean
        return i_closest_mean       

    def compute_means(self,data,assignment):
        k = self.get_k()
        means = np.zeros((self.get_k(),self.get_numdims()))
        n = np.zeros(self.get_k())
        npoints = data.get_numpoints()
        for ipoint in range(npoints):
            means[assignment[ipoint],:] = means[assignment[ipoint],:] + data.get_internal_element(ipoint)
            n[assignment[ipoint]] += 1.0
        for imean in range(k):
            means[imean,:] = means[imean,:] / n[imean]
        return means
    


      
