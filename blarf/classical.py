import numpy as np
import math
from blarf.potential import potential
from blarf.dataset import dataset

class classical():
    def __init__(self):
        self.numdims = 0
        self.positions = np.zeros(self.get_numdims())
        self.momenta = np.zeros(self.get_numdims())
        self.masses = np.ones(self.get_numdims())
        self.timestep = 0.0
        self.time = 0.0
        self.maxtime = 0.0
        self.temperature = -1.0
        self.num_rescale_steps = -1

    def set_numdims(self,n):
        self.numdims = n
        self.positions = np.zeros(n)
        self.momenta = np.zeros(n)
        self.masses = np.ones(self.get_numdims())

    def get_numdims(self):
        return self.numdims

    def set_positions(self,q):
        self.positions = q.copy()

    def get_positions(self):
        return self.positions.copy()
    
    def set_momenta(self,p):
        self.momenta = p.copy()

    def get_momenta(self):
        return self.momenta.copy()

    def set_masses(self,m):
        self.masses = m.copy()

    def get_masses(self):
        return self.masses.copy()

    def set_timestep(self,h):
        self.timestep = h

    def get_timestep(self):
        return self.timestep

    def set_time(self,t):
        self.time = t

    def get_time(self):
        return self.time

    def set_maxtime(self,t):
        self.maxtime = t

    def get_maxtime(self):
        return self.maxtime

    def set_temperature(self,t):
        self.temperature = t

    def get_temperature(self):
        return self.temperature

    def set_num_rescale_steps(self,n):
        self.num_rescale_steps = n

    def get_num_rescale_steps(self):
        return self.num_rescale_steps

    def compute_kinetic_energy(self):
        ke = 0.0
        p = self.get_momenta()
        m = self.get_masses()
        for idim in range(self.get_numdims()):
            ke += 0.5 * p[idim] * p[idim] / m[idim]
        return ke

    def propagate_step(self, pes):
        q_t = self.get_positions()
        p_t = self.get_momenta()
        m = self.get_masses()
        v_t = p_t / m
        h = self.get_timestep()
        t = self.get_time()
        
        # taking all masses to be 1.0
        
        pes.eval_pes(q_t)

        f_t = pes.get_forces()

        a_t = f_t / m

        v_tphdt = v_t + 0.5 * h * a_t

        q_tpdt = q_t + h * v_tphdt

        pes.eval_pes(q_tpdt)

        f_tpdt = pes.get_forces()

        a_tpdt = f_t / m

        v_tpdt = v_tphdt + 0.5 * h * a_tpdt

        p_tpdt = v_tpdt * m

        tpdt = t + h

        self.set_positions(q_tpdt)
        self.set_momenta(p_tpdt)
        self.set_time(tpdt)

    def propagate(self, pes, data, stride = 1):
        istep = 0
        while self.get_time() < self.get_maxtime():
            self.propagate_step(pes)
            ke = self.compute_kinetic_energy()
            pe = pes.get_energy()
            totale = ke + pe
            #print "KE, PE, E ", ke, pe, totale
            if self.get_num_rescale_steps() > 0:
                if istep%self.get_num_rescale_steps() == 0:
                    self.rescale_momentum()
            if istep%stride == 0:
                data.add_point(self.get_positions(),pe,0.0)
            istep += 1

    def rescale_momentum(self):
        ke = self.compute_kinetic_energy()
        temp = self.get_temperature()
        ndims = self.get_numdims()
        factor = math.sqrt( temp * float(ndims) / ke )
        mom = self.get_momenta()
        mom = factor * mom
        self.set_momenta(mom)
            
            
            
            

        
    
