import numpy as np
import blarf
import os

natoms = 6

ff = blarf.potential()

timemax = 1000.0
stride = 100
timestep = 0.001

ff.init_ethylene(1.0,0.05,0.3)

r = np.zeros(natoms*3)

r[5] = 1.1
r[7] = 0.7
r[8] = -0.5
r[9] = 0.1
r[10] = -0.7
r[11] = -0.5
r[12] = 0.7
r[14] = 1.5
r[15] = -0.6
r[17] = 1.5


print "r ", r

ff.eval_pes(r)

print "e ", ff.get_energy()
print "f ", ff.get_forces()

vv = blarf.classical()

vv.set_numdims(natoms*3)

vv.set_numdims(natoms*3)
vv.set_positions(r)
vv.set_momenta(np.zeros(natoms*3))
vv.set_timestep(timestep)
vv.set_maxtime(timemax)
vv.set_num_rescale_steps(100)
vv.set_temperature(0.3)

data = blarf.dataset(natoms*3)

vv.propagate(ff,data,stride=stride)

print data.get_numpoints()
print data.get_positions()
print data.get_energies_exact()

thindata = data.thin_dataset(1)

thindata.set_internal_type("reciprical_bond")
thindata.compute_internals()

r2 = np.zeros(natoms*3)
r2[5] = 1.2
r2[6] = -0.4
r2[7] = 0.5
r2[8] = -0.6
r2[9] = 0.3
r2[10] = -0.5
r2[11] = -0.5
r2[12] = 0.7
r2[14] = 1.4
r2[15] = -0.6
r2[16] = 0.1
r2[17] = 1.6

vv2 = blarf.classical()

vv2.set_numdims(natoms*3)
vv2.set_positions(r2)
vv2.set_momenta(np.zeros(natoms*3))
vv2.set_timestep(timestep)
vv2.set_maxtime(timemax)
vv2.set_num_rescale_steps(100)
vv2.set_temperature(0.3)

data2 = blarf.dataset(natoms*3)

vv2.propagate(ff,data2,stride=stride)

thindata2 = data2.thin_dataset(1)
thindata2.set_internal_type("reciprical_bond")
thindata2.compute_internals()

#ks = [10, 20, 40, 80, 160, 320, 640]
#width_factors = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
rcs = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001]
numtrials = 3
#width_factors = [2.0, 4.0]
#numtrials = 2

if os.path.exists("mur7-2-eth.dat"):
    os.remove("mur7-2-eth.dat")

if os.path.exists("mur7-2x-eth.dat"):
    os.remove("mur7-2x-eth.dat")

w=1.0
k=100

iw = 0

clust = blarf.cluster(k,thindata.get_numinternals())
        
clust.k_means_optimization(thindata)
        
#thindata.h5_output("thindata3.hdf5")

network = blarf.rbfn()

network.set_width_factor(w)

#network.init_from_cluster_reciprical_bonds_traditionalrbf(clust)
network.init_from_cluster_reciprical_bonds_onedimensional(clust)

#        network.solve_weights(thindata)
network.optimize_regularization_constant(-6,thindata,thindata2)

        #filename = "network3." + str(w) + "." + str(itrial) + ".hdf5"
        
        #network.h5_output(filename)

print "fit mur ", thindata.get_mean_unsigned_residual()
print "cross mur ", thindata2.get_mean_unsigned_residual()


        
