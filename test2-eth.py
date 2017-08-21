import numpy as np
import blarf
import os

natoms = 6

k = 6

ff = blarf.potential()

timemax = 10.0
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

#width_factors = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
#numtrials = 5
width_factors = [1.0]
numtrials = 1

if os.path.exists("mur.dat"):
    os.remove("mur.dat")
    
if os.path.exists("mur2.dat"):
    os.remove("mur2.dat")
    
iw = 0
for w in width_factors:
    murs = np.zeros(numtrials)
    murs2 = np.zeros(numtrials)
    for itrial in range(numtrials):

        clust = blarf.cluster(k,thindata.get_numinternals())
        
        clust.k_means_optimization(thindata)
        
        network = blarf.rbfn()

        network.set_width_factor(w)
        network.set_regularization_constant(1.0)
        #network.init_from_cluster_reciprical_bonds_traditionalrbf(clust)
        network.init_from_cluster_reciprical_bonds_onedimensional(clust)

        network.solve_weights(thindata)

        #filename = "network." + str(w) + "." + str(itrial) + ".hdf5"
        
        #network.h5_output(filename)

        murs[itrial] = thindata.get_mean_unsigned_residual()  

        #thindata.h5_output("thindata.hdf5")

        network.compute_energies_approx(thindata2)
        
        murs2[itrial] = thindata2.get_mean_unsigned_residual()  
        
        #thindata2.h5_output("thindata2.hdf5")

    f = open('mur2-eth.dat', 'a')

    s = str(w) + "  "
    for imur in range(murs.size):
        s = s + str(murs[imur]) + "  "
    s = s + str(np.sum(murs) / murs.size) + "\n"
    
    f.write(s)
    
    f.close()
    
    f = open('mur2x-eth.dat', 'a')

    s = str(w) + "  "
    for imur2 in range(murs2.size):
        s = s + str(murs2[imur2]) + "  "
    s = s + str(np.sum(murs2) / murs2.size) + "\n"
    
    f.write(s)
    
    f.close()

    iw += 1

        
