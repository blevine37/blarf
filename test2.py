import numpy as np
import blarf
import os

natoms = 5

k = 200

ff = blarf.potential()

ff.init_chain(natoms,1.0,1.0,(1.414/1.122),0.1)

r = np.zeros(natoms*3)

r[3] = 0.2
r[5] = 1.0
r[7] = 1.1
r[8] = 1.0
#3
r[10] = 1.1
r[11] = 1.9
r[13] = 0.9
r[14] = 3.0
# 5
#r[15] = 0.3
#r[16] = 0.9
#r[17] = 4.0
# 6
#r[19] = 1.1
#r[20] = 4.8

print "r ", r

ff.eval_pes(r)

print "e ", ff.get_energy()
print "f ", ff.get_forces()

vv = blarf.classical()

vv.set_numdims(natoms*3)
vv.set_positions(r)
vv.set_momenta(np.zeros(natoms*3))
vv.set_timestep(0.001)
vv.set_maxtime(1000.0)
vv.set_num_rescale_steps(100)
vv.set_temperature(0.1)

data = blarf.dataset(natoms*3)

vv.propagate(ff,data,stride=100)

print data.get_numpoints()
print data.get_positions()
print data.get_energies_exact()

thindata = data.thin_dataset(1)
thindata.set_internal_type("reciprical_bond")
thindata.compute_internals()

#width_factors = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
#numtrials = 5
width_factors = [32.0]
numtrials = 1

if os.path.exists("mur.dat"):
    os.remove("mur.dat")
    
iw = 0
for w in width_factors:
    murs = np.zeros(numtrials)
    for itrial in range(numtrials):

        clust = blarf.cluster(k,thindata.get_numinternals())
        
        clust.k_means_optimization(thindata)
        
        thindata.h5_output("thindata.hdf5")

        network = blarf.rbfn()

        network.set_width_factor(w)
        network.init_from_cluster_reciprical_bonds_traditionalrbf(clust)
        #network.init_from_cluster_reciprical_bonds_onedimensional(clust)

        network.solve_weights(thindata)

        filename = "network." + str(w) + "." + str(itrial) + ".hdf5"
        
        network.h5_output(filename)

        murs[itrial] = thindata.get_mean_unsigned_residual()  

    f = open('mur.dat', 'a')

    s = str(w) + "  "
    for imur in range(murs.size):
        s = s + str(murs[imur]) + "  "
    s = s + str(np.sum(murs) / murs.size) + "\n"
    
    f.write(s)
    
    f.close()

    iw += 1

        
