import numpy as np
import blarf

natoms = 7

k = 50

ff = blarf.potential()

ff.init_chain(natoms,1.0,1.0,(1.414/1.122),0.1)

r = np.zeros(natoms*3)

r[3] = 0.2
r[5] = 1.0
r[7] = 1.1
r[8] = 1.0
r[10] = 1.1
r[11] = 1.9
r[13] = 0.9
r[14] = 3.0
# 5
r[15] = 0.3
r[16] = 0.9
r[17] = 4.0
# 6
r[19] = 1.1
r[20] = 4.8

print "r ", r

ff.eval_pes(r)

print "e ", ff.get_energy()
print "f ", ff.get_forces()

vv = blarf.classical()

vv.set_numdims(natoms*3)
vv.set_positions(r)
vv.set_momenta(np.zeros(natoms*3))
vv.set_timestep(0.001)
vv.set_maxtime(100.0)
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

print thindata.get_numpoints()
print thindata.get_positions()
print thindata.get_energies_exact()
print thindata.get_numinternals()
print thindata.get_internals()
print thindata.get_internal_element(100)

clust = blarf.cluster(k,thindata.get_numinternals())

clust.k_means_optimization(thindata)
print clust.get_mean_element(0)
print clust.get_mean_element(1)
print clust.get_mean_element(2)
print clust.get_mean_element(3)
print clust.get_mean_element(4)
