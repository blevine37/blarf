import numpy as np
import blarf

natoms = 5

ff = blarf.potential()

ff.init_chain(natoms,1.0,1.0,(1.414/1.122),0.1)

r = np.zeros(natoms*3)

r[5] = 1.0
r[7] = 1.1
r[8] = 1.0
r[10] = 1.1
r[11] = 1.9
r[13] = 0.9
r[14] = 3.0

print "r ", r

ff.eval_pes(r)

print "e ", ff.get_energy()
print "f ", ff.get_forces()

vv = blarf.classical()

vv.set_numdims(natoms*3)
vv.set_positions(r)
vv.set_momenta(np.zeros(natoms*3))
vv.set_timestep(0.001)
vv.set_maxtime(10.0)

vv.propagate(ff)


