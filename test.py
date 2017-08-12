import numpy as np
import blarf

natoms = 3

ff = blarf.potential()

ff.init_chain(natoms,1.0,1.0,(1.314/1.122),0.1)

r = np.zeros(natoms*3)

r[5] = 1.0
r[7] = 1.0
r[8] = 1.0

print "r ", r

ff.eval_forcefield(r)

print "e ", ff.get_energy()
print "f ", ff.get_forces()

