import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

qc = QuantumCircuit(2)
#qc.initialize([0.5,0.5,0.5,0.5])
#qc.initialize([0,0,0,1])
#qc.cu(np.pi/3, -2.97167419886273 + 5*np.pi/4, -2.97167419886273 + np.pi/4, 2.97167419886273, 0, 1)
qc.cu(np.pi/3, 2.97167419886273 - 5*np.pi/4, 2.97167419886273 - np.pi/4, -2.97167419886273 + 2*np.pi, 0, 1)
#qc.cu(np.pi/2, 0, np.pi, 0, 0, 1)
#qc.cry(np.pi, 0, 1)
#qc.cx(0, 1)
#qc.measure_all()
print(qc.draw(output='text'))

sim = Aer.get_backend("unitary_simulator")
#sim = Aer.get_backend("statevector_simulator")
#sim.set_options(method="unitary")

qc = transpile(qc, sim)
res = sim.run(qc).result()
print(res.get_unitary())
#print(res.get_statevector())
