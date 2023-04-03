import numpy as np

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import Aer
from qiskit.compiler import transpile
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info import process_fidelity

theta = 0.5

qr = QuantumRegister(3)
bob_bell_cr = ClassicalRegister(1)
alice_bell_cr = ClassicalRegister(1)
alice_psi_cr = ClassicalRegister(1)
qte = QuantumCircuit(qr)
qte.add_register(bob_bell_cr)
qte.add_register(alice_bell_cr)
qte.add_register(alice_psi_cr)

qte.h(1)
qte.cx(1, 0)

# q2 applies ry, has a parameter theta

qte.ry(theta, 2)

# q2 is transmitted to q0
qte.cx(2, 1)
qte.h(2)

qte.measure(qr[2], alice_psi_cr)
qte.measure(qr[1], alice_bell_cr)

#  c_if by q1 q2
qte.x(0).c_if(alice_bell_cr, 1)
qte.z(0).c_if(alice_psi_cr, 1)

# # # q0 measurement store in bob_bell_cr
# qte.measure(0, bob_bell_cr)


# backend = Aer.get_backend('aer_simulator')
# job = backend.run(qte)
# result = job.result()

# # get counts
# counts = result.get_counts(qte)
# print(counts)


# qte.draw(output='mpl')

# using statevector simulator
backend = Aer.get_backend('statevector_simulator')
job = backend.run(qte)
result = job.result()

# get statevector
statevector = result.get_statevector(qte)
print(statevector)

# statevector to array
statevector = np.array(statevector)

statevector = np.abs(statevector)**2

print(statevector)

odd = statevector[1::2]
odd = np.sum(odd)

even = statevector[::2]
even = np.sum(even)

print('q0 is 0',even)
print('q0 is 1',odd)

print([even,odd])
