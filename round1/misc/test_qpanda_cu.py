import numpy as np
from pyqpanda import *

qvm = CPUQVM()
qvm.init_qvm()

qubits = qvm.qAlloc_many(2)
clbits = qvm.cAlloc_many(2)

circ = QCircuit()
prog = QProg()

def my_cu(theta: float, phi: float, lam: float, gamma: float):

    return [1, 0, 0, 0,
            0, np.exp(1j*gamma)*np.cos(theta), 0, -np.exp(1j*(gamma + lam))*np.sin(theta),
            0, 0, 1, 0,
            0, np.exp(1j*(gamma+phi))*np.sin(theta), 0, np.exp(1j*(gamma+phi+lam))*np.cos(theta)
            ]


def my_cu_new(theta: float, phi: float, lam: float, gamma: float):

    return [1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, np.exp(1j*gamma)*np.cos(theta), -np.exp(1j*(gamma + lam))*np.sin(theta),
            0, 0, np.exp(1j*(gamma+phi))*np.sin(theta), np.exp(1j*(gamma+phi+lam))*np.cos(theta)
            ]

def mat_u(theta: float, phi: float, lam: float, gamma: float):
    return [np.exp(1j*gamma)*np.cos(theta), -np.exp(1j*(gamma + lam))*np.sin(theta), np.exp(1j*(gamma+phi))*np.sin(theta), np.exp(1j*(gamma+phi+lam))*np.cos(theta)]

def mat_ry(theta):
    return [np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta)]

#circ << CU(my_cu(np.pi/3, -2.97167419886273 + 5*np.pi/4, -2.97167419886273 + np.pi/4, 2.97167419886273), qubits[0], qubits[1])
#circ << CU(my_cu_new(np.pi/3, -2.97167419886273 + 5*np.pi/4, -2.97167419886273 + np.pi/4, 2.97167419886273), qubits[0], qubits[1])
#circ << CU(mat_u(np.pi/3, -2.97167419886273 + 5*np.pi/4, -2.97167419886273 + np.pi/4, 2.97167419886273), qubits[0], qubits[1])
#circ << CU(np.pi/3, -2.97167419886273 + 5*np.pi/4, -2.97167419886273 + np.pi/4, 2.97167419886273, qubits[0], qubits[1])
#circ << CU([0, 1, 1, 0], qubits[0], qubits[1])
#circ << CU([-0.85355339+0.14644661j, -0.35355339-0.35355339j, -0.35355339-0.35355339j, -0.14644661+0.85355339j], qubits[0], qubits[1])
#circ << CU([0.70710678+0.00000000e+00j, 0.70710678-8.65956056e-17j, 0.70710678+0.00000000e+00j, -0.70710678+8.65956056e-17j], qubits[0], qubits[1])
circ << CU(mat_ry(np.pi), qubits[0], qubits[1])
prog << circ << Measure(qubits[0], clbits[0])

print(draw_qprog(prog, output='text'))

qvm.init_state([0, 0, 0, 1])
#qvm.init_state([0.5, 0.5, 0.5, 0.5])

qvm.run_with_configuration(prog, clbits, 1024)
print(qvm.get_qstate())
