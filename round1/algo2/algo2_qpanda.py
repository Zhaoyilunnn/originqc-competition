import numpy as np
from typing import List
from pyqpanda import *


def mat_ry(theta):
    return [np.cos(theta/2), -np.sin(theta)/2, np.sin(theta/2), np.cos(theta/2)]


def qft_dagger(qprog: QProg, clock: List[Qubit]):
    assert len(clock) == 2
    qprog << SWAP(clock[0], clock[1]) \
          << H(clock[0]) \
          << CR(clock[0], clock[1], -np.pi/2) \
          << H(clock[1])


def qft(qprog: QProg, clock: List[Qubit]):
    assert len(clock) == 2
    qprog << H(clock[1]) \
          << CR(clock[0], clock[1], np.pi/2) \
          << H(clock[0]) \
          << SWAP(clock[0], clock[1])


#TODO(zhaoyilun): Change unitary to be derived from angles
def qpe(qprog: QProg, clock: List[Qubit], input: Qubit):
    assert len(clock) == 2
    qprog << CU([-0.85355339+0.14644661j, -0.35355339-0.35355339j, -0.35355339-0.35355339j, -0.14644661+0.85355339j], clock[0], input) \
          << CU([0.70710678+0.00000000e+00j, 0.70710678-8.65956056e-17j, 0.70710678+0.00000000e+00j, -0.70710678+8.65956056e-17j], clock[1], input)
    qft_dagger(qprog, clock)


def inv_qpe(qprog: QProg, clock: List[Qubit], input: Qubit):
    assert len(clock) == 2
    qft(qprog, clock)
    qprog << CU([0.70710678+0.00000000e+00j, 0.70710678-8.65956056e-17j, 0.70710678+0.00000000e+00j, -0.70710678+8.65956056e-17j], clock[1], input) \
          << CU([-0.85355339-0.14644661j, -0.35355339+0.35355339j, -0.35355339+0.35355339j, -0.14644661-0.85355339j], clock[0], input)


def main():
    QVM = CPUQVM()
    QVM.init_qvm()

    qubits = QVM.qAlloc_many(4)
    clbits = QVM.cAlloc_many(2)

    ancilla = qubits[0]
    clock = qubits[1:3]
    input = qubits[3]

    prog = QProg()

    init_state = [-0.248865947128923,0.968537939556125]

    prog << RY(input, 2*np.pi - 2*np.arcsin(init_state[1]))
    prog << H(clock[0]) << H(clock[1]) \

    # Perform the QPE
    qpe(prog, clock, input)

    # C-RY gates
    # This section is to test and implement C = 1
    #print(draw_qprog(prog, output='text'))

    prog << CU(mat_ry(np.pi), clock[0], ancilla) \
         << CU(mat_ry(np.pi/3), clock[1], ancilla)

    #print(draw_qprog(prog, output='text'))

    prog << Measure(ancilla, clbits[0])

    # Perform the inverse QPE
    inv_qpe(prog, clock, input)
    #print(draw_qprog(prog, output='text'))

    # Perform a Hadamard Transform
    prog << H(clock[0]) << H(clock[1])

    prog << Measure(input, clbits[1])

    print(draw_qprog(prog, output='text'))

    # Run simulation
    res = QVM.run_with_configuration(prog, clbits, 65536)
    print(res)


if __name__ == '__main__':
    main()
