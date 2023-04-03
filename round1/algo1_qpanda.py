from pyqpanda import *
import math
import numpy as np
import sys


def question1(input: str) -> list:
    qvm = CPUQVM()
    qvm.init_qvm()
    qubits = qvm.qAlloc_many(2)
    clbits = qvm.cAlloc_many(2)

    prog = QProg()
    circ = QCircuit()

    circ << H(qubits[0]) \
         << CNOT(qubits[0], qubits[1])

    prog << circ << Measure(qubits[0], clbits[0])

    init_state = [0, 0, 0, 0]
    init_state[int(input, 2)] = 1
    print(init_state)
    qvm.init_state(init_state)
    res = qvm.run_with_configuration(prog, clbits, 1024)
    return qvm.get_qstate()



def question2(theta: float) -> list:
    pass


if __name__ == '__main__':
    print(question1(str(sys.argv[1])))
