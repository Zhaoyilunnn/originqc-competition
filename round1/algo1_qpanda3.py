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

    circ << H(qubits[1]) \
         << CNOT(qubits[1], qubits[0])

    prog << circ << Measure(qubits[0], clbits[0])

    init_state = [0, 0, 0, 0]
    init_state[int(input, 2)] = 1
    print(init_state)
    qvm.init_state(init_state)

    # As long as we set shots>1, qpanda will return full state
    res = qvm.run_with_configuration(prog, clbits, 2)
    return qvm.get_qstate()



def question2(theta: float) -> list:
    qvm = CPUQVM()
    qvm.init_qvm()
    qubits = qvm.qAlloc_many(3)
    clbits = qvm.cAlloc_many(3)

    prog = QProg()
    circ = QCircuit()

    circ << H(qubits[1]) \
         << CNOT(qubits[1], qubits[0]) \
         << RY(qubits[2], theta) \
         << CNOT(qubits[2], qubits[1]) \
         << H(qubits[2])

    prog << circ << Measure(qubits[1], clbits[1]) << Measure(qubits[2], clbits[2])

    #print(draw_qprog(prog))
    #qvm.run_with_configuration(prog, clbits, 1024)
    #print(qvm.get_qstate())
    counts = [0, 0]
    for _ in range(1024):
        qvm.run_with_configuration(prog, clbits, 1)
        prog_new = deep_copy(prog)
        if clbits[1].get_val() == 1:
            prog_new << X(qubits[0])
        if clbits[2].get_val() == 1:
            prog_new << Z(qubits[0])
        prog_new << Measure(qubits[0], clbits[0])
        #print(draw_qprog(prog_new))
        qvm.run_with_configuration(prog_new, clbits, 1)
        #print(clbits[0].get_val())
        counts[clbits[0].get_val()] += 1
    print(counts)


if __name__ == '__main__':
    #print(question1(str(sys.argv[1])))
    question2(float(sys.argv[1]))
