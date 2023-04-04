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
         << H(qubits[1])


    counts = [0, 0]

    branch_true_0 = QProg()
    branch_true_1 = QProg()

    branch_true_0 << X(qubits[0])
    branch_true_1 << Z(qubits[0])

    qif0 = QIfProg(clbits[1] == 1, branch_true_0)
    qif1 = QIfProg(clbits[2] == 1, branch_true_1)

    prog << circ \
         << Measure(qubits[2], clbits[2]) \
         << Measure(qubits[1], clbits[1]) \
         << qif0 \
         << qif1 \
         << Measure(qubits[0], clbits[0])

    result = qvm.prob_run_list(prog, [qubits[0]], -1)

    return result

    #for _ in range(4096):

    #    # 1. Run first part of circuit before measurement
    #    res = qvm.run_with_configuration(prog, clbits, 2)
    #    cur_state = qvm.get_qstate()
    #    alice_bell_cr = clbits[1].get_val()
    #    alice_psi_cr = clbits[2].get_val()
    #    print(alice_bell_cr, alice_psi_cr, clbits[0].get_val())
    #    #print(cur_state)

    #    qvm2 = CPUQVM()
    #    qvm2.init_qvm()
    #    qubits2 = qvm2.qAlloc_many(3)
    #    clbits2 = qvm2.cAlloc_many(3)

    #    circ2 = QCircuit()
    #    circ3 = QCircuit()
    #    circ2 << X(qubits2[0])
    #    circ3 << Z(qubits2[0])

    #    prog2 = QProg()

    #    if alice_bell_cr == 1:
    #        prog2 << circ2
    #    if alice_psi_cr == 1:
    #        prog2 << circ3
    #    prog2 << Measure(qubits2[0], clbits2[0])
    #    print(draw_qprog(prog2))


    #    qvm2.init_state(cur_state)
    #    #print(qvm2.get_qstate())
    #    qvm2.prob_run_tuple_list()(prog2, clbits2, 1024)
    #    print(qvm2.get_qstate())
    #    ##print(clbits2[0].get_val())
    #    res = clbits2[0].get_val()
    #    counts[res] += 1
    #print(counts)


if __name__ == '__main__':
    #print(question1(str(sys.argv[1])))
    question2(float(sys.argv[1]))
