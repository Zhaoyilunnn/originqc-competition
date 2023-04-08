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
    qvm.run_with_configuration(prog, clbits, 1024)
    #print(qvm.get_qstate())
    sv = qvm.get_qstate()
    print("State vector before measurement::", sv)
    probs = np.abs(sv)**2
    print("Probs before measurement::", probs)


    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    I = np.array([[1, 0], [0, 1]])

    prob_list = [0, 0, 0, 0]
    ampls = [0, 0, 0, 0]
    sv_list = [None] * 4
    u_list = [None] * 4

    # Complete state
    # |\psi\rangle = |q2\rangle \otimes |q1\rangle \otimes |q0\rangle
    # Thus unitary should be u2 \otimes u1 \otimes u0
    u_list[0] = np.kron(I, np.kron(I, I)) # case 00
    u_list[1] = np.kron(np.kron(I, I), X) # case 01
    u_list[2] = np.kron(np.kron(I, I), Z) # case 10
    u_list[3] = np.kron(np.kron(I, I), Z@X) # case 11

    prob_psi = [0, 0]

    for i in range(4):
        prob_list[i] = probs[2*i] + probs[2*i + 1]
        if prob_list[i] > 0:
            print(i)
            ampls[i] = np.sqrt(1 / prob_list[i])
            sv_list[i] = np.array([sv[j]*ampls[i] if j == 2*i or j == 2*i+1 else 0 for j in range(len(sv))])
            print("sv before XZ::", sv_list[i])
            sv_list[i] = u_list[i] @ sv_list[i]
            sv_list[i] = np.abs(sv_list[i])**2
            prob_psi[0] = np.sum(sv_list[i][::2])
            prob_psi[1] = np.sum(sv_list[i][1::2])
            print(prob_psi)

    return prob_psi


if __name__ == '__main__':
    #print(question1(str(sys.argv[1])))
    question2(float(sys.argv[1]))
