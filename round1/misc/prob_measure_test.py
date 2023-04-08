from pyqpanda import *

if __name__ == "__main__":
    qvm = CPUQVM()
    qvm.init_qvm()
    qubits = qvm.qAlloc_many(2)
    cbits = qvm.cAlloc_many(2)

    prog = QProg()
    prog << H(qubits[0])\
        << CNOT(qubits[0], qubits[1])

    print("prob_run_dict: ")
    result1 = qvm.prob_run_dict(prog, qubits, -1)
    print(result1)

    print("prob_run_tuple_list: ")
    result2 = qvm.prob_run_tuple_list(prog, qubits, -1)
    print(result2)

    print("prob_run_list: ")
    result3 = qvm.prob_run_list(prog, qubits, -1)
    print(result3)
