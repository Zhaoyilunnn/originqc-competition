from pyqpanda import *

if __name__ == "__main__":

    qvm = CPUQVM()
    qvm.init_qvm()
    qubits = qvm.qAlloc_many(3)
    cbits = qvm.cAlloc_many(3)
    cbits[0].set_val(0)
    cbits[1].set_val(3)

    prog = QProg()
    branch_true = QProg()
    branch_false = QProg()

    # 构建QIf正确分支以及错误分支
    branch_true << H(qubits[0])<< H(qubits[1]) << H(qubits[2])
    branch_false << H(qubits[0]) << CNOT(qubits[0], qubits[1]) << CNOT(qubits[1], qubits[2])

    # 构建QIf
    qif = QIfProg(cbits[0] > cbits[1], branch_true, branch_false)

    # QIf插入到量子程序中
    prog << qif

    print(draw_qprog(prog))

    # 概率测量，并返回目标量子比特的概率测量结果，下标为十进制
    result = qvm.prob_run_tuple_list(prog, qubits, -1)

    # 打印概率测量结果
    print(result)
