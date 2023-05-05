import numpy as np

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import Aer, execute
from qiskit.compiler import transpile
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info import process_fidelity

from qiskit.visualization import plot_histogram

a = np.array([[1,1],[1/np.sqrt(2),-1/np.sqrt(2)]])
b = np.array([[1/2],[-1/np.sqrt(2)]])
x = np.linalg.solve(a, b)

print(a,'\n', b)
print(x)
print(np.allclose(np.dot(a, x), b))

import sympy as sp

sp.init_printing(use_unicode=True)

B_sp = sp.Matrix([[sp.sqrt(2)/4],[-sp.sqrt(2)/2]])

A_sp = sp.Matrix([[1,1],[1,-1]])/sp.sqrt(2)

P,D = A_sp.diagonalize()


# normalize the eigenvectors  第1列，第2列
# P_Asp_new[:,0] = P_Asp_new[:,0]/ P_Asp_new[:,0].norm()
# P_Asp_new[:,1] = P_Asp_new[:,1]/ P_Asp_new[:,1].norm()

# P_normal 和 P 形状一样
P_normal = P.copy() # 用copy()，不然会改变P的值



P_normal[:,0] = P[:,0]/ P[:,0].norm()
P_normal[:,1] = P[:,1]/ P[:,1].norm()


# M_ancilla make A become eigenvalues are 1 and 2
M_ancilla = P_normal * sp.Matrix([[-1,0],[0,2]]) * P_normal.inv()

M_ancilla, M_ancilla.evalf()

# M_ancilla * A_sp
P_new, D_new = (M_ancilla * A_sp).diagonalize()

# B_new = M_ancilla * B_sp
B_new = M_ancilla * B_sp

B_new.evalf()

# 归一化
B_new_normal = B_new / B_new.norm()

B_new_normal.evalf()

theta_bneg = sp.asin(- B_new_normal.evalf()[1])  # theta_bneg is beyond the range of [-pi/2,pi/2]

#  theta_b = pi - theta_bneg
theta_b = 2* (2*sp.pi + theta_bneg)- 2 * sp.pi

theta_b.evalf()

U_diag = sp.Matrix([[sp.I,0],[0,-1]])

U = P_normal * U_diag * P_normal.inv()

# U的共轭转置
U_dag = U.conjugate().transpose()

