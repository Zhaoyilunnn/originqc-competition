import numpy as np

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute
from qiskit_aer import Aer
from qiskit.compiler import transpile
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info import process_fidelity

from qiskit.visualization import plot_histogram


# Create the various registers needed
clock = QuantumRegister(2, name='clock')
input = QuantumRegister(1, name='b')
ancilla = QuantumRegister(1, name='ancilla')
measurement = ClassicalRegister(2, name='c')

# Create an empty circuit with the specified registers
circuit = QuantumCircuit(ancilla, clock, input, measurement)

circuit.barrier()
print(circuit.draw(output='text'))

# State preparation.
# intial_state = [np.sqrt(5)/5,-2* np.sqrt(5)/5]
intial_state = [-0.248865947128923,0.968537939556125]

circuit.ry(2* np.pi - 2*np.arcsin(intial_state[1]), input[0])

circuit.barrier()

# draw the circuit
print(circuit.draw(output='text'))

# Perform a Hadamard Transform
circuit.h(clock)

# barrier
circuit.barrier()

#:::::::::::::sv0:::::::::::::
#circuit.save_state()

# draw the circuit
print(circuit.draw(output='text'))

def qft_dagger(circ, q):

    circ.swap(clock[0], clock[1])
    circ.h(clock[0])
    circ.cu1(-np.pi/2, clock[0], clock[1])
    circ.h(clock[1])


def qft(circ, q):
    circ.h(clock[1])
    circ.cu1(np.pi/2, clock[0], clock[1])
    circ.h(clock[0])
    circ.swap(clock[0], clock[1])

def qpe(circ, clock, target):


    # # e^{i*A*t}
    #  4 paras [pi/3, -2.97167419886273 + 5*pi/4, -2.97167419886273 + pi/4, 2.97167419886273]
    circuit.cu(np.pi/3, -2.97167419886273 + 5*np.pi/4, -2.97167419886273 + np.pi/4, 2.97167419886273, clock[0], input, label='U')

    # # e^{i*A*t*2}
    # 4 paras [pi/2,0,pi/0]
    circuit.cu(np.pi/2, 0, np.pi, 0, clock[1], input, label='U2')


    circuit.barrier();

    # Perform an inverse QFT on the register holding the eigenvalues
    qft_dagger(circuit, clock)

def inv_qpe(circ, clock, target):

    # Perform a QFT on the register holding the eigenvalues
    qft(circuit, clock)

    circuit.barrier()

    # # e^{i*A*t*2}
    # 4 paras [pi/2,0,pi/0]
    circuit.cu(np.pi/2, 0, np.pi, 0, clock[1], input, label='U2')

    # #circuit.barrier();

    # # e^{i*A*t}
    # 4 paras [pi/3, 2.97167419886273 - 5*pi/4, 2.97167419886273 - pi/4, -2.97167419886273 + 2*pi]
    circuit.cu(np.pi/3, 2.97167419886273 - 5*np.pi/4, 2.97167419886273 - np.pi/4, -2.97167419886273 + 2*np.pi, clock[0], input, label='U')


    circuit.barrier()

# Perform the QPE
qpe(circuit, clock, input)

##:::::::::::::sv1:::::::::::::
#circuit.save_state()

# draw the circuit
print(circuit.draw(output='text'))

# barrier
circuit.barrier()

# C-RY gates
# This section is to test and implement C = 1
circuit.cry(np.pi, clock[0], ancilla)
circuit.cry(np.pi/3, clock[1], ancilla)
##:::::::::::::sv2:::::::::::::
#circuit.save_state()

circuit.barrier()

circuit.measure(ancilla, measurement[0])
circuit.barrier()
##:::::::::::::sv3:::::::::::::
#circuit.save_state()

# Perform the inverse QPE
inv_qpe(circuit, clock, input)
#:::::::::::::sv3:::::::::::::
circuit.save_state()

# Perform a Hadamard Transform
circuit.h(clock)

circuit.barrier()


circuit.measure(input, measurement[1])

# draw the circuit
print(circuit.draw(output='text'))

# Execute the circuit using the simulator
simulator = Aer.get_backend('aer_simulator')

job = execute(circuit, backend=simulator, shots=65536)

#Get the result of the execution
result = job.result()

# Get the counts, the frequency of each answer
counts = result.get_counts(circuit)
print(counts)

#sv = result.get_statevector(0)
#print("sv0::", sv)
#sv = result.get_statevector(0)
#print("sv1::", sv.data)
sv = result.get_statevector(0)
print("sv2::", sv.data)

# Display the results
plot_histogram(counts)
