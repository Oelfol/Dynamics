# joel_circuit.py
# Code for testing special arrangements with the example Joel 2a

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import math
import numpy as np




def init_state(qc, ancilla, initial_state, n):
    state_temp, anc, index = np.binary_repr(initial_state).zfill(n)[::-1], int(ancilla), 0
    for x in state_temp:
        if x == '1':
            qc.x(index + anc)
        index += 1


def magnetization(t, dt, site, initialstate, j, epsilon, a, n):

    t_ = t
    if t == 0:
        t_ = 1
        print("First order trotter in progress..")
    trotter_steps = math.ceil(abs(j) * t_ * dt / epsilon)
    print('trotter steps:', trotter_steps, " t:", t)

    # Address needed constants for particular paper
    pseudo_constant_a = 4.0
    delta = j * t * dt / (trotter_steps * pseudo_constant_a)
    qreg_q = QuantumRegister(7, 'q')
    creg_c = ClassicalRegister(1, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)
    ancilla = 1  # need this argument back
    init_state(circuit, ancilla, initialstate, n)
    circuit.barrier(qreg_q[1], qreg_q[2], qreg_q[3], qreg_q[4], qreg_q[5], qreg_q[6])
    for step in range(trotter_steps):
        qreg_q = [1, 2, 3, 4, 5, 6]
        circuit.cx(qreg_q[0], qreg_q[1])
        circuit.h(qreg_q[2])
        circuit.h(qreg_q[3])
        circuit.cx(qreg_q[5], qreg_q[4])
        circuit.rx(2 * delta - math.pi / 2, qreg_q[0])
        circuit.rz(2 * delta * a, qreg_q[1])
        circuit.rz(2 * delta * a, qreg_q[4])
        circuit.rx(2 * delta - math.pi / 2, qreg_q[5])
        circuit.h(qreg_q[1])
        circuit.h(qreg_q[4])
        circuit.cx(qreg_q[1], qreg_q[0])
        circuit.cx(qreg_q[4], qreg_q[5])
        circuit.rx(-2 * delta, qreg_q[1])
        circuit.rx(-2 * delta, qreg_q[4])
        circuit.cx(qreg_q[1], qreg_q[2])
        circuit.h(qreg_q[1])
        circuit.rz(2 * delta - math.pi / 2, qreg_q[2])
        circuit.cx(qreg_q[4], qreg_q[3])
        circuit.cx(qreg_q[0], qreg_q[1])
        circuit.rz(2 * delta - math.pi / 2, qreg_q[3])
        circuit.h(qreg_q[4])
        circuit.rx(math.pi / 2, qreg_q[0])
        circuit.rx(-1 * math.pi / 2, qreg_q[1])
        circuit.cx(qreg_q[5], qreg_q[4])
        circuit.rz(2 * delta * a, qreg_q[1])
        circuit.rx(-1 * math.pi / 2, qreg_q[4])
        circuit.rx(math.pi / 2, qreg_q[5])
        circuit.cx(qreg_q[2], qreg_q[1])
        circuit.rz(2 * delta * a, qreg_q[4])
        circuit.rz(-2 * delta, qreg_q[1])
        circuit.h(qreg_q[2])
        circuit.cx(qreg_q[3], qreg_q[4])
        circuit.cx(qreg_q[2], qreg_q[1])
        circuit.h(qreg_q[3])
        circuit.rz(-2 * delta, qreg_q[4])
        circuit.rx(-1 * math.pi / 2, qreg_q[1])
        circuit.rx(math.pi / 2, qreg_q[2])
        circuit.cx(qreg_q[3], qreg_q[4])
        circuit.rx(math.pi / 2, qreg_q[3])
        circuit.rx(-1 * math.pi / 2, qreg_q[4])
        circuit.cx(qreg_q[2], qreg_q[3])
        circuit.rx(2 * delta - math.pi / 2, qreg_q[2])
        circuit.rz(2 * delta * a, qreg_q[3])
        circuit.h(qreg_q[2])
        circuit.cx(qreg_q[2], qreg_q[3])
        circuit.h(qreg_q[2])
        circuit.rz(-2 * delta, qreg_q[3])
        circuit.cx(qreg_q[2], qreg_q[3])
        circuit.rx(math.pi / 2, qreg_q[2])
        circuit.rx(-1 * math.pi / 2, qreg_q[3])
    circuit.barrier()
    circuit.h(0)
    circuit.cz(control_qubit=0, target_qubit=site + 1)
    circuit.h(0)

    return circuit

