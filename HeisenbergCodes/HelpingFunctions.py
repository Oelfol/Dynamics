###########################################################################
# HelpingFunctions.py
# Part of HeisenbergCodes
# Updated January '21
#
# General accessory functions, including read/write.
# Current Qiskit:
# qiskit 0.21.0
# qiskit-terra 0.15.2
# qiskit-aer 0.6.1
# qiskit-ibmq-provider 0.9.0

# WIP
###########################################################################

###########################################################################
# CONTENTS:
# get_pauli_matrices()
# init_spin_state(initialpsi, num_states)
# spin_op(operator, site, n, unity)
# gen_pairs(n, auto, open_chain)
# gen_m(leng, steps)
# commutes(a, b)
# sort_counts(count, qs, shots)
# sort_counts_no_div(count, qs)
# choose_control_gate(choice, qc, c, t)
# real_or_imag_measurement(qc, j)
# zz_operation(qc, a, b, delta)
# three_cnot_evolution(qc, pair, ancilla, j, t, dt, trotter_steps, ising, a_constant)
# execute_real(qc, device, shots)
# run_circuit(anc, qc, noise, ibmq_params, n)
# write_data(vec, loc)
# write_numpy_array(array, loc)
# read_numpy_array(filename)
# read_var_file(filename)
###########################################################################

from qiskit import execute
import numpy as np
import scipy.sparse as sps
import math
import csv
import IBMQSetup as setup
from qiskit.compiler import transpile
from qiskit.transpiler.passes import CrosstalkAdaptiveSchedule
from qiskit.converters import circuit_to_dag
from qiskit.converters import dag_to_circuit
import crosstalkAdaptiveScheduleMod as cas


# ========================================= Pauli matrices and pseudospin operators ================================== >

sx = sps.csc_matrix(np.array([[0, 1], [1, 0]]))
sy = sps.csc_matrix(np.complex(0, 1) * np.array([[0, -1], [1, 0]]))
sz = sps.csc_matrix(np.array([[1, 0], [0, -1]]))
identity = sps.csc_matrix(np.eye(2, dtype=complex))
plus = sps.csc_matrix(sx * (1 / 2) + np.complex(0, 1) * sy * (1 / 2))
minus = sps.csc_matrix(sx * (1 / 2) - np.complex(0, 1) * sy * (1 / 2))


def get_pauli_matrices():
    return sx, sy, sz, identity, plus, minus

# ======================================= Helper functions for classical calculations ================================ >


def init_spin_state(initialpsi, num_states):
    # initial psi is the computational-basis index
    psi_0 = np.zeros([num_states, 1], complex)
    psi_0[initialpsi, 0] = 1
    return sps.csc_matrix(psi_0)


def spin_op(operator, site, n, unity):
    """
    Generates Pauli spin operators & spin-plus/-minus operators
    :param operator: operator {'x','y','z','+','-'} (str)
    :param site: site index of operator (int)
    :param n: number of sites (int)
    :param unity: sets h-bar/2 to unity (bool)
    """
    array = identity
    ops_list = ['x', 'y', 'z', '+', '-']
    pauli_ops = [(1 / 2) * sx, (1 / 2) * sy, (1 / 2) * sz, plus, minus]

    if unity:
        pauli_ops = [sx, sy, sz, 2 * plus, 2 * minus]

    if site == (n - 1):
        array = pauli_ops[ops_list.index(operator)]
    for x in reversed(range(0, n - 1)):
        if x == site:
            array = sps.kron(array, pauli_ops[ops_list.index(operator)])
        else:
            array = sps.kron(array, identity)

    return array


def gen_pairs(n, auto, open_chain):
    """
    auto: whether needing autocorrelations, not for hamiltonian matrix
    open_chain: includes periodic boundary conditions of False
    """
    nn, autos = [], []
    for p in range(n - 1):
        if auto:
            autos.append((p, p))
        nn.append((p, p + 1))
    if auto:
        autos.append((n - 1, n - 1))
    if n > 2 and not open_chain:
        nn.append((0, n - 1))
    return nn, autos


def gen_m(leng, steps):
    # Generate an empty data matrix
    return sps.lil_matrix(np.zeros([leng, steps]), dtype=complex)


def commutes(a, b):
    # Test whether operators commute
    comp = a.dot(b) - b.dot(a)
    comp = comp.toarray().tolist()
    if np.count_nonzero(comp) == 0:
        return True
    else:
        return False


# ============================================== Helper functions for Qiskit Simulations ============================= >


def sort_counts(count, qs, shots):
    # sort counts and divide out shots
    vec = []
    for i in range(2 ** qs):
        binary = np.binary_repr(i).zfill(qs)
        if binary in count.keys():
            vec.append(count[binary] / shots)
        else:
            vec.append(0.0)
    return vec


def sort_counts_no_div(count, qs):
    # sort counts without dividing out shots
    vec = []
    for i in range(2 ** qs):
        binary = np.binary_repr(i).zfill(qs)
        if binary in count.keys():
            vec.append(count[binary])
        else:
            vec.append(0.0)
    return vec


def choose_control_gate(choice, qc, c, t):
    # Applying chosen controlled unitary for correlations and magnetization
    if choice == 'x':
        qc.cx(control_qubit=c, target_qubit=t)
    elif choice == 'y':
        qc.cy(control_qubit=c, target_qubit=t)
    elif choice == 'z':
        qc.cz(control_qubit=c, target_qubit=t)


def real_or_imag_measurement(qc, j):
    # For ancilla-assisted measurements
    if j == 0:
        qc.h(0)
    elif j == 1:
        qc.rx(-1 * math.pi / 2, 0)


def zz_operation(qc, a, b, delta):
    # Helper for time evolution operator
    qc.cx(a, b)
    qc.rz(2 * delta, b)
    qc.cx(a, b)


def three_cnot_evolution(qc, pair, ancilla, j, t, dt, trotter_steps, ising, a_constant):
    a_, b_ = pair[0] + ancilla, pair[1] + ancilla
    if ising:
        zz_operation(qc, a_, b_, j * t * dt  / trotter_steps)
    else:
        delta = j * t * dt / trotter_steps
        qc.cx(a_, b_)
        qc.rx(2 * delta - math.pi / 2, a_)
        qc.rz(2 * delta * a_constant, b_)
        qc.h(a_)
        qc.cx(a_, b_)
        qc.h(a_)
        qc.rz(- 2 * delta, b_)
        qc.cx(a_, b_)
        qc.rx(math.pi / 2, a_)
        qc.rx(-math.pi / 2, b_)


def execute_real(qc, device, shots):
    # run circuit on real hardware
    result = execute(qc, backend=device, shots=shots).result()
    return result


def run_circuit(anc, qc, noise, ibmq_params, n):
    [device, nm, bg, simulator, crosstalk_props, real_sim, dev_name, provider] = ibmq_params
    # real_sim: if True, run on the real backend, otherwise noise model.
    shots = setup.shots

    msrd = []
    if anc == 1:
        qc.measure(0, 0)
        msrd.append(0)
    else:
        for x in range(n):
            qc.measure(x, x)
            msrd.append(x)

    # Transpile circuit and use adaptive scheduling for the noise model TODO complete adaptive code
    circ = transpile(qc, backend=device, backend_properties=device.properties())
    dag = circuit_to_dag(circ)

    # weight factor must be tuned per application
    pass_ = cas.CrosstalkAdaptiveSchedule(device.properties(), crosstalk_props, provider,
                                          weight_factor=0.5, dev_name=dev_name)
    scheduled_dag = pass_.run(dag)
    scheduled_circ = dag_to_circuit(scheduled_dag)
    print(scheduled_circ)
    counts = []
    # Ideal simulation :
    if not noise:
        result = execute(qc, backend=simulator, shots=shots).result()
        counts += [result.get_counts(i) for i in range(len(result.results))]
    # Noise model or hardware:
    else:
        if real_sim:
            result = execute_real(scheduled_circ, device, shots)
            counts += [result.get_counts(i) for i in range(len(result.results))]
        else:
            result = execute(scheduled_circ, backend=simulator, shots=shots, noise_model=nm, basis_gates=bg).result()
            counts += [result.get_counts(i) for i in range(len(result.results))]

    if anc == 1:
        probs = sort_counts(counts[0], anc, shots)
        return probs[0] - probs[1]
    else:
        measurements = sort_counts(counts[0], n, shots)
        return measurements


def gen_even_odd_pairs(n, open_chain):
    # Used only for second-order trotter # TODO cant use, unknown error see if fixable
    nn_even = []
    nn_odd = []
    for p in range(n - 1):
        if p%2 == 0:
            nn_even.append((p, p + 1))
        else:
            nn_odd.append((p, p + 1))
    if n > 2 and not open_chain:
        if (n - 1) % 2 == 0:
            nn_even.append((0, n - 1))
        else:
            nn_odd.append((0, n - 1))
    return [nn_even, nn_odd]

# =============================== Writing / Reading Helpers ======================================================= >


def write_data(vec, loc):
    # write data from a 1d list into a csv file by rows
    with open(loc,'a',newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(vec)
    csvFile.close()


def write_numpy_array(array, loc):
    # write a numpy array into a text file
    np.savetxt(loc, array, delimiter=",")


def read_numpy_array(filename):
    # read numpy array from a textfile
    lines = np.loadtxt(filename, delimiter=",", unpack=False)
    return lines


def read_var_file(filename):
    data = []
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        data += list(lines)
    csvfile.close()
    return data