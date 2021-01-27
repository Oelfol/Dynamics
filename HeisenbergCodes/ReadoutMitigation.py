###########################################################################
# ReadoutMitigation.py
# Part of HeisenbergCodes
# Updated January '21
#
# Iterative Bayesian Unfolding for readout error mitigation
# Partially using PyUnfold package
# sources: https://github.com/jrbourbeau/pyunfold,
# https://www.theoj.org/joss-papers/joss.00741/10.21105.joss.00741.pdf
###########################################################################

from pyunfold import iterative_unfold
#from pyunfold.callbacks import Logger
from qiskit import QuantumCircuit, execute
import numpy as np
import math
import csv
import scipy.sparse as sps

# File storage format : Each row of csv file corresponds to all P(obtain J| true I), and num_shots is part of filename

import IBMQSetup as ibmq
device_name = 'ibmq_ourense'
setup = ibmq.ibmqSetup(dev_name=device_name)
device, nm, bg, coupling_map = setup.get_noise_model()
simulator = setup.get_simulator()

# =============================== Avoiding circular imports from hf ==================================================>
# The functions are identical


def write_data_temp(vec, loc):
    # write data from a 1d list into a csv file by rows
    with open(loc,'a',newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(vec)
    csvFile.close()


def write_numpy_array_temp(array, loc):
    # write a numpy array into a text file
    np.savetxt(loc, array, delimiter=",")

casablanca_array = np.array([[9.801999999999999602e-01, 3.685999999999999693e-02],
                          [1.980000000000000163e-02, 9.631399999999999961e-01]])
ourense_array = np.array([[9.930600000000000538e-01, 1.905000000000000096e-02],
                          [6.939999999999999988e-03, 9.809499999999999886e-01]])


def read_numpy_array_temp(filename):
    # read numpy array from a textfile
    # lines = np.loadtxt(filename, delimiter=",", unpack=False)
    # numpy is being insane again.
    # jerry fix:

    lines = np.array([[0, 0],[0, 0]])
    if filename == 'RMArrays/ourense_RM_Jan14_AncillaQubit2.txt':
        lines = lines + ourense_array
    elif filename == 'RMArrays/casablanca_RM_Jan14_AncillaQubit2.txt':
        lines = lines + casablanca_array

    return lines


def read_var_file_temp(filename):
    data = []
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        data += list(lines)
    csvfile.close()
    return data


def gen_m_temp(leng, steps):
    # Generate an empty data matrix -- > same as gen_m, circular import issues
    return sps.lil_matrix(np.zeros([leng, steps]), dtype=complex)


def sort_counts_temp(count, qs, shots):
    # sort counts and divide out shots
    vec = []
    for i in range(2 ** qs):
        binary = np.binary_repr(i).zfill(qs)
        if binary in count.keys():
            vec.append(count[binary] / shots)
        else:
            vec.append(0.0)
    return vec

# =================================Circuits to gather readout error data===============================================>


def init_state(num_qubits, qc, initial_state):
    # Initialize a circuit in the desired spin state.
    state_temp, index = np.binary_repr(initial_state).zfill(num_qubits)[::-1], 0
    for x in state_temp:
        if x == '1':
            qc.x(index)
        index += 1


def run_circuit(qc, shots, qubits):
    # run has optimization_level = 0 in order to have preserve chosen ancilla qubit.
    result = execute(qc, backend=simulator, shots=shots, noise_model=nm, basis_gates=bg, optimization_level=0).result()
    counts = [result.get_counts(i) for i in range(len(result.results))]
    countslist = sort_counts_temp(counts[0], qubits, shots)
    return countslist


def readout_mitigation_circuits_all(num_qubits, shots, filename):
    # find readout error probabilities for num_qubits (not ancilla circuits)
    # num_qubits is the number of qubits on the device, since virtual -- > hardware mapping will have to be handled
    # in SpinChains.py

    data = gen_m_temp(2 ** num_qubits, 2 ** num_qubits).toarray()
    for state in range(2 ** num_qubits):
        qc = QuantumCircuit(num_qubits, num_qubits)
        init_state(num_qubits, qc, state)
        for x in range(num_qubits):
            qc.measure(x, x)
        probs = run_circuit(qc, shots, num_qubits)
        data[:, state] += np.array(probs).T

    # write conditional probabilities to array and store
    write_numpy_array_temp(np.real(data), filename)


def readout_mitigation_circuits_ancilla(num_qubits, shots, filename, qubit=0):
    # num_qubits is the number of qubits in the device
    # qubit is the ancilla qubit used (based on coherence times -- has to be manually given)

    data = gen_m_temp(2, 2).toarray()

    # measure state 0
    qc0 = QuantumCircuit(num_qubits, 1)
    qc0.measure(qubit, 0)
    probs0 = run_circuit(qc0, shots, 1)
    data[:, 0] += np.array(probs0).T

    # measure state 1
    qc1 = QuantumCircuit(num_qubits, 1)
    qc1.x(qubit)
    qc1.measure(qubit, 0)
    probs1 = run_circuit(qc1, shots, 1)
    data[:, 1] += np.array(probs1).T

    # write conditional probabilities to array and store
    write_numpy_array_temp(np.real(data), filename)


# ================================= Calls for recording counts ======================================================>

# readout_mitigation_circuits_all(5, 50000, "ourense_RM_Jan14_AllQubits.txt")
# Used for everything apart from joel
# readout_mitigation_circuits_ancilla(5, 100000, "ourense_RM_Jan14_AncillaQubit2.txt", qubit=2)
# ancilla qubit 2 --- > hardware qubit # 2

# Used for Joel 6-site problem:
# readout_mitigation_circuits_ancilla(6, 50000, "casablanca_RM_Jan14_AncillaQubit2.txt", qubit=1)

# ================================== PyUnfold for Iterative bayesian unfolding =====================================>


def get_efficienties(measured):
    # return efficiencies, efficiencies_err
    return np.ones_like(measured), np.full_like(measured, 0, dtype=float)


def std_dev(measured, shots):
    mean, stdev = 0, 0
    for x in measured:
        mean += measured.index(x) * (x / shots)
    for x in range(len(measured)):
        stdev += ((x - mean)**2) * (1 / shots)
    return math.sqrt(stdev)


def get_measured_err(measured, shots):
    return np.ones_like(measured) * math.sqrt(1 / shots) * std_dev(measured, shots)


def get_response_matrix(rm_filename):
    r_matrix = read_numpy_array_temp(rm_filename)
    r_matrix_err = np.zeros_like(r_matrix)
    return r_matrix, r_matrix_err


def unfold(filename, shots, measured, num_qubits):
    # rm_filename: txt file storing the response matrix
    # measured data: for a particular circuit (not normalized)
    # shots: in the measured data
    measured_err = get_measured_err(measured, shots)
    r_matrix, r_matrix_err = get_response_matrix(filename)
    efficiencies, efficiencies_err = get_efficienties(measured)
    unfolded_results = iterative_unfold(data=measured, data_err=measured_err, response=r_matrix,
                                        response_err=r_matrix_err, efficiencies=efficiencies,
                                        efficiencies_err=efficiencies_err, ts='rmd', ts_stopping=0.001)
                                        #callbacks=[Logger()])

    temp = unfolded_results['unfolded'] / shots

    r_matrix2 = read_numpy_array_temp(filename)

    num_states = 2 ** num_qubits
    t_vector = [1 / num_states] * num_states

    # Arbitrary number of iterations chosen, to be manally optimized since this should be simpler
    for iter in range(3):
        sum_j = 0
        for j in range(num_states):
            for k in range(num_states):
                sum_j += r_matrix2[j, k] * t_vector[k]
        for i in range(num_states):
            sum_j2 = 0
            for j in range(num_states):
                sum_j2 += (r_matrix2[j, i] * t_vector[i] / sum_j) * measured[j]
            t_vector[i] = sum_j2
    return (((np.array(t_vector) / shots) + temp) * (1 / 2)).tolist()
