
# Code to implement IBU method for readout error mitigation, using PyUnfold
# Combines pyunfold with own attempted code, for better general agreement. 
# source: https://github.com/jrbourbeau/pyunfold , https://www.theoj.org/joss-papers/joss.00741/10.21105.joss.00741.pdf
# File storage format : Each row of csv file corresponds to all P(obtain J| true I), and num_shots is part of filename
# October 2020


from pyunfold import iterative_unfold
from pyunfold.callbacks import Logger
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import noise
from qiskit import QuantumCircuit, Aer, execute, IBMQ
import numpy as np
import math
import spinchainshelpers as sch

# ========================================= IBM account and noise model setup ======================================== >

provider = IBMQ.enable_account('29e5a75de595227f8e2477aa7abab6595dfafbc3375d7b510fa8be7c03b95c031e49887af3da2d18efb30719038730cdf0ee806523bfdfbc4c231f0ad93f7e74',
                               hub='ibm-q', group='open', project='main')
simulator = Aer.get_backend('qasm_simulator')


def qc_noise_model(dev_name):
    # regular noise model from the backend

    device = provider.get_backend(dev_name)
    properties = device.properties()
    gate_lengths = noise.device.parameters.gate_length_values(properties)
    noise_model = NoiseModel.from_backend(properties, gate_lengths=gate_lengths)
    basis_gates = noise_model.basis_gates
    coupling_map = device.configuration().coupling_map
    return device, noise_model, basis_gates, coupling_map


# ibmq_athens, ibmq_rome, ibmq_bogota
dev_name = 'ibmq_santiago'
device, nm, bg, cm = qc_noise_model(dev_name)

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
    countslist = sch.sort_counts(counts[0], qubits, shots)
    return countslist


def readout_mitigation_circuits_all(num_qubits, shots, filename):
    # find readout error probabilities for num_qubits (not ancilla circuits)
    # num_qubits is the number of qubits on the device, since virtual -- > hardware mapping will have to be handled
    # in SpinChains.py

    data = sch.gen_m(2 ** num_qubits, 2 ** num_qubits).toarray()
    for state in range(2 ** num_qubits):
        qc = QuantumCircuit(num_qubits, num_qubits)
        init_state(num_qubits, qc, state)
        for x in range(num_qubits):
            qc.measure(x, x)
        probs = run_circuit(qc, shots, num_qubits)
        data[:, state] += np.array(probs).T

    # write conditional probabilities to array and store
    sch.write_numpy_array(np.real(data), filename)


def readout_mitigation_circuits_ancilla(num_qubits, shots, filename, qubit=0):
    # num_qubits is the number of qubits in the device
    # qubit is the ancilla qubit used (based on coherence times -- has to be manually given)

    data = sch.gen_m(2, 2).toarray()

    # measure state 0
    qc0 = QuantumCircuit(num_qubits, 1)
    qc0.measure(qubit, 0)
    probs0 = run_circuit(qc0, shots, 1)
    data[:, 0] += np.array(probs0).T

    #measure state 1
    qc1 = QuantumCircuit(num_qubits, 1)
    qc1.x(qubit)
    qc1.measure(qubit, 0)
    probs1 = run_circuit(qc1, shots, 1)
    data[:, 1] += np.array(probs1).T

    # write conditional probabilities to array and store
    sch.write_numpy_array(np.real(data), filename)


# ================================= Calls for recording counts ======================================================>


#readout_mitigation_circuits_all(5, 50000, "santiago_RM_Oct20_AllQubits.txt")
# Used for Tachinno 5a
#readout_mitigation_circuits_ancilla(5, 100000, "santiago_RM_Oct20_AncillaQubit2.txt", qubit=2)
# ancilla qubit 2 --- > hardware qubit # 2

# Used for Joel 6-site problem:
#readout_mitigation_circuits_ancilla(6, 50000, "santiago_RM_Oct20_AncillaQubit2.txt", qubit=2)
# consider making this a 5-site problem


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
    r_matrix = sch.read_numpy_array(rm_filename)
    r_matrix_err = np.zeros_like(r_matrix)
    return r_matrix, r_matrix_err


def unfold(filename, shots, measured, num_qubits):
    # rm_filename: txt file storing the response matrix
    # measured data: for a particular circuit (not normalized)
    # shots: in the measured data

    measured_err = get_measured_err(measured, shots)
    r_matrix, r_matrix_err = get_response_matrix(filename)
    efficiencies, efficiencies_err = get_efficienties(measured)
    print('____')
    unfolded_results = iterative_unfold(data=measured, data_err=measured_err, response=r_matrix,
                                        response_err=r_matrix_err, efficiencies=efficiencies,
                                        efficiencies_err=efficiencies_err, ts='rmd', ts_stopping=0.001,
                                        callbacks=[Logger()])

    temp = unfolded_results['unfolded'] / shots

    # Separate method, combined as average :: 
    r_matrix2 = sch.read_numpy_array(filename)

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
