###########################################################################
# IBMQSetup.py
# Part of HeisenbergCodes
# Updated January '21
#
# Code to retrieve IBMQ account and initiate settings.
# Also store crosstalk properties
# Current Qiskit:
# qiskit 0.21.0
# qiskit-terra 0.15.2
# qiskit-aer 0.6.1
# qiskit-ibmq-provider 0.9.0
###########################################################################

# References for adaptive scheduling in Qiskit:
# https://qiskit.org/documentation/stubs/qiskit.transpiler.passes.CrosstalkAdaptiveSchedule.html
# https://qiskit.org/documentation/release_notes.html#terra-0-9
# https://mrmgroup.cs.princeton.edu/papers/pmurali-asplos20.pdf

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import noise
from qiskit import Aer, IBMQ
from qiskit import QuantumCircuit
from qiskit import execute
import numpy as np

# Sometimes, this command is not necessary (if console says no account in session).
# IBMQ.disable_account()

TOKEN = '29e5a75de595227f8e2477aa7abab6595dfafbc3375d7b510fa8be7c03b95c031e49887' \
        'af3da2d18efb30719038730cdf0ee806523bfdfbc4c231f0ad93f7e74'
provider = IBMQ.enable_account(TOKEN, hub='ibm-q', group='open', project='main')
simulator = Aer.get_backend('qasm_simulator')

shots = 50000

qubitDict = dict()
qubitDict['ibmq_rome'] = 5
qubitDict['ibmq_valencia'] = 5
qubitDict['ibmq_16_melbourne'] = 15
qubitDict['ibmq_armonk'] = 1
qubitDict['ibmq_ourense'] = 5
qubitDict['ibmq_qasm_simulator'] = 32
qubitDict['ibmq_vigo'] = 5
qubitDict['ibmqx2'] = 5
qubitDict['ibmq_bogota'] = 5
qubitDict['ibmq_santiago'] = 5
qubitDict['ibmq_athens'] = 5
qubitDict['ibmq_casablanca'] = 7

##################################################################
# Crosstalk Properties Dictionaries (made Jan 11)

# Santiago
crs_tlk_santiago = {(0, 1) : {(2, 3) : 0.09840000000000003, (2) : 0.09863999999999999},
                   (1, 2) : {(0, 3) : 0.09170000000000002, (3, 4) : 0.09546000000000002,
                            (0) : 0.10972000000000003, (3) : 0.10004},
                   (2, 3) : {(0, 1) : 0.09650000000000002, (1, 4) : 0.09903999999999998,
                            (1) : 0.13328, (4) : 0.10828000000000002},
                   (3, 4) : {(1, 2) : 0.09370000000000002, (2) : 0.09710000000000002}}


crs_talk_athens =  {(0, 1) : {(2, 3) : 0.06143999999999999, (2) : 0.08281999999999999},
                   (1, 2) : {(0, 3) : 0.06118, (3, 4) : 0.06172,
                            (0) : 0.07064, (3) : 0.07568},
                   (2, 3) : {(0, 1) : 0.05877999999999999, (1, 4) : 0.060899999999999996,
                            (1) : 0.06992, (4) : 0.06561999999999998},
                   (3, 4) : {(1, 2) : 0.061700000000000005, (2) : 0.08146}}


crs_tlk_casablanca = {}

crosstalk_props = {'ibmq_santiago': crs_tlk_santiago, 'ibmq_athens': crs_talk_athens}


def gen_pairs_crosstalk(n):
    # must be here to avoid circular import
    nn = []
    for p in range(n - 1):
        nn.append((p, p + 1))
    return nn


def sort_counts_crosstalk(count, qs):
    # must be here to avoid circular import
    vec = []
    for i in range(2 ** qs):
        binary = np.binary_repr(i).zfill(qs)
        if binary in count.keys():
            vec.append(count[binary] / shots)
        else:
            vec.append(0.0)
    return vec

# ==================================== Code for crosstalk measurements===============================================>
# Crosstalk rates are calculated using noise models, not the backend, since the data will use noise models


def crosstalk_circuits(cnot_pair, simultaneous_cnot, simultaneous_sing, device_params, n_qubits):
    # optimization level = 0 --> should correspond to hardware qubits
    [device, nm, bg, coupling_map] = device_params

    for gate2 in simultaneous_sing:
        print("------")
        qc = QuantumCircuit(n_qubits, n_qubits)
        qc.barrier()
        qc.cnot(cnot_pair[0], cnot_pair[1])
        qc.x(gate2)
        for x in range(n_qubits):
            qc.measure(x, x)
        result = execute(qc, backend=simulator, shots=shots, optimization_level=0,
                         basis_gates=bg, noise_model=nm).result()
        count = [result.get_counts(i) for i in range(len(result.results))]
        measurements = sort_counts_crosstalk(count[0], n_qubits)
        ideal_measurement = ('0'*n_qubits)[(gate2 + 1):] + '1' + ('0'*n_qubits)[:gate2]
        ideal_index = int(ideal_measurement, 2)
        cond_error_rate = sum(measurements[:ideal_index] + measurements[(ideal_index + 1):])
        print("Error rate for CNOT", cnot_pair, ", given single gate on ", gate2, " : ", cond_error_rate)

    for gate1 in simultaneous_cnot:
        print("------")
        qc = QuantumCircuit(n_qubits, n_qubits)
        qc.barrier()
        qc.cnot(cnot_pair[0], cnot_pair[1])
        qc.cnot(gate1[0], gate1[1])
        for x in range(n_qubits):
            qc.measure(x, x)
        result = execute(qc, backend=simulator, shots=shots, optimization_level=0, basis_gates=bg,
                         noise_model=nm).result()
        count = [result.get_counts(i) for i in range(len(result.results))]
        measurements = sort_counts_crosstalk(count[0], n_qubits)
        cond_error_rate = sum(measurements[1:])
        print("Error rate for CNOT", cnot_pair, ", given CNOT ", gate1, " : ", cond_error_rate)


# ==================================== General Setup class ============================================================>


class ibmqSetup():
    def __init__(self, sim=True, dev_name='', shots=shots):
        self.sim = sim
        self.shots = shots
        self.dev_name = dev_name

    def get_noise_model(self):
        # regular noise model from the backend

        device = provider.get_backend(self.dev_name)
        properties = device.properties()
        gate_lengths = noise.device.parameters.gate_length_values(properties)
        noise_model = NoiseModel.from_backend(properties, gate_lengths=gate_lengths)
        basis_gates = noise_model.basis_gates
        coupling_map = device.configuration().coupling_map
        crosstalk_properties = self.get_crosstalk_properties()
        return device, noise_model, basis_gates, coupling_map, crosstalk_props, self.dev_name, provider

    def get_device(self):
        return provider.get_backend(self.dev_name)

    def get_simulator(self):
        return simulator

    def get_n_qubits(self):
        return qubitDict[self.dev_name]

    def get_ct_noise_model(self):
        # obtain noise model for crosstalk measurements
        device = provider.get_backend(self.dev_name)
        properties = device.properties()
        gate_lengths = noise.device.parameters.gate_length_values(properties)
        noise_model = NoiseModel.from_backend(properties, gate_lengths=gate_lengths)
        basis_gates = noise_model.basis_gates
        coupling_map = device.configuration().coupling_map
        return [device, noise_model, basis_gates, coupling_map]

    def measure_crosstalk_properties(self):
        # implemented for the specific cases n = 5 , n = 7 (5 needs 12 circuits; 7 needs 22 circuits)
        # assumption is made: directionality of cnot is unimportant for the optimization

        n_qubits = self.get_n_qubits()
        # 1 pair for each cnot gate
        pairs = gen_pairs_crosstalk(n_qubits)

        device, noise_model, basis_gates, coupling_map = self.get_ct_noise_model()
        device_params = [device, noise_model, basis_gates, coupling_map]

        for pair in pairs:
            # generate list of nearby one-qubit gates or cnot gates to account for as indices
            simultaneous_cnot = []
            simultaneous_sing = []
            if pair[0] > 1:
                simultaneous_cnot.append((pair[0] - 2, pair[1] - 2))
            if (n_qubits - 1) - pair[1] > 1:
                simultaneous_cnot.append((pair[0] + 2, pair[1] + 2))
            if (pair[0] > 0) and (pair[1] < (n_qubits - 1)):
                simultaneous_cnot.append((pair[0] - 1, pair[1] + 1))
            if (n_qubits - 1) - pair[1] > 0:
                simultaneous_sing.append(pair[1] + 1)
            if pair[0] > 0:
                simultaneous_sing.append(pair[0] - 1)

            simultaneous_sing.sort()
            # Values are printed in console while running, and stored in code.
            crosstalk_circuits(pair, simultaneous_cnot, simultaneous_sing, device_params, n_qubits)
        return

    def get_crosstalk_properties(self):
        return crosstalk_props[self.dev_name]


# Code calls for setting up crosstalk circuits:
#set1 = ibmqSetup(dev_name='ibmq_santiago')
#set1.measure_crosstalk_properties()

#set2 = ibmqSetup(dev_name='ibmq_athens')
#set2.measure_crosstalk_properties()

