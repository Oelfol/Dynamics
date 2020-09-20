# Extra Qiskit Functions

"""
Code to keep the qiskit functions which do not fit in with the general QuantumSim/ClassicalSim class
(correlation functions for '+-') and the Francis implementation of Francis Fig. 5a

"""

import numpy as np
import math
import matplotlib
import warnings
import matplotlib.pyplot as plt
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import noise
from qiskit import QuantumCircuit, Aer, execute
import scipy.sparse as sps
from cycler import cycler

warnings.filterwarnings('ignore')
# ========================================= IBM account and noise model setup ======================================== >

matplotlib.rcParams['figure.figsize'] = [10, 10]
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

dev_name = 'ibmq_santiago'
device, nm, bg, cm = qc_noise_model(dev_name) # noise model, basis gates, coupling map

colors = ['g', 'k', 'maroon', 'mediumblue', 'slateblue', 'limegreen', 'b', 'r', 'olive']
custom_cycler = (cycler(color=colors) + cycler(lw=[2] * len(colors)))

def set_up_axes(num_axes):
    if num_axes == 1:
        fig, ax = plt.subplots()
        ax.set_prop_cycle(custom_cycler)
        ax.margins(x=0)
        return fig, ax

    elif num_axes == 2:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.set_prop_cycle(custom_cycler)
        ax2.set_prop_cycle(custom_cycler)
        ax1.margins(x=0)
        ax2.margins(x=0)
        fig.subplots_adjust(hspace=0)
        return fig, (ax1, ax2)

    elif num_axes == 3:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.set_prop_cycle(custom_cycler)
        ax2.set_prop_cycle(custom_cycler)
        ax3.set_prop_cycle(custom_cycler)
        ax1.margins(x=0)
        ax2.margins(x=0)
        ax3.margins(x=0)
        fig.subplots_adjust(hspace=0)
        return fig, (ax1, ax2, ax3)


def plot_dataset_byrows(ax, legendtitle, ylabel, xlabel):
    ax.legend(loc='right', title=legendtitle)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)


def two_point_correlations_plotter(alpha, beta, j, dt, chosen_pairs, data_real, data_imag, data_gates):
    data_real_id, data_real_noise = data_real[0], data_real[1]
    data_imag_id, data_imag_noise = data_imag[0], data_imag[1]

    re_label = r'$Re {\langle S_{' + alpha + '}^{i}(t) $' + r'$S_{' + beta + '}^{j}(0) $' + r'$\rangle $'
    im_label = r'$Im {\langle S_{' + alpha + '}^{i}(t) $' + r'$S_{' + beta + '}^{j}(0) $' + r'$\rangle $'
    fig, (ax1, ax2, ax3) = set_up_axes(3)
    scaler = j * dt
    for x in chosen_pairs:
        x1 = [i * scaler for i in range(len(data_real_id.toarray()[chosen_pairs.index(x)][:].tolist()))]
        ax1.plot(x1, data_real_id.toarray()[chosen_pairs.index(x)][:].tolist(), label=str(x))
        ax1.plot(x1, data_real_noise.toarray()[chosen_pairs.index(x)][:].tolist(), label=str(x), linestyle=":")

        x2 = [i * scaler for i in range(len(data_imag_id.toarray()[chosen_pairs.index(x)][:].tolist()))]
        ax2.plot(x2, data_imag_id.toarray()[chosen_pairs.index(x)][:].tolist(), label=str(x))
        ax2.plot(x2, data_imag_noise.toarray()[chosen_pairs.index(x)][:].tolist(), label=str(x), linestyle=":")

    x3 = [i * scaler for i in range(len(data_imag_id.toarray()[chosen_pairs.index(x)][:].tolist()))]
    ax3.plot(x3, data_gates.toarray()[0, :].tolist())
    ax3.set_ylabel("Gates", fontsize='medium')
    plot_dataset_byrows(ax1, 'Site Pairs', re_label, 'Jt')
    plot_dataset_byrows(ax2, 'Site Pairs', im_label, 'Jt')
    fig.suptitle('Two-Point Correlations', fontsize=16)
    fig.tight_layout(pad=2.0)
    plt.show()


def gen_m(leng, steps):
    # Generate an empty data matrix
    return sps.lil_matrix(np.zeros([leng, steps]), dtype=complex)


def sort_counts(count, qs, shots):
    vec = []
    for i in range(2 ** qs):
        binary = np.binary_repr(i).zfill(qs)
        if binary in count.keys():
            vec.append(count[binary] / shots)
        else:
            vec.append(0.0)
    return vec


def time_ev_op_decomp(qc, a, b, j, t):
    qc.cx(control_qubit=a, target_qubit=b)
    qc.rx(-2 * j * t - (math.pi / 2), a)
    qc.rz(-2 * j * t, b)
    qc.h(a)
    qc.cx(control_qubit=a, target_qubit=b)
    qc.h(a)
    qc.rz(2 * j * t, b)
    qc.cx(control_qubit=a, target_qubit=b)
    qc.rx(math.pi / 2, a)
    qc.rx(-1 * math.pi / 2, b)


def four_site_ibmq(total_time, dt, j):
    # For Francis Fig 5a

    pairs, counting = [(1, 1), (2, 1), (3, 1)], 0
    data_real_id, data_imag_id = gen_m(len(pairs), total_time), gen_m(len(pairs), total_time)
    data_real_noise, data_imag_noise = gen_m(len(pairs), total_time), gen_m(len(pairs), total_time)

    for pair in pairs:
        for t in range(total_time):
            for d in range(2):
                counting += 1
                print(counting)
                qc_id, qc_noise = QuantumCircuit(5, 1), QuantumCircuit(5, 1)

                qc_id.h(0)
                qc_noise.h(0)

                qc_id.cx(control_qubit=0, target_qubit=pair[1])
                qc_noise.cx(control_qubit=0, target_qubit=pair[1])

                time_ev_op_decomp(qc_id, 1, 2, j, t * dt)
                time_ev_op_decomp(qc_id, 3, 4, j, t * dt)
                qc_id.swap(2, 3)
                time_ev_op_decomp(qc_id, 4, 1, j, t * dt)
                time_ev_op_decomp(qc_id, 2, 3, j, t * dt)
                qc_id.swap(2, 3)
                qc_id.cx(control_qubit=0, target_qubit=pair[0])

                time_ev_op_decomp(qc_noise, 1, 2, j, t * dt)
                time_ev_op_decomp(qc_noise, 3, 4, j, t * dt)
                qc_noise.swap(2, 3)
                time_ev_op_decomp(qc_noise, 4, 1, j, t * dt)
                time_ev_op_decomp(qc_noise, 2, 3, j, t * dt)
                qc_noise.swap(2, 3)
                qc_noise.cx(control_qubit=0, target_qubit=pair[0])

                if d == 0:
                    qc_id.h(0)
                    qc_noise.h(0)
                elif d == 1:
                    qc_id.rx(-1 * math.pi / 2, 0)
                    qc_noise.rx(-1 * math.pi / 2, 0)

                qc_id.measure(0, 0)
                qc_noise.measure(0, 0)
                result_id = execute(qc_id, backend=simulator, shots=50000).result()
                result_noise = execute(qc_noise, backend=simulator, shots=50000, noise_model=nm,
                                       basis_gates=bg, optimization_level=0).result()
                counts_id = [result_id.get_counts(i) for i in range(len(result_id.results))]
                counts_noise = [result_noise.get_counts(i) for i in range(len(result_noise.results))]
                sorted_counts_id = sort_counts(counts_id[0], 1, 50000)
                sorted_counts_noise = sort_counts(counts_noise[0], 1, 50000)
                measurement_id = sorted_counts_id[0] - sorted_counts_id[1]
                measurement_noise = sorted_counts_noise[0] - sorted_counts_noise[1]

                if d == 0:
                    data_real_id[pairs.index(pair), t] += measurement_id
                    data_real_noise[pairs.index(pair), t] += measurement_noise
                elif d == 1:
                    data_imag_id[pairs.index(pair), t] += measurement_id
                    data_imag_noise[pairs.index(pair), t] += measurement_noise

    data_real, data_imag = [data_real_id, data_real_noise], [data_imag_id, data_imag_noise]
    two_point_correlations_plotter('x', 'x', j, dt, pairs, data_real, data_imag)

def initialize_circuit():
    qc = QuantumCircuit(3, 1)
    qc.h(0)  
    qc.x(2)  
    return qc


def run_circuit(qc, j, noise):  
    measurement = 0

    if noise:
        result = execute(qc, backend=simulator, shots=50000).result()
        counts = [result.get_counts(i) for i in range(len(result.results))]
        probs = sort_counts(counts[0], 1, 50000)
        measurement = probs[0] * 2 - 1
    else:
        result = execute(qc, backend=simulator, shots=50000, noise_model=nm,
                                       basis_gates=bg, optimization_level=0).result()
        counts = [result.get_counts(i) for i in range(len(result.results))]
        probs = sort_counts(counts[0], 1, 50000)
        measurement = probs[0] * 2 - 1

    if j == 0:
        return measurement
    else:
        return 1j * measurement


def real_or_imag_measurement(qc, j):
    # For ancilla-assisted measurements

    if j == 0:
        qc.h(0)
    elif j == 1:
        qc.rx(-1 * math.pi / 2, 0)


def two_site_ibmq_noise(time_total, dt, j):
    # equal-time correlator '+-' --- corresponds to (0,0), (0, 1), (1,1)

    pairs = [(1, 1), (1, 2), (2, 2)]
    data = gen_m(len(pairs), time_total)

    for pair in pairs:
        for t in range(time_total):
            result = 0
            for i in range(2):
                temp_result = 0

                qc = initialize_circuit()
                time_ev_op_decomp(qc, 1, 2, j, t * dt)
                qc.cx(control_qubit=0, target_qubit=pair[1])
                qc.cx(control_qubit=0, target_qubit=pair[0])
                real_or_imag_measurement(qc, i)
                qc.measure(0, 0)
                temp_result += run_circuit(qc, i, False)

                qc = initialize_circuit()
                time_ev_op_decomp(qc, 1, 2, j, t * dt)
                qc.cy(control_qubit=0, target_qubit=pair[1])
                qc.cx(control_qubit=0, target_qubit=pair[0])
                real_or_imag_measurement(qc, i)
                qc.measure(0, 0)
                temp_result += -1j * run_circuit(qc, i, False)

                qc = initialize_circuit()
                time_ev_op_decomp(qc, 1, 2, j, t * dt)
                qc.cx(control_qubit=0, target_qubit=pair[1])
                qc.cy(control_qubit=0, target_qubit=pair[0])
                real_or_imag_measurement(qc, i)
                qc.measure(0, 0)
                temp_result += 1j * run_circuit(qc, i, False)

                qc = initialize_circuit()
                time_ev_op_decomp(qc, 1, 2, j, t * dt)
                qc.cy(control_qubit=0, target_qubit=pair[1])
                qc.cy(control_qubit=0, target_qubit=pair[0])
                real_or_imag_measurement(qc, i)
                qc.measure(0, 0)
                temp_result += run_circuit(qc, i, False)

                result += temp_result / 4  # replace pseudospin factor

                print("running code")

            result = np.conj(result) * result
            data[pairs.index(pair), t] += result
    return data


def two_site_ibmq_id(time_total, dt, j):
    # equal-time correlator '+-'
    # corresponds to (0,0), (0, 1), (1,1)

    pairs = [(1, 1), (1, 2), (2, 2)]
    data = gen_m(len(pairs), time_total)

    for pair in pairs:
        for t in range(time_total):
            result = 0
            for i in range(2):
                temp_result = 0

                qc = initialize_circuit()
                time_ev_op_decomp(qc, 1, 2, j, t * dt)
                qc.cx(control_qubit=0, target_qubit=pair[1])
                qc.cx(control_qubit=0, target_qubit=pair[0])
                real_or_imag_measurement(qc, i)
                qc.measure(0, 0)
                temp_result += run_circuit(qc, i, True)

                qc = initialize_circuit()
                time_ev_op_decomp(qc, 1, 2, j, t * dt)
                qc.cy(control_qubit=0, target_qubit=pair[1])
                qc.cx(control_qubit=0, target_qubit=pair[0])
                real_or_imag_measurement(qc, i)
                qc.measure(0, 0)
                temp_result += -1j * run_circuit(qc, i, True)

                qc = initialize_circuit()
                time_ev_op_decomp(qc, 1, 2, j, t * dt)
                qc.cx(control_qubit=0, target_qubit=pair[1])
                qc.cy(control_qubit=0, target_qubit=pair[0])
                real_or_imag_measurement(qc, i)
                qc.measure(0, 0)
                temp_result += 1j * run_circuit(qc, i, True)

                qc = initialize_circuit()
                time_ev_op_decomp(qc, 1, 2, j, t * dt)
                qc.cy(control_qubit=0, target_qubit=pair[1])
                qc.cy(control_qubit=0, target_qubit=pair[0])
                real_or_imag_measurement(qc, i)
                qc.measure(0, 0)
                temp_result += run_circuit(qc, i, True)

                result += temp_result / 4  # replace pseudospin factor

                print("running code")

            result = np.conj(result) * result
            data[pairs.index(pair), t] += result
    return data

id_data = two_site_ibmq_id(100, 1e-2, 1)
noise_data = two_site_ibmq_noise(100, 1e-2, 1)

fig, ax = set_up_axes(1)
pairs_temp = [(1, 1), (1, 2), (2, 2)]
for x in pairs_temp:
    ax.plot(id_data.toarray()[pairs_temp.index(x)][:], label=str(x), linewidth=2, linestyle="-")
    ax.plot(noise_data.toarray()[pairs_temp.index(x)][:], label=str(x), linewidth=2, linestyle=":")
plot_dataset_byrows(ax, 'Site Pairs', '<S_i(+)S_j(-)>', 'Time')
plt.show()



# FRANCIS FIG 5a (Two Point Correlations)

# four_site_ibmq(600, .01, 0.84746)
