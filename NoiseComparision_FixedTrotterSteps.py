# NoiseComparison_FixedTrotterSteps

"""
August '20
Code to simulate dynamics of Heisenberg/Ising spin chains
With fixed trotter steps, comparing ideal qiskit aer simulation to basic device noise model
"""

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import noise

import numpy as np
import qiskit as q
import qiskit.extensions.unitary as qeu
from qiskit.quantum_info.operators import Operator
from cycler import cycler
import cProfile
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer, execute, IBMQ
import random
import itertools
import math
import matplotlib
import time
import warnings
import scipy.linalg, numpy.linalg
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.sparse.linalg as sla
import scipy.linalg as sl

warnings.filterwarnings('ignore')

# ========================================= IBM account and noise model setup ======================================== >

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


dev_name = 'ibmq_16_melbourne'
device, noise_model, basis_gates, coupling_map = qc_noise_model(dev_name)

# ==================================== Plotting Helpers ============================================================ >

colors = ['g', 'k', 'maroon', 'mediumblue', 'slateblue', 'limegreen', 'b', 'r', 'olive']
custom_cycler = (cycler(color=colors) + cycler(lw=[2] * len(colors)))

SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 25
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)


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


def occ_plotter(chosen_states, j, data, n, total_time, dt):
    data_id, data_noise = data[0], data[1]

    fig, ax = set_up_axes(1)
    for state in chosen_states:
        x1 = [i * abs(j) * dt for i in range(total_time)]
        dex = chosen_states.index(state)
        label = np.binary_repr(state).zfill(n)
        ax.plot(x1, data_id[dex, :].toarray().tolist()[0], label=label, linewidth=2, linestyle="-")
        ax.plot(x1, data_noise[dex, :].toarray().tolist()[0], label=label, linewidth=2, linestyle=":")
    plot_dataset_byrows(ax, 'States', 'Probability', r'$\it{Jt}$')
    plt.show()


# still need to fix these latex labels


def two_point_correlations_plotter(alpha, beta, j, dt, chosen_pairs, data_real, data_imag):
    data_real_id, data_real_noise = data_real[0], data_real[1]
    data_imag_id, data_imag_noise = data_imag[0], data_imag[1]

    re_label = r'$Re {\langle S_{' + alpha + '}}(t) $' + r'$S_{' + beta + '}(0) $' + r'$\rangle $'
    im_label = r'$Im {\langle S_{' + alpha + '}}(t) $' + r'$S_{' + beta + '}(0)$' + r'$\rangle $'
    fig, (ax1, ax2) = set_up_axes(2)
    scaler = j * dt
    for x in chosen_pairs:
        x1 = [i * scaler for i in range(len(data_real_id.toarray()[chosen_pairs.index(x)][:].tolist()))]
        ax1.plot(x1, data_real_id.toarray()[chosen_pairs.index(x)][:].tolist(), label=str(x))
        ax1.plot(x1, data_real_noise.toarray()[chosen_pairs.index(x)][:].tolist(), label=str(x), linestyle=":")

        x2 = [i * scaler for i in range(len(data_imag_id.toarray()[chosen_pairs.index(x)][:].tolist()))]
        ax2.plot(x2, data_imag_id.toarray()[chosen_pairs.index(x)][:].tolist(), label=str(x))
        ax2.plot(x2, data_imag_noise.toarray()[chosen_pairs.index(x)][:].tolist(), label=str(x), linestyle=":")

    plot_dataset_byrows(ax1, 'Site Pairs', re_label, 'Jt')
    plot_dataset_byrows(ax2, 'Site Pairs', im_label, 'Jt')
    plt.show()


def total_magnetization_plotter(j, total_time, dt, data_id, data_noise):
    fig, ax = set_up_axes(1)
    x1 = [i * j * dt for i in range(total_time)]
    ax.plot(x1, data_id.toarray()[0][:].tolist(), linewidth=2, linestyle='-')
    ax.plot(x1, data_noise.toarray()[0][:].tolist(), linewidth=2, linestyle=':')
    plt.xlabel(r'$\it{Jt}$')
    plt.ylabel('Total Magnetization')
    plt.show()


def all_site_magnetization_plotter(n, j, dt, total_time, data_id, data_noise):
    fig, ax = set_up_axes(1)
    for site in range(n):
        x1 = [i * j * dt for i in range(total_time)]
        ax.plot(x1, data_id.toarray()[site][:].tolist(), linewidth=2, label=site, linestyle='-')
        ax.plot(x1, data_noise.toarray()[site][:].tolist(), linewidth=2, label=site, linestyle=':')
    plot_dataset_byrows(ax, 'Sites', 'Magnetization', r'$\it{Jt}$')
    plt.show()


# ========================================= Pauli matrices and pseudospin operators ================================== >

sx = sps.csc_matrix(np.array([[0, 1], [1, 0]]))
sy = sps.csc_matrix(np.complex(0, 1) * np.array([[0, -1], [1, 0]]))
sz = sps.csc_matrix(np.array([[1, 0], [0, -1]]))
identity = sps.csc_matrix(np.eye(2, dtype=complex))
plus = sps.csc_matrix(sx * (1 / 2) + np.complex(0, 1) * sy * (1 / 2))
minus = sps.csc_matrix(sx * (1 / 2) - np.complex(0, 1) * sy * (1 / 2))


# ======================================= Helper functions for hamiltonian matrix ==================================== >

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


def gen_pairs(n, choice, auto, open_chain):
    """
    choice: nearest or next-nearest exchange 'NN' or 'NNN'
    auto: whether needing autocorrelations, not for hamiltonian matrix
    open_chain: includes periodic boundary conditions of False
    """
    nn, nnn, autos = [], [], []
    for p in range(n - 1):
        if auto:
            autos.append((p, p))
        nn.append((p, p + 1))
    if auto:
        autos.append((n - 1, n - 1))
    if n > 2 and not open_chain:
        nn.append((0, n - 1))
    if choice:
        if n > 3:
            for f in range(n - 2):
                nnn.append((f, f + 2))
        if n > 4 and not open_chain:
            nnn.append((0, n - 2))
            nnn.append((1, n - 1))

    return nn, nnn, autos


def gen_m(leng, steps):
    # Generate an empty data matrix
    return sps.lil_matrix(np.zeros([leng, steps]), dtype=complex)


def plot_dataset_byrows(ax, legendtitle, ylabel, xlabel):
    ax.legend(loc='right', title=legendtitle, fontsize='x-large')  # need to make legend optional
    ax.set_ylabel(ylabel, fontsize='x-large')
    ax.set_xlabel(xlabel, fontsize='x-large')


class ClassicalSpinChain:

    def __init__(self, j=0.0, j2=0.0, bg=0.0, a=0.0, n=0, open_chain=False, unity=False, choice=False, ising=False,
                 paper='', transverse=False):

        self.j = j  # coupling constant
        self.j2 = j2  # how much to scale next-nearest coupling
        self.bg = bg  # magnetic field strength
        self.a = a  # anisotropy constant jz/j (xxy model, otherwise ignore)
        self.n = n  # sites
        self.open_chain = open_chain  # whether open chain
        self.states = 2 ** n  # number of basis states
        self.unity = unity  # whether h-bar/2 == 1 (h-bar == 1 elsewise)
        self.choice = choice  # whether next-nearest, bool
        self.ising = ising  # for ising model
        self.paper = paper  # refer to settings for a specific paper
        self.transverse = transverse  # whether transverse field ising model

        self.hamiltonian = sps.lil_matrix(np.zeros([2 ** n, 2 ** n], complex))
        neighbors, nn_neighbors, autos = gen_pairs(self.n, self.choice, False, self.open_chain)
        multipliers = [1, 1, self.a]

        ops = ['x', 'y', 'z']
        if ising:
            ops = ['z']

        for y in neighbors:
            for op in ops:
                dex = ops.index(op)
                s1 = spin_op(op, y[0], self.n, self.unity)
                s2 = spin_op(op, y[1], self.n, self.unity)
                self.hamiltonian += s1.dot(s2) * self.j * multipliers[dex]
            for z in nn_neighbors:
                for op in ops:
                    dex = ops.index(op)
                    s1 = spin_op(op, z[0], self.n, self.unity)
                    s2 = spin_op(op, z[1], self.n, self.unity)
                    self.hamiltonian += s1.dot(s2) * self.j * multipliers[dex] * self.j2
        for x in range(self.n):
            if self.bg != 0 and self.transverse == False:
                s_final = spin_op('z', x, self.n, self.unity) * self.bg / 2
                self.hamiltonian += s_final
            elif self.bg != 0 and self.transverse == True:
                s_final = spin_op('x', x, self.n, self.unity) * self.bg / 2
                self.hamiltonian += s_final

        self.hamiltonian = sps.csc_matrix(self.hamiltonian)


# ============================================== Helper functions for Quantum Sim ==================================== >


def commutes(a, b):
    # Test whether the hamiltonian commutes with itself

    comm = a.dot(b) - b.dot(a)
    comp = comm.toarray().tolist()
    if max(comp) == 0:
        return True
    else:
        return False


def sort_counts(count, qs, shots):
    vec = []
    for i in range(2 ** qs):
        binary = np.binary_repr(i).zfill(qs)  # [::-1]
        if binary in count.keys():
            vec.append(count[binary] / shots)
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


def xx_operation(qc, a, b, delta):
    # Helper for time evolution operator

    qc.ry(math.pi / 2, a)
    qc.ry(math.pi / 2, b)
    qc.cx(a, b)
    qc.rz(2 * delta, b)
    qc.cx(a, b)
    qc.ry(-1 * math.pi / 2, a)
    qc.ry(-1 * math.pi / 2, b)


def yy_operation(qc, a, b, delta):
    # Helper for time evolution operator

    qc.rx(math.pi / 2, a)
    qc.rx(math.pi / 2, b)
    qc.cx(a, b)
    qc.rz(2 * delta, b)
    qc.cx(a, b)
    qc.rx(-1 * math.pi / 2, a)
    qc.rx(-1 * math.pi / 2, b)


def xyz_operation(qc, a, b, ancilla, j, t, dt, j2, trotter_steps, constant, ising, a_constant):
    # Time evolving the system termwise by the elements in the hamiltonian

    nnn_term = j2
    if abs(a - b) == 1:
        nnn_term = 1

    if not ising:
        xx_operation(qc, a + ancilla, b + ancilla, j * t * dt * nnn_term / (constant * trotter_steps))
        yy_operation(qc, a + ancilla, b + ancilla, j * t * dt * nnn_term / (constant * trotter_steps))
    zz_operation(qc, a + ancilla, b + ancilla, j * a_constant * t * dt * nnn_term / (constant * trotter_steps))


# =================================================== Quantum Simulation ============================================= >

class QuantumSim:

    def __init__(self, j=0.0, bg=0.0, a=1.0, n=0, j2=1, choice=True, open_chain=True, transverse=False, paper='',
                 ising=False, ts=5):

        self.j = j  # coupling constant
        self.bg = bg  # magnetic field strength
        self.n = n  # number of spins
        self.states = 2 ** n  # number of basis states
        self.choice = choice  # whether to have next-nearest neighbors
        self.a = a  # anisotropy jz / j
        self.j2 = j2  # how much to scale next-nearest coupling
        self.open_chain = open_chain  # open chain = periodic boundary
        self.transverse = transverse  # for transverse field ising model
        self.paper = paper  # direct reference for all the unique aspects of different papers
        self.ising = ising  # Whether using the ising model
        self.unity = False  # Depends on paper
        self.trotter_steps = ts
        if self.paper == 'francis':
            self.unity = True

        self.classical_hamiltonian = ClassicalSpinChain(j=self.j, j2=self.j2,
                                                        bg=self.bg, a=self.a, n=self.n, open_chain=self.open_chain,
                                                        unity=self.unity, choice=self.choice).hamiltonian
        self.h_commutes = commutes(self.classical_hamiltonian, self.classical_hamiltonian)
        self.pairs_nn, self.pairs_nnn, autos = gen_pairs(self.n, self.choice, False, self.open_chain)
        self.total_pairs = self.pairs_nn + self.pairs_nnn

    def init_state(self, qc, ancilla, initial_state):
        # Initialize a circuit in the desired spin state
        # Add a qubit if there is an ancilla measurement

        state_temp = np.binary_repr(initial_state).zfill(self.n)[::-1]
        anc = int(ancilla)
        index = 0
        for x in state_temp:
            if x == '1':
                qc.x(index + anc)
            index += 1

    def first_order_trotter(self, qc, dt, t, ancilla):

        trotter_steps = 1
        if self.h_commutes:
            print("H commutes, trotter_steps = 1")
        else:
            trotter_steps=self.trotter_steps

        if t == 0:
            print("First order trotter in progress..")

        # Address needed constants for particular paper
        pseudo_constant_a = 1.0
        mag_constant = 1.0
        if self.paper == 'joel':
            pseudo_constant_a = 4.0
            mag_constant = 2.0

        for step in range(trotter_steps):
            for k in range(self.n):
                if self.bg != 0.0:
                    if self.transverse:
                        qc.rx(self.bg * dt * t / (trotter_steps * mag_constant), k + ancilla)
                    else:
                        qc.rz(self.bg * dt * t / (trotter_steps * mag_constant), k + ancilla)
            for x in self.total_pairs:
                xyz_operation(qc, x[0], x[1], ancilla, self.j, t, dt, self.j2, trotter_steps * pseudo_constant_a, 1.0,
                              self.ising, self.a)

    def second_order_trotter(self, qc, dt, t, ancilla):
        trotter_steps = 1
        if self.h_commutes:
            print("H commutes, trotter_steps = 1")
        else:
            trotter_steps = self.trotter_steps

        if t == 0:
            print("Second order trotter in progress..")

        # Address needed constants for particular paper
        pseudo_constant_a = 1.0
        mag_constant = 1.0
        if self.paper in ['joel']:
            pseudo_constant_a = 4.0
            mag_constant = 2.0

        for k in range(self.n):
            if self.bg != 0.0:
                if self.transverse:
                    qc.rx(self.bg * dt * t / (trotter_steps * mag_constant), k + ancilla)
                else:
                    qc.rz(self.bg * dt * t / (trotter_steps * mag_constant), k + ancilla)

        for step in range(trotter_steps):
            for x in reversed(self.total_pairs[1:]):
                xyz_operation(qc, x[0], x[1], ancilla, self.j, t, dt, self.j2, trotter_steps * pseudo_constant_a, 2.0,
                              self.ising, self.a)

            mid = self.total_pairs[0]
            xyz_operation(qc, mid[0], mid[1], ancilla, self.j, t, dt, self.j2, trotter_steps * pseudo_constant_a, 1.0,
                          self.ising, self.a)

            for x in self.total_pairs[1:]:
                xyz_operation(qc, x[0], x[1], ancilla, self.j, t, dt, self.j2, trotter_steps * pseudo_constant_a, 2.0,
                              self.ising, self.a)

    def run_circuit(self, anc, qc, noise):

        if anc == 1:
            # All ancilla-assisted measurements
            qc.measure(0, 0)

            if not noise:
                # ideal simulation
                result = execute(qc, backend=simulator, shots=1024).result()
                counts = [result.get_counts(i) for i in range(len(result.results))]
                probs = sort_counts(counts[0], anc, 1024)
                return probs[0] - probs[1]
            else:
                # noisy simulation
                result = execute(qc, backend=simulator, shots=1024, noise_model=noise_model,
                                 basis_gates=basis_gates).result()
                counts = [result.get_counts(i) for i in range(len(result.results))]
                probs = sort_counts(counts[0], anc, 1024)
                return probs[0] - probs[1]

        else:
            # No ancilla (occupation probabilities)
            for x in range(self.n):
                qc.measure(x, x)

            if not noise:
                # ideal simulation
                result = execute(qc, backend=simulator, shots=1024).result()
                counts = [result.get_counts(i) for i in range(len(result.results))]
                measurements = sort_counts(counts[0], self.n, 1024)
                return measurements
            else:
                # noisy simulation
                result = execute(qc, backend=simulator, shots=1024, noise_model=noise_model,
                                 basis_gates=basis_gates).result()
                counts = [result.get_counts(i) for i in range(len(result.results))]
                measurements = sort_counts(counts[0], self.n, 1024)
                return measurements

    def magnetization_per_site(self, t, dt, site, initialstate, trotter_alg, hadamard=False):

        qc_id = QuantumCircuit(self.n + 1, 1)
        qc_noise = QuantumCircuit(self.n + 1, 1)

        self.init_state(qc_id, True, initialstate)
        self.init_state(qc_noise, True, initialstate)

        qc_id.h(0)
        qc_noise.h(0)

        if hadamard:
            qc_id.h(2)  # ------------------------------------------------------------------> tachinno fig 5a
            qc_noise.h(2)

        trotter_alg(qc_id, dt, t, 1)
        trotter_alg(qc_noise, dt, t, 1)

        choose_control_gate('z', qc_id, 0, site + 1)
        choose_control_gate('z', qc_noise, 0, site + 1)

        qc_id.h(0)
        qc_noise.h(0)

        measurement_id = self.run_circuit(1, qc_id, False)
        measurement_noise = self.run_circuit(1, qc_noise, True)

        # Address needed constants for particular paper
        pseudo_constant = 1
        if self.paper in ['joel', 'tachinno']:
            pseudo_constant = 2

        return measurement_id / pseudo_constant, measurement_noise / pseudo_constant

    def all_site_magnetization(self, trotter_alg, total_time=0, dt=0.0, initialstate=0, hadamard=False):
        # Plot changes in magnetization for each site over total time

        data_id = gen_m(self.n, total_time)
        data_noise = gen_m(self.n, total_time)

        for t in range(total_time):
            for site in range(self.n):
                m_id, m_noise = self.magnetization_per_site(t, dt, site, initialstate, trotter_alg, hadamard=hadamard)
                data_id[site, t] += m_id
                data_noise[site, t] += m_noise

        # Plot the results
        all_site_magnetization_plotter(self.n, abs(self.j), dt, total_time, data_id, data_noise)

    def total_magnetization(self, trotter_alg, total_time=0, dt=0.0, initialstate=0):
        # Plot changes in total magnetization over total time

        data_id = gen_m(1, total_time)
        data_noise = gen_m(1, total_time)

        for t in range(total_time):
            total_magnetization_id = 0
            total_magnetization_noise = 0
            for site in range(self.n):
                measurement_id, measurement_noise = self.magnetization_per_site(t, dt, site, initialstate, trotter_alg)
                total_magnetization_id += measurement_id
                total_magnetization_noise += measurement_noise
            data_id[0, t] += total_magnetization_id
            data_noise[0, t] += total_magnetization_noise

        # Plot the results
        total_magnetization_plotter(abs(self.j), total_time, dt, data_id, data_noise)

    def equal_time_correlations(self, trotter_alg, total_t=0, dt=0.0, alpha='', beta='', initialstate=0):

        data_real_id, data_imag_id = gen_m(len(chosen_pairs), total_t), gen_m(len(chosen_pairs), total_t)
        data_real_noise, data_imag_noise = gen_m(len(chosen_pairs), total_t), gen_m(len(chosen_pairs), total_t)

        for pair in chosen_pairs:
            for t in range(total_t):
                for j in range(2):
                    qc_id = QuantumCircuit(self.n + 1, 1)
                    qc_noise = QuantumCircuit(self.n + 1, 1)

                    self.init_state(qc_id, anc, initialstate)
                    self.init_state(qc_noise, anc, initialstate)

                    qc_id.h(0)
                    qc_noise.h(0)

                    trotter_alg(qc_id, dt, t, 0)
                    trotter_alg(qc_noise, dt, t, 0)

                    choose_control_gate(beta, qc_id, 0, pair[1] + 1)
                    choose_control_gate(alpha, qc_id, 0, pair[0] + 1)
                    choose_control_gate(beta, qc_noise, 0, pair[1] + 1)
                    choose_control_gate(alpha, qc_noise, 0, pair[0] + 1)

                    real_or_imag_measurement(qc_id, j)
                    real_or_imag_measurement(qc_noise, j)
                    measurement_id = self.run_circuit(1, qc_id, False)
                    measurement_noise = self.run_circuit(1, qc_noise, True)

                    if j == 0:
                        data_real_id[chosen_pairs.index(pair), t] += measurement_id
                        data_real_noise[chosen_pairs.index(pair), t] += measurement_noise
                    elif j == 1:
                        data_imag_id[chosen_pairs.index(pair), t] += measurement_id
                        data_imag_noise[chosen_pairs.index(pair), t] += measurement_noise

        data_real = [data_real_id, data_real_noise]
        data_imag = [data_imag_id, data_imag_noise]

    def two_point_correlations(self, trotter_alg, total_t, dt, alpha, beta, chosen_pairs, initialstate=0):

        data_real_id, data_imag_id = gen_m(len(chosen_pairs), total_t), gen_m(len(chosen_pairs), total_t)
        data_real_noise, data_imag_noise = gen_m(len(chosen_pairs), total_t), gen_m(len(chosen_pairs), total_t)

        # Address needed constants for particular paper
        pseudo_constant = 1.0
        if self.paper in ['joel', 'tachinno']:
            pseudo_constant = 4.0

        for pair in chosen_pairs:
            for t in range(total_t):
                for j in range(2):
                    qc_id = QuantumCircuit(self.n + 1, 1)
                    qc_noise = QuantumCircuit(self.n + 1, 1)

                    self.init_state(qc_id, 1, initialstate)
                    self.init_state(qc_noise, 1, initialstate)

                    qc_id.h(0)
                    qc_noise.h(0)

                    choose_control_gate(beta, qc_id, 0, pair[1] + 1)
                    trotter_alg(qc_id, dt, t, 1)
                    choose_control_gate(alpha, qc_id, 0, pair[0] + 1)
                    real_or_imag_measurement(qc_id, j)
                    measurement_id = self.run_circuit(1, qc_id, False) / pseudo_constant

                    choose_control_gate(beta, qc_noise, 0, pair[1] + 1)
                    trotter_alg(qc_noise, dt, t, 1)
                    choose_control_gate(alpha, qc_noise, 0, pair[0] + 1)
                    real_or_imag_measurement(qc_noise, j)
                    measurement_noise = self.run_circuit(1, qc_noise, True) / pseudo_constant

                    if j == 0:
                        data_real_id[chosen_pairs.index(pair), t] += measurement_id
                        data_real_noise[chosen_pairs.index(pair), t] += measurement_noise
                    elif j == 1:
                        data_imag_id[chosen_pairs.index(pair), t] += measurement_id
                        data_imag_noise[chosen_pairs.index(pair), t] += measurement_noise

        data_real = [data_real_id, data_real_noise]
        data_imag = [data_imag_id, data_imag_noise]

        # Plot the results
        # Need to alter the plotters

        two_point_correlations_plotter(alpha, beta, abs(self.j), dt, chosen_pairs, data_real, data_imag)

    def occupation_probabilities(self, trotter_alg, total_time=0, dt=0.0, initialstate=0, chosen_states=[]):

        data_id = gen_m(len(chosen_states), total_time)
        data_noise = gen_m(len(chosen_states), total_time)

        for t in range(total_time):
            qc_id = QuantumCircuit(self.n, self.n)
            qc_noise = QuantumCircuit(self.n, self.n)

            self.init_state(qc_id, 0, initialstate)
            self.init_state(qc_noise, 0, initialstate)

            trotter_alg(qc_id, dt, t, 0)
            measurements_id = self.run_circuit(0, qc_id, False)
            for x in chosen_states:
                data_id[chosen_states.index(x), t] = measurements_id[x]

            trotter_alg(qc_noise, dt, t, 0)
            measurements_noise = self.run_circuit(0, qc_noise, True)
            for x in chosen_states:
                data_noise[chosen_states.index(x), t] = measurements_noise[x]

        data = [data_id, data_noise]
        # Plot the results
        # need to alter the plotters
        occ_plotter(chosen_states, abs(self.j), data, self.n, total_time, dt)



# ______________________________________________________________________________________________________________________

# TACHINNO FIG 5a  (All Site Magnetization)
# In tachinno, the arrows are actually referring to the computational basis states (up is 1-comp).
# for tachinnno, the pseudospin factors are multiplied into quantum sim
# needs hadamard manually added, in magnetization_per_site(), for quantum sim


# QUANTUM 2ND ORDER
#spins9 = QuantumSim(j=1, j2=1, bg=0, n=2, choice=False, paper='tachinno')
#spins9.all_site_magnetization(spins9.second_order_trotter, total_time=35, dt=.1, initialstate=0, hadamard=True)

# QUANTUM 1ST ORDER
#spins9 = QuantumSim(j=1, j2=1, bg=0, n=2, choice=False, paper='tachinno')
#spins9.all_site_magnetization(spins9.first_order_trotter, total_time=35, dt=.1, initialstate=0, hadamard=True)


# ---------------------------------------------------------------------------------------------------------------------

# JOEL FIG 2a  (All Site Magnetization)
# In joel, the arrows are referring the the normal spin states (up is zero-comp)

# QUANTUM 1ST ORDER
#spins10 = QuantumSim(j=1, a=1 * 0.5, j2=0, bg=0, n=6, choice=False, open_chain=True, paper='joel')
#spins10.all_site_magnetization(spins10.first_order_trotter, total_time=80, dt=.1, initialstate=spins10.states - 2)

# QUANTUM 2ND ORDER
# spins10 = QuantumSim(j=1, a=1 * 0.5, j2=0, bg=0, n=6, choice=False, open_chain=True, paper='joel')
# spins10.all_site_magnetization(spins10.second_order_trotter, total_time=80, dt=.1, initialstate=spins10.states - 2)


# ----------------------------------------------------------------------------------------------------------------------
# TACCHINO FIG 5b (Occupation Probabilities)
# In tachinno, the arrows are actually referring to the computational basis states (up is 1-comp).

# QUANTUM 1ST ORDER
#spins6 =QuantumSim(j=1, bg=20, n=3, a = 1, j2=1, choice=False, open_chain=True, paper='tachinno')
#i = int('100', 2)                                 # initial state
#c = [int(x, 2) for x in ['100', '010', '111']]    # chosen states
#spins6.occupation_probabilities(spins6.first_order_trotter, total_time=35, dt=.1, initialstate=i, chosen_states=c)

# QUANTUM 2ND ORDER
# spins6 = QuantumSim(j=1, bg=20, n=3, a = 1, j2=1, choice=False, open_chain=True, paper='tachinno')
# i = int('100', 2) # works
# c = [int(x, 2) for x in ['100', '010', '111']]
# spins6.occupation_probabilities(spins6.second_order_trotter, total_time=35, dt=.1, initialstate=i, chosen_states=c)


# --------------------------------------------------------------------------------------------------------------------------
# TACCHINO FIG 7 (Two Point Correlations)
# tachinno counts up sites from '1', so site 1 --> site 0
# tachinno has to have unity = True since the hamiltonian is handled using the pauli operators,
# but the final result gets scaled out according to the spin operators used in the measurement


# QUANTUM 1ST ORDER
#spins13 = QuantumSim(j=1, bg=20, a=1, n=3, choice=False, open_chain=True, paper='tachinno')
#initstate = int('000', 2)
# plot auto, nearest, and next-nearest two-point correlations
#spins13.two_point_correlations(spins13.first_order_trotter, total_t=330, dt=.01, alpha='x',
#                                beta='x', chosen_pairs=[(0, 0)], initialstate=initstate)
# spins13.two_point_correlations(spins13.first_order_trotter, total_t=33, dt=.1, alpha='x',
#                               beta='x', chosen_pairs=[(1, 0)], initialstate=initstate)
# spins13.two_point_correlations(spins13.first_order_trotter, total_t=33, dt=.1, alpha='x',
#                               beta='x', chosen_pairs=[(2, 0)], initialstate=initstate)


# QUANTUM 2ND ORDER
# spins13 = QuantumSim(j=1, bg=20, a=1, n=3, choice=False, open_chain=True, paper='tachinno')
# initstate = int('111', 2)
# plot auto, nearest, and next-nearest two-point correlations
# spins13.two_point_correlations(spins13.second_order_trotter, total_t=330, dt=.01, alpha='x',
# beta='x', chosen_pairs=[(0, 0)], initialstate=initstate)
# spins13.two_point_correlations(spins13.second_order_trotter, total_t=330, dt=.01, alpha='x',
#                                beta='x', chosen_pairs=[(1, 0)], initialstate=initstate)
# spins13.two_point_correlations(spins13.second_order_trotter, total_t=330, dt=.01, alpha='x',
#                                beta='x', chosen_pairs=[(2, 0)], initialstate=initstate)
