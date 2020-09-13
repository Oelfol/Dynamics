# TestingIBMQBackends.py
# Code to run Heisenberg model examples on hardware. Uses only 1st order trotter
# & is based on CompareTrotterAlgorithms.py (still takes paper args) 
# September 2020

import numpy as np
import qiskit as q
from cycler import cycler
import random
import itertools
import math
import matplotlib
import time
import warnings
import cProfile
import scipy.linalg, numpy.linalg
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.sparse.linalg as sla
import scipy.linalg as sl
import qiskit.extensions.unitary as qeu
from qiskit.quantum_info.operators import Operator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import noise
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer, execute, IBMQ

"""
warnings.filterwarnings('ignore')
#https://pypi.org/project/Jupyter-Beeper/#description

import jupyter_beeper

b = jupyter_beeper.Beeper()

# Default config is frequency=440 Hz, secs=0.7 seconds, and
# blocking=False (b.beep() will return when the sound begins)
b.beep()

# We have to put a sleep statement, since the previous call 
# for b.beep() is non blocking, and then it will overlap with
# the next call to b.beep()
time.sleep(2)

# This will not return until the beep is completed
b.beep(frequency=530, secs=0.7, blocking=True)
"""

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


dev_name = 'ibmq_santiago'
device, nm, bg, cm = qc_noise_model(dev_name)  # noise model, basis gates, coupling map

# ==================================== Plotting Helpers ============================================================ >
colors = ['g', 'k', 'maroon', 'mediumblue', 'slateblue', 'limegreen', 'b', 'r', 'olive']
custom_cycler = (cycler(color=colors) + cycler(lw=[2] * len(colors)))
matplotlib.rcParams['figure.figsize'] = [10, 10]
plt.rc('font', size=12)
plt.rc('axes', titlesize=12)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('legend', fontsize=12)
plt.rc('figure', titlesize=12)


def set_up_axes_one(rows, cols):
    fig, axs = plt.subplots(rows, cols)
    for col in range(cols):
        for row in range(rows):
            ax = axs[row, col]
            ax.set_prop_cycle(custom_cycler)
            ax.margins(x=0)
    fig.tight_layout(pad=3.5)
    return fig, axs


def set_up_axes_two(num_axes):
    if num_axes == 1:
        fig, ax = plt.subplots()
        ax.set_prop_cycle(custom_cycler)
        ax.margins(x=0)
        fig.tight_layout(pad=3.5)
        return fig, ax

    elif num_axes == 2:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.set_prop_cycle(custom_cycler)
        ax2.set_prop_cycle(custom_cycler)
        ax1.margins(x=0)
        ax2.margins(x=0)
        fig.subplots_adjust(hspace=0)
        fig.tight_layout(pad=3.5)
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
        fig.tight_layout(pad=3.5)
        return fig, (ax1, ax2, ax3)



def occ_plotter(chosen_states, j, n, total_time, dt, data_gates, data, data_cl):

    data_one_id, data_one_noise = data[0], data[1]

    fig, (one, cl, gates) = set_up_axes_two(3)
    x1 = [i * abs(j) * dt for i in range(total_time)]
    for state in chosen_states:
        dex = chosen_states.index(state)
        label = np.binary_repr(state).zfill(n)
        one.plot(x1, data_one_id[dex, :].toarray().tolist()[0], label=label, linestyle="-")
        one.plot(x1, data_one_noise[dex, :].toarray().tolist()[0], label=label, linestyle=":")
        cl.plot(x1, data_cl[dex, :].toarray().tolist()[0], label=label)

    gates.plot(x1, data_gates[0, :].toarray().tolist()[0], label="First-Order")
    gates.set_ylabel("Gates")
    gates.set_xlabel("Jt")

    plot_dataset_byrows(gates, "Formula", "Gates", r'$\it{Jt}$')
    plot_dataset_byrows(one, "States", "Probability", r'$\it{Jt}$' + " (First-Order Trotter)")
    plot_dataset_byrows(cl, "States", "Probability", r'$\it{Jt}$' + " (Exact)")

    fig.suptitle('Occupation Probabilities')
    plt.show()


def two_point_correlations_plotter(j, dt, pairs, data_one, data_cl):
    # todo out alpha, beta
    
    plt.rc('font', size=14)
    plt.rc('axes', titlesize=14)
    plt.rc('axes', labelsize=16)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.rc('legend', fontsize=14)
    plt.rc('figure', titlesize=14)
    
    matplotlib.rcParams['figure.figsize'] = [20, 10]
    real_one, imag_one, gates_one = data_one[0], data_one[1], data_one[2]
    real_one_id, imag_one_id = real_one[0], imag_one[0]
    real_one_noise, imag_one_noise = real_one[1], imag_one[1]
    real_cl_data, im_cl_data = data_cl[0], data_cl[1]
    re_label = r'$Re \langle S_{\alpha}^{i}(t)S_{\beta}^{j}(0)\rangle $'
    im_label = r'$Im \langle S_{\alpha}^{i}(t)S_{\beta}^{j}(0)\rangle $'

    fig, axs = set_up_axes(2, 2)
    real1, im1, im2, gate1 = axs[0, 0], axs[1, 0], axs[0, 1], axs[1, 1]
    real_cl, im_cl = axs[0, 1], axs[1, 1]

    scaler = abs(j) * dt
    p = [i * scaler for i in range(len(real_one_id.toarray()[0][:].tolist()))]
    for x in pairs:
        dx = pairs.index(x)
        real1.plot(p, real_one_id.toarray()[dx][:].tolist(), label=str(x))
        real1.plot(p, real_one_noise.toarray()[dx][:].tolist(), label=str(x), linestyle=":")
        real_cl.plot(p, real_cl_data.toarray()[dx][:].tolist(), label=str(x))
        im_cl.plot(p, im_cl_data.toarray()[dx][:].tolist(), label=str(x))
        im1.plot(p, imag_one_id.toarray()[dx][:].tolist(), label=str(x))
        im1.plot(p, imag_one_noise.toarray()[dx][:].tolist(), label=str(x), linestyle=":")

    gate1.plot(p, gates_one.toarray()[0, :].tolist())
    gate1.set_ylabel("Gates", fontsize="medium")

    plot_dataset_byrows(real1, "Site Pairs", re_label, "Jt (First-Order Trotter)")
    plot_dataset_byrows(im1, "Site Pairs", im_label, "Jt (First-Order Trotter)")
    plot_dataset_byrows(real_cl, "Site Pairs", re_label, "Jt (Exact)")
    plot_dataset_byrows(im_cl, "Site Pairs", im_label, "Jt (Exact)")

    fig.suptitle('Two-Point Correlations')
    plt.show()


def total_magnetization_plotter(j, total_t, dt, data, data_gates, data_cl):

    data_one_id, data_one_noise = data[0], data[1]

    fig, (one, gates, cl) = set_up_axes_two(3)
    x1 = [i * j * dt for i in range(total_t)]
    one.plot(x1, data_one_id.toarray()[0][:].tolist(), linestyle='-')
    one.plot(x1, data_one_noise.toarray()[0][:].tolist(), linestyle=":")
    gates.plot(x1, data_gates.toarray()[0][:].tolist(), label="First-Order")
    cl.plot(x1, data_cl.toarray()[0][:].tolist())

    one.set_xlabel(r'$\it{Jt}$')
    one.set_ylabel('Total Magnetization')
    cl.set_ylabel('Total Magnetization')
    cl.set_xlabel(r'$\it{Jt}$')
    gates.set_ylabel("Gates")
    gates.set_xlabel(r'$\it{Jt}$')
    fig.suptitle('Total Magnetization')
    plt.show()


def all_site_magnetization_plotter(n, j, dt, total_t, data, data_gates, data_cl):

    data_one_id, data_one_noise = data[0], data[1]

    fig, (one, cl, gates) = set_up_axes_two(3)
    x1 = [i * j * dt for i in range(total_t)]

    for site in range(n):
        one.plot(x1, data_one_id.toarray()[site][:].tolist(), linestyle='-', label=site)
        one.plot(x1, data_one_noise.toarray()[site][:].tolist(), linestyle=":", label=site)
        cl.plot(x1, data_cl.toarray()[site][:].tolist(), label=site)

    gates.plot(x1, data_gates.toarray()[0][:].tolist(), label="First-Order", linestyle='-')
    gates.set_ylabel("Gates")

    plot_dataset_byrows(one, 'Sites', 'Magnetization', r'$\it{Jt}$')
    plot_dataset_byrows(cl, 'Sites', 'Magnetization', r'$\it{Jt}$')
    fig.suptitle('Magnetization per Site', fontsize=16)
    plt.show()


# ========================================= Pauli matrices and pseudospin operators ================================== >

sx = sps.csc_matrix(np.array([[0, 1], [1, 0]]))
sy = sps.csc_matrix(np.complex(0, 1) * np.array([[0, -1], [1, 0]]))
sz = sps.csc_matrix(np.array([[1, 0], [0, -1]]))
identity = sps.csc_matrix(np.eye(2, dtype=complex))
plus = sps.csc_matrix(sx * (1 / 2) + np.complex(0, 1) * sy * (1 / 2))
minus = sps.csc_matrix(sx * (1 / 2) - np.complex(0, 1) * sy * (1 / 2))


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
    ax.legend(loc='right', title=legendtitle)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)


def commutes(a, b):
    # Test whether operators commute
    comp = a.dot(b) - b.dot(a)
    comp = comp.toarray().tolist()
    if np.count_nonzero(comp) == 0:
        return True
    else:
        return False


# ============================================ Classical Simulations ================================================= >


class ClassicalSpinChain:

    def __init__(self, j=0.0, j2=0.0, bg=0.0, a=0.0, n=0, open_chain=False, unity=False, choice=False, ising=False,
                 paper='', transverse=False):

        self.j = j  # ===============================================# coupling constant
        self.j2 = j2  # =============================================# how much to scale next-nearest coupling
        self.bg = bg  # =============================================# magnetic field strength
        self.a = a  # ===============================================# anisotropy constant jz/j
        self.n = n  # ===============================================# sites
        self.open_chain = open_chain  # =============================# whether open chain
        self.states = 2 ** n  # =====================================# number of basis states
        self.unity = unity  # =======================================# whether h-bar/2 == 1 (h-bar == 1 elsewise)
        self.choice = choice  # =====================================# whether next-nearest, bool
        self.ising = ising  # =======================================# for ising model
        self.paper = paper  # =======================================# refer to settings for a specific paper
        self.transverse = transverse  # =============================# whether transverse field ising model
        self.hamiltonian = sps.lil_matrix(np.zeros([2 ** n, 2 ** n], complex))
        neighbors, nn_neighbors, autos = gen_pairs(self.n, self.choice, False, self.open_chain)
        multipliers = [1, 1, self.a]

        ops = ['x', 'y', 'z']
        if ising:
            ops = ['z']
            multipliers = [self.a]

        for y in neighbors:
            for op in ops:
                dex = ops.index(op)
                s1, s2 = spin_op(op, y[1], self.n, self.unity), spin_op(op, y[0], self.n, self.unity)
                self.hamiltonian += s1.dot(s2) * self.j * multipliers[dex]
        for z in nn_neighbors:
            for op in ops:
                dex = ops.index(op)
                s1, s2 = spin_op(op, z[1], self.n, self.unity), spin_op(op, z[0], self.n, self.unity)
                self.hamiltonian += s1.dot(s2) * self.j * multipliers[dex] * self.j2
        for x in range(self.n):
            if self.bg != 0 and self.transverse == False:
                s_final = spin_op('z', x, self.n, self.unity) * self.bg / 2
                self.hamiltonian += s_final
            elif self.bg != 0 and self.transverse == True:
                s_final = spin_op('x', x, self.n, self.unity) * self.bg / 2
                self.hamiltonian += s_final

        self.hamiltonian = sps.csc_matrix(self.hamiltonian)

    def test_commuting_matrices(self):
        ops, multipliers = ['x', 'y', 'z'], [1, 1, self.a]
        neighbors, nn_neighbors, autos = gen_pairs(self.n, self.choice, False, self.open_chain)
        if self.ising:
            ops = ['z']
            multipliers = [self.a]

        n, dex_terms = self.n, 0
        even_terms = sps.lil_matrix(np.zeros([2 ** n, 2 ** n], complex))
        odd_terms = sps.lil_matrix(np.zeros([2 ** n, 2 ** n], complex))

        for y in neighbors:
            for op in ops:
                dex = ops.index(op)
                s1, s2 = spin_op(op, y[1], self.n, self.unity), spin_op(op, y[0], self.n, self.unity)
                if dex_terms % 2 == 0:
                    even_terms += s1.dot(s2) * self.j * multipliers[dex]
                else:
                    odd_terms += s1.dot(s2) * self.j * multipliers[dex]
            dex_terms += 1

        # not yet implemented for next-nearest neighbors TODO

        commuting = True
        if not commutes(even_terms, odd_terms):
            commuting = False

        terms = even_terms + odd_terms
        for x in range(self.n):
            if self.bg != 0 and self.transverse == False:
                s_final = spin_op('z', x, self.n, self.unity) * self.bg / 2
                if not commutes(terms, s_final):
                    commuting = False
            elif self.bg != 0 and self.transverse == True:
                s_final = spin_op('x', x, self.n, self.unity) * self.bg / 2
                if not commutes(terms, s_final):
                    commuting = False

        return commuting

    def eig(self):

        h = self.hamiltonian.toarray()
        evals, evect = sl.eigh(h)[0], sl.eigh(h)[1]
        return evals, sps.csc_matrix(evect)

    def equal_time_correlations_c(self, total_time, dt, psi0, op_order):

        nn, nnn, auto = gen_pairs(self.n, self.choice, True, self.open_chain)
        pairs_ = nn + nnn + auto
        data = gen_m(len(pairs_), total_time)

        for t in range(total_time):
            if self.unity:
                t = t / 2
            u = sl.expm(self.hamiltonian * dt * t * (-1j))
            bra = np.conj(u.dot(psi0).T)
            for pair in pairs_:
                si = spin_op(op_order[0], pair[0], self.n, self.unity)
                sj = spin_op(op_order[1], pair[1], self.n, self.unity)
                ket = si.dot(sj.dot(u.dot(psi0)))
                res = bra.dot(ket).toarray()[0][0]
                data[pairs_.index(pair), t] = np.conj(res) * res

    def two_point_correlations_c(self, total_time, dt, psi0, op_order, pairs=[]):

        if pairs == []:
            # if wanting to generate all pairs, feed in none
            nn, nnn, auto = gen_pairs(self.n, self.choice, True, self.open_chain)
            pairs = nn + nnn + auto
        dyn_data_real, dyn_data_imag = gen_m(len(pairs), total_time), gen_m(len(pairs), total_time)

        # Account for differences between examples from papers
        pseudo_constant = 1
        if self.paper in ['tachinno']:
            pseudo_constant = 4

        for t in range(total_time):
            t_ = t  # self.j * t # why was this multiplied by j ???
            u = sl.expm(self.hamiltonian * dt * t_ * (-1j))
            u_dag = sl.expm(self.hamiltonian * dt * t_ * (1j))
            psi_dag = np.conj(psi0).transpose()
            for x in range(len(pairs)):
                si = spin_op(op_order[1], pairs[x][0], self.n, self.unity)
                sj = spin_op(op_order[0], pairs[x][1], self.n, self.unity)
                ket = u_dag.dot(si.dot(u.dot(sj.dot(psi0))))
                res = psi_dag.dot(ket).toarray()[0][0]
                dyn_data_real[x, t] = np.real(res) / pseudo_constant
                dyn_data_imag[x, t] = np.imag(res) / pseudo_constant
        return dyn_data_real, dyn_data_imag

    def occupation_probabilities_c(self, total_time=0, dt=0.0, initialstate=None, chosen_states=[]):

        psi0 = initialstate
        basis_matrix = sps.csc_matrix(np.eye(self.states))
        data = gen_m(len(chosen_states), total_time)
        for t in range(total_time):
            t_ = t  # * self.j again, why???
            u = sl.expm(self.hamiltonian * dt * t_ * (-1j))
            psi_t = u.dot(psi0)
            index_ = 0
            for i in chosen_states:  # range(self.states):
                basis_bra = (basis_matrix[i, :])
                probability = (basis_bra.dot(psi_t)).toarray()[0][0]
                probability = ((np.conj(psi_t).transpose()).dot(np.conj(basis_bra).transpose())).toarray()[0][
                                  0] * probability
                data[index_, t] = probability
                index_ += 1

        return data

    def magnetization_per_site_c(self, total_time, dt, psi0, site):

        # Account for differences between examples from papers
        pseudo_constant, data = 1.0, gen_m(1, total_time)
        if self.paper in ['tachinno']:
            pseudo_constant = 2.0

        for t in range(total_time):
            t_ = t * self.j
            u = sl.expm(self.hamiltonian * dt * t_ * (-1j))
            bra = np.conj(u.dot(psi0).transpose())
            s_z = spin_op('z', site, self.n, self.unity)
            ket = s_z.dot(u.dot(psi0))
            data[0, t] += (bra.dot(ket).toarray()[0][0]) / pseudo_constant
        return data

    def all_site_magnetization_c(self, total_time, dt, psi0):

        data = gen_m(self.n, total_time)

        for site in range(self.n):
            site_data = self.magnetization_per_site_c(total_time, dt, psi0, site)
            data[site, :] += site_data

        return data

    def total_magnetization_c(self, total_time, dt, psi0):

        data = gen_m(1, total_time)
        for site in range(self.n):
            site_data = self.magnetization_per_site_c(total_time, dt, psi0, site)
            data = data + site_data

        return data


# ============================================== Helper functions for Qiskit Simulations ============================= >


def sort_counts(count, qs, shots):
    vec = []
    for i in range(2 ** qs):
        binary = np.binary_repr(i).zfill(qs)
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
        qc.rx(1 * math.pi / 2, 0)


def zz_operation(qc, a, b, delta):
    # Helper for time evolution operator
    qc.cx(a, b)
    qc.rz(2 * delta, b)
    qc.cx(a, b)


def xx_operation(qc, a, b, delta):
    # Helper for time evolution operator
    # Not currently in use
    qc.ry(math.pi / 2, a)
    qc.ry(math.pi / 2, b)
    qc.cx(a, b)
    qc.rz(2 * delta, b)
    qc.cx(a, b)
    qc.ry(-1 * math.pi / 2, a)
    qc.ry(-1 * math.pi / 2, b)


def yy_operation(qc, a, b, delta):
    # Helper for time evolution operator
    # Not currently in use
    qc.rx(math.pi / 2, a)
    qc.rx(math.pi / 2, b)
    qc.cx(a, b)
    qc.rz(2 * delta, b)
    qc.cx(a, b)
    qc.rx(-1 * math.pi / 2, a)
    qc.rx(-1 * math.pi / 2, b)


def three_cnot_evolution(qc, a, b, ancilla, j, t, dt, j2, trotter_steps, constant, ising, a_constant, boundary):
    # Time evolving the system termwise by the elements in the hamiltonian using S2 gateset # TODO clarify
    nnn_term = j2 
    if (abs(a - b) == 1 or boundary==1): # TODO fix this in CompareTrotterAlgorithms.py
        nnn_term = 1

    a_, b_ = a + ancilla, b + ancilla
    if ising:
        zz_operation(qc, a_, b_, j * a_constant * t * dt * nnn_term / (constant * trotter_steps))

    else:
        delta = j * t * dt * nnn_term / (constant * trotter_steps)
        qc.cx(a_, b_)
        qc.rx(2 * delta * a_constant - math.pi / 2, a_)
        qc.rz(2 * delta * a_constant, b_)
        qc.h(a_)
        qc.cx(a_, b_)
        qc.h(a_)
        qc.rz(- 2 * delta, b_)
        qc.cx(a_, b_)
        qc.rx(math.pi / 2, a_)
        qc.rx(-math.pi / 2, b_)




# =================================================== Qiskit Simulations ============================================= >

class QuantumSim:

    def __init__(self, j=0.0, bg=0.0, a=1.0, n=0, j2=1, choice=True, open_chain=True, transverse=False, paper='',
                 ising=False, epsilon=0):

        self.j = j  # ================================================= # coupling constant
        self.bg = bg  # ================================================# magnetic field strength
        self.n = n  # ==================================================# number of spins
        self.states = 2 ** n  # ========================================# number of basis states
        self.choice = choice  # ========================================# whether to have next-nearest neighbors
        self.a = a  # ==================================================# anisotropy jz / j
        self.j2 = j2  # ================================================# how much to scale next-nearest coupling
        self.open_chain = open_chain  # ================================# open chain = periodic boundary
        self.transverse = transverse  # ================================# for transverse field ising model
        self.paper = paper  # =========================# direct reference for all the unique aspects of different papers
        self.ising = ising  # ==========================================# Whether using the ising model
        self.unity = False  # ==========================================# Depends on paper
        self.epsilon = epsilon  # ======================================# Desired precision - use to find trotter steps
        if self.paper == 'francis':
            self.unity = True

        self.h_commutes = ClassicalSpinChain(j=self.j, j2=self.j2, bg=self.bg, a=self.a, n=self.n,
                                             open_chain=self.open_chain, unity=self.unity,
                                             choice=self.choice).test_commuting_matrices()
        self.pairs_nn, self.pairs_nnn, autos = gen_pairs(self.n, self.choice, False, self.open_chain)
        self.total_pairs = self.pairs_nn + self.pairs_nnn

    def init_state(self, qc, ancilla, initial_state):
        # Initialize a circuit in the desired spin state. Add a qubit if there is an ancilla measurement.
        state_temp, anc, index = np.binary_repr(initial_state).zfill(self.n)[::-1], int(ancilla), 0
        for x in state_temp:
            if x == '1':
                qc.x(index + anc)
            index += 1

    def update_gates(self):
        if not self.ising:
            return 12
        elif self.ising:
            return 3

    def first_order_trotter(self, qc, dt, t, ancilla):
        trotter_steps, num_gates = 1, 0
        if self.h_commutes:
            print("H commutes, trotter_steps = 1")
        else:
            t_ = t
            if t == 0:
                t_ = 1
                print("First order trotter in progress..")
            trotter_steps = math.ceil(abs(self.j) * t_ * dt / self.epsilon)
            print('trotter steps:', trotter_steps, " t:", t)

        # Address needed constants for particular paper
        pseudo_constant_a, mag_constant = 1.0, 1.0
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
                num_gates = num_gates + 1
            for x in self.total_pairs:
                boundary = 0
                if x[0] == 0 and x[1] == self.n - 1 and self.open_chain == False: 
                    boundary += 1 
                three_cnot_evolution(qc, x[0], x[1], ancilla, self.j, t, dt, self.j2, trotter_steps * pseudo_constant_a, 1.0,
                                     self.ising, self.a, boundary)
                num_gates = num_gates + self.update_gates()


        return num_gates


    def run_circuit(self, anc, qc, noise):

        if anc == 1:
            # All ancilla-assisted measurements
            qc.measure(0, 0)

            if not noise:
                # ideal simulation
                result = execute(qc, backend=simulator, shots=50000).result()
                counts = [result.get_counts(i) for i in range(len(result.results))]
                probs = sort_counts(counts[0], anc, 50000)
                return probs[0] - probs[1]
            else:
                # run on the backend
                result = execute(qc, backend=device, shots=1800).result()
                counts = [result.get_counts(i) for i in range(len(result.results))]
                probs = sort_counts(counts[0], anc, 1800)
                return probs[0] - probs[1]

        else:
            # No ancilla (occupation probabilities)
            for x in range(self.n):
                qc.measure(x, x)

            if not noise:
                # ideal simulation
                result = execute(qc, backend=simulator, shots=50000).result()
                counts = [result.get_counts(i) for i in range(len(result.results))]
                measurements = sort_counts(counts[0], self.n, 50000)
                return measurements
            else:
                # run on the backend
                result = execute(qc, backend=device, shots=1800).result()
                counts = [result.get_counts(i) for i in range(len(result.results))]
                measurements = sort_counts(counts[0], self.n, 1800)
                return measurements

    def magnetization_per_site_q(self, t, dt, site, initialstate, trotter_alg, hadamard=False):
        # --------- #
        qc_id, qc_noise = QuantumCircuit(self.n + 1, 1), QuantumCircuit(self.n + 1, 1)

        self.init_state(qc_id, True, initialstate)
        self.init_state(qc_noise, True, initialstate)
        qc_id.h(0)
        qc_noise.h(0)
        if hadamard:
            qc_id.h(2)  # ----------------------------------------------------------------------------> tachinno fig 5a
            qc_noise.h(2)

        num_gates_id = trotter_alg(qc_id, dt, t, 1)
        num_gates_noise = trotter_alg(qc_noise, dt, t, 1)
        num_gates = num_gates_noise  # only need one

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

        return measurement_id / pseudo_constant, measurement_noise / pseudo_constant, num_gates
        # --------- #

    def all_site_magnetization_q(self, trotter_alg, total_time=0, dt=0.0, initialstate=0, hadamard=False):
        # --------- #
        data_id = gen_m(self.n, total_time)
        data_noise = gen_m(self.n, total_time)
        data_gates = gen_m(1, total_time)
     
        id_or_noisy = 2
        print("Number of circuits needed:", total_time * id_or_noisy * self.n)
    
        for t in range(total_time):
            print("Time: ", t)
            num_gates_total = 0
            for site in range(self.n):
                print("Site: ", site)
                m_id, m_noise, num_gates = self.magnetization_per_site_q(t, dt, site, initialstate, trotter_alg,
                                                                         hadamard=hadamard)
                data_id[site, t] += m_id
                data_noise[site, t] += m_noise
                num_gates_total += num_gates
            data_gates[0, t] += num_gates_total / self.n

        data = [data_id, data_noise]
        return data, data_gates
        # --------- #

    def total_magnetization_q(self, trotter_alg, total_time=0, dt=0.0, initialstate=0):
        # --------- #
        data_id = gen_m(1, total_time)
        data_noise = gen_m(1, total_time)
        data_gates = gen_m(1, total_time)
        
        id_or_noisy = 2
        print("Number of circuits needed:", total_time * id_or_noisy * self.n)

        for t in range(total_time):
            print("Time: ", t)
            total_magnetization_id = 0
            total_magnetization_noise = 0
            num_gates_total = 0
            for site in range(self.n):
                print("Site: ", site)
                measurement_id, measurement_noise, num_gates = self.magnetization_per_site_q(t, dt, site, initialstate,
                                                                                             trotter_alg)
                total_magnetization_id += measurement_id
                total_magnetization_noise += measurement_noise
                num_gates_total += num_gates

            data_id[0, t] += total_magnetization_id
            data_noise[0, t] += total_magnetization_noise
            data_gates[0, t] += num_gates_total / self.n

        data = [data_id, data_noise]
        return data, data_gates
        # --------- #

    def equal_time_correlations_q(self, trotter_alg, total_t=0, dt=0.0, alpha='', beta='', initialstate=0):
        # --------- #
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
        # --------- #

    def two_point_correlations_q(self, trotter_alg, total_t, dt, alpha, beta, chosen_pairs, initialstate=0):
        # --------- #
        data_real_id, data_imag_id = gen_m(len(chosen_pairs), total_t), gen_m(len(chosen_pairs), total_t)
        data_real_noise, data_imag_noise = gen_m(len(chosen_pairs), total_t), gen_m(len(chosen_pairs), total_t)
        data_gates = gen_m(1, total_t)
        
        id_or_noisy = 2
        print("Number of circuits needed:", total_t * id_or_noisy * 2 * len(chosen_pairs))

        # Address needed constants for particular paper
        pseudo_constant = 1.0
        if self.paper in ['joel', 'tachinno']:
            pseudo_constant = 4.0
   
        for pair in chosen_pairs:
            print("Pair: ", pair)
            for t in range(total_t):
                print("Time: ", time)
                gates_t = 0
                for j in range(2):
                    qc_id = QuantumCircuit(self.n + 1, 1)
                    qc_noise = QuantumCircuit(self.n + 1, 1)

                    self.init_state(qc_id, 1, initialstate)
                    self.init_state(qc_noise, 1, initialstate)

                    qc_id.h(0)
                    qc_noise.h(0)

                    choose_control_gate(beta, qc_id, 0, pair[1] + 1)
                    num_gates_id = trotter_alg(qc_id, dt, t, 1)
                    choose_control_gate(alpha, qc_id, 0, pair[0] + 1)
                    real_or_imag_measurement(qc_id, j)
                    measurement_id = self.run_circuit(1, qc_id, False) / pseudo_constant

                    choose_control_gate(beta, qc_noise, 0, pair[1] + 1)
                    num_gates_noise = trotter_alg(qc_noise, dt, t, 1)
                    choose_control_gate(alpha, qc_noise, 0, pair[0] + 1)
                    real_or_imag_measurement(qc_noise, j)
                    measurement_noise = self.run_circuit(1, qc_noise, True) / pseudo_constant

                    # only need one
                    num_gates = num_gates_noise
                    gates_t += num_gates

                    if j == 0:
                        data_real_id[chosen_pairs.index(pair), t] += measurement_id
                        data_real_noise[chosen_pairs.index(pair), t] += measurement_noise
                    elif j == 1:
                        data_imag_id[chosen_pairs.index(pair), t] += measurement_id
                        data_imag_noise[chosen_pairs.index(pair), t] += measurement_noise

                if chosen_pairs.index(pair) == 0:
                    data_gates[0, t] += gates_t / 2

        data_real = [data_real_id, data_real_noise]
        data_imag = [data_imag_id, data_imag_noise]
        return data_real, data_imag, data_gates
        # --------- #

    def occupation_probabilities_q(self, trotter_alg, total_time=0, dt=0.0, initialstate=0, chosen_states=[]):
        # --------- #
        data_id = gen_m(len(chosen_states), total_time)
        data_noise = gen_m(len(chosen_states), total_time)
        data_gates = gen_m(1, total_time)
        
        id_or_noisy = 2
        print("Number of circuits needed:", total_time * id_or_noisy)

        for t in range(total_time):
            print("Time: ", time)
            qc_id = QuantumCircuit(self.n, self.n)
            qc_noise = QuantumCircuit(self.n, self.n)

            self.init_state(qc_id, 0, initialstate)
            self.init_state(qc_noise, 0, initialstate)

            num_gates_id = trotter_alg(qc_id, dt, t, 0)
            measurements_id = self.run_circuit(0, qc_id, False)
            for x in chosen_states:
                data_id[chosen_states.index(x), t] = measurements_id[x]

            num_gates_noise = trotter_alg(qc_noise, dt, t, 0)
            measurements_noise = self.run_circuit(0, qc_noise, True)
            for x in chosen_states:
                data_noise[chosen_states.index(x), t] = measurements_noise[x]

            # only need one
            num_gates = num_gates_noise
            data_gates[0, t] = num_gates
     
        data = [data_id, data_noise]
        return data, data_gates
        # --------- #


# ========================================= Compare Classical & Quantum Simulations  =================================>


class HeisenbergModel():

    def __init__(self, j=0.0, bg=0.0, a=1.0, n=0, j2=1, choice=True, open_chain=True, transverse=False, paper='',
                 ising=False, epsilon=0, unity=False):
        # --------- #
        self.classical_chain = ClassicalSpinChain(j=j, j2=j2, bg=bg, a=a, n=n, open_chain=open_chain, unity=unity,
                                                  choice=choice, ising=ising, paper=paper, transverse=transverse)
        self.quantum_chain = QuantumSim(j=j, bg=bg, a=a, n=n, j2=j2, choice=choice, open_chain=open_chain,
                                        transverse=transverse, paper=paper, ising=ising, epsilon=epsilon)
        self.first = self.quantum_chain.first_order_trotter
        # --------- #

    def all_site_magnetization(self, total_t=0, dt=0, initstate=0, hadamard=False):
        # --------- #
        num_states = self.classical_chain.states
        psi0 = sps.csc_matrix(np.zeros((num_states, 1)))
        
        if hadamard:
            spins3 = self.classical_chain.states
            psi0 += init_spin_state(0, spins3) - (init_spin_state(2, spins3) + init_spin_state(3, spins3)) / math.sqrt(2)
        else:
            psi0 += init_spin_state(initstate, num_states)
        data_one, data_gates_one = self.quantum_chain.all_site_magnetization_q(self.first, total_time=total_t, dt=dt,
                                                                               initialstate=initstate,
                                                                               hadamard=hadamard)
        print("Its running")
        data_cl = self.classical_chain.all_site_magnetization_c(total_t, dt, psi0)
        n, j = self.classical_chain.n, self.classical_chain.j
        all_site_magnetization_plotter(n, j, dt, total_t, data_one, data_gates_one, data_cl)
        # --------- #

    def total_magnetization(self, total_t=0, dt=0, initstate=0):
        # --------- #
        psi0 = init_spin_state(initstate, self.classical_chain.states)
        data_one, data_gates_one = self.quantum_chain.total_magnetization_q(self.first, total_t, dt,
                                                                            initialstate=initstate)
        data_cl = self.classical_chain.total_magnetization_c(total_t, dt, psi0)
        j_ = self.classical_chain.j
        total_magnetization_plotter(j_, total_t, dt, data_one, data_gates_one, data_cl)
        # --------- #

    def equal_time_correlations(self):  # Implement TODO -- would need an example
        # --------- #
        # --------- #
        pass

    def two_point_correlations(self, op_order='', total_t=0, dt=0, pairs=[], initstate=0):
        # --------- #
        alpha, beta = op_order[0], op_order[1]
        psi0 = init_spin_state(initstate, self.classical_chain.states)
        data_real_one, data_imag_one, data_gates_one = self.quantum_chain.two_point_correlations_q(self.first, total_t,
                                                                                                   dt, alpha, beta,
                                                                                                   pairs,
                                                                                                   initialstate=initstate)
        data_real_cl, data_imag_cl = self.classical_chain.two_point_correlations_c(total_t, dt, psi0, op_order,
                                                                                   pairs=pairs)
        print("Its running")
        j_ = self.classical_chain.j
        d_one  = [data_real_one, data_imag_one, data_gates_one]
        data_cl = [data_real_cl, data_imag_cl]
        two_point_correlations_plotter(j_, dt, pairs, d_one, data_cl)

        # --------- #

    def occupation_probabilities(self, total_t=0, dt=0, initstate=0, chosen_states=[]):
        # --------- #
        print("Its going")
        psi0 = init_spin_state(initstate, self.classical_chain.states)
        data_one, data_gates_one = self.quantum_chain.occupation_probabilities_q(self.first, total_t,
                                                                                 dt, initialstate=initstate,
                                                                                 chosen_states=chosen_states)
        data_cl = self.classical_chain.occupation_probabilities_c(total_t, dt, initialstate=psi0,
                                                                  chosen_states=chosen_states)
        n = self.classical_chain.n
        occ_plotter(chosen_states, self.classical_chain.j, n, total_t, dt, data_gates_one, data_one, data_cl)
        # --------- #


# ==================================== TESTERS ==================================================================== >

# TACHINNO FIG 5a  (All Site Magnetization)
# 140 circuits 
model = HeisenbergModel(j=1, bg=0, n=2, j2=0, choice=False, paper='tachinno', epsilon=0.2, unity=True)
model.all_site_magnetization(total_t=35, dt=0.1, initstate=0, hadamard=True)

# ---------------------------------------------------------------------------------------------------------------------

# JOEL FIG 2a  (All Site Magnetization)
# 9600 circuits 
#model = HeisenbergModel(j=1, bg=0, a=0.5, n=6, j2=0, choice=False, open_chain=True, paper='joel', epsilon=0.4)
#initstate=model.classical_chain.states - 2
#model.all_site_magnetization(total_t=800, dt=0.01, initstate=initstate)

# ----------------------------------------------------------------------------------------------------------------------
# TACCHINO FIG 5b (Occupation Probabilities)
#700 circuits 
#model = HeisenbergModel(j=1, bg=20, a=1, n=3, j2=1, choice=False, open_chain=True, transverse=False, paper='tachinno',
#                 ising=False, epsilon=0.2, unity=True)
#c = [int(x, 2) for x in ['100', '010', '111']]
#model.occupation_probabilities(total_t=350, dt=0.01, initstate=int('100', 2), chosen_states=c)


# --------------------------------------------------------------------------------------------------------------------------
# TACCHINO FIG 7 (Two Point Correlations)

#model = HeisenbergModel(j=1, bg=20, a=1, n=3, j2=1, choice=False, open_chain=True, transverse=False, paper='tachinno',
#                 ising=False, epsilon=0.2, unity=True)
#auto, nearest, next-nearest two-point correlations
# 1320 circuits 
#model.two_point_correlations(op_order='xx', total_t=330, dt=.01, pairs=[(1,0)], initstate=int('000', 2))
#model.two_point_correlations(op_order='xx', total_t=330, dt=.01, pairs=[(0,0)], initstate=int('000', 2))
#model.two_point_correlations(op_order='xx', total_t=330, dt=.01, pairs=[(2,0)], initstate=int('000', 2))


# ----------------------------------------------------------------------------------------------------------------------
# TACCHINO FIG 5c (TOTAL MAGNETIZATION)
# 2600 circuits 
#model = HeisenbergModel(j=1, bg=2, a=1, n=2, j2=1, choice=False, open_chain=True, transverse=True, paper='tachinno',
#                 ising=True, epsilon=0.2, unity=True)
#model.total_magnetization(total_t=650, dt=0.01, initstate=0)

# ----------------------------------------------------------------------------------------------------------------------
# 7200 circuits 
#model = HeisenbergModel(j=-.84746, j2=1, bg=0, a=1, n=4, open_chain=False, unity=True, choice=False, paper='francis',
#                        epsilon=0.2)
#model.two_point_correlations(op_order='xx', total_t=600, dt=.01, pairs=[(1, 1), (2, 1), (3, 1)], initstate=0)
