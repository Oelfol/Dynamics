# DynamicSpinSusceptibility
# Code to calculate dynamic spin susceptibility for three and four-site problems using discreet fourier transform.
# October 2020

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
from matplotlib import cm
import scipy.sparse as sps
import scipy.sparse.linalg as sla
import scipy.linalg as sl
import qiskit.extensions.unitary as qeu
from qiskit.quantum_info.operators import Operator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import noise
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer, execute, IBMQ
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


dev_name = 'ibmq_santiago'
device, nm, bg, cm = qc_noise_model(dev_name)

# ==================================== Plotting Helpers ============================================================ >
colors = ['g', 'k', 'maroon', 'mediumblue', 'slateblue', 'limegreen', 'b', 'r', 'olive']
custom_cycler = (cycler(color=colors) + cycler(lw=[2] * len(colors)))
matplotlib.rcParams['figure.figsize'] = [10, 10]


def set_up_axes(rows, cols):
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
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_prop_cycle(custom_cycler)
        ax2.set_prop_cycle(custom_cycler)
        ax1.margins(x=0)
        ax2.margins(x=0)
        fig.subplots_adjust(hspace=0)
        fig.tight_layout(pad=1.5)
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


def two_point_correlations_plotter(alpha, beta, j, dt, pairs, data_one, data_two, data_cl):
    # todo out alpha, beta
    real_one, imag_one, gates_one = data_one[0], data_one[1], data_one[2]
    real_two, imag_two, gates_two = data_two[0], data_two[1], data_two[2]
    real_one_id, imag_one_id = real_one[0], imag_one[0]
    real_one_noise, imag_one_noise = real_one[1], imag_one[1]
    real_two_id, imag_two_id = real_two[0], imag_two[0]
    real_two_noise, imag_two_noise = real_two[1], imag_two[1]
    real_cl_data, im_cl_data = data_cl[0], data_cl[1]
    re_label = r'$Re \langle S_{\alpha}^{i}(t)S_{\beta}^{j}(0)\rangle $'
    im_label = r'$Im \langle S_{\alpha}^{i}(t)S_{\beta}^{j}(0)\rangle $'

    fig, axs = set_up_axes(4, 2)
    real1, im1, real2, im2, gate1, gate2 = axs[0, 0], axs[1, 0], axs[2, 0], axs[3, 0], axs[2, 1], axs[3, 1]
    real_cl, im_cl = axs[0, 1], axs[1, 1]

    scaler = abs(j) * dt
    p = [i * scaler for i in range(len(real_one_id.toarray()[0][:].tolist()))]
    for x in pairs:
        dx = pairs.index(x)
        real1.plot(p, real_one_id.toarray()[dx][:].tolist(), label=str(x))
        real1.plot(p, real_one_noise.toarray()[dx][:].tolist(), label=str(x), linestyle=":")
        real2.plot(p, real_two_id.toarray()[dx][:].tolist(), label=str(x))
        real2.plot(p, real_two_noise.toarray()[dx][:].tolist(), label=str(x), linestyle=":")
        real_cl.plot(p, real_cl_data.toarray()[dx][:].tolist(), label=str(x))
        im_cl.plot(p, im_cl_data.toarray()[dx][:].tolist(), label=str(x))
        im1.plot(p, imag_one_id.toarray()[dx][:].tolist(), label=str(x))
        im1.plot(p, imag_one_noise.toarray()[dx][:].tolist(), label=str(x), linestyle=":")
        im2.plot(p, imag_two_id.toarray()[dx][:].tolist(), label=str(x))
        im2.plot(p, imag_two_noise.toarray()[dx][:].tolist(), label=str(x), linestyle=":")
    gate1.plot(p, gates_one.toarray()[0, :].tolist())
    gate2.plot(p, gates_two.toarray()[0, :].tolist())
    gate1.set_ylabel("Gates (First)", fontsize="small")
    gate2.set_ylabel("Gates (Second)", fontsize="small")

    plot_dataset_byrows(real1, "Site Pairs", re_label, "Jt (First-Order Trotter)")
    plot_dataset_byrows(real2, "Site Pairs", re_label, "Jt (Second-Order Trotter)")
    plot_dataset_byrows(im1, "Site Pairs", im_label, "Jt (First-Order Trotter)")
    plot_dataset_byrows(im2, "Site Pairs", im_label, "Jt (Second-Order Trotter)")
    plot_dataset_byrows(real_cl, "Site Pairs", re_label, "Jt (Exact)")
    plot_dataset_byrows(im_cl, "Site Pairs", im_label, "Jt (Exact)")

    fig.suptitle('Two-Point Correlations')
    plt.show()


def dyn_spin_susceptibility_plotter(data_cl, data_q, w, k):
    matplotlib.rcParams['figure.figsize'] = [5, 5]
    fig, (ax1, ax2) = set_up_axes_two(2)
    surf = ax1.contourf(k, w, data_cl, 100, cmap="magma")
    fig.colorbar(surf)
    ax1.set_title(r'$|S(q, \omega)|^2 (Exact) $', size='medium')
    ax1.set_xlabel("q", size='medium')
    ax1.set_ylabel(r'$\omega (J)$', size='medium')

    surf2 = ax2.contourf(k, w, data_q, 100, cmap="magma")
    ax2.set_title(r'$|S(q, \omega)|^2 (Second - Order Trotter) $', size='medium')
    ax2.set_xlabel("q", size='medium')
    ax2.set_ylabel(r'$\omega (J)$', size='medium')

    plt.xticks(fontsize='medium')
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
                s1, s2 = spin_op(op, y[0], self.n, self.unity), spin_op(op, y[1], self.n, self.unity)
                self.hamiltonian += s1.dot(s2) * self.j * multipliers[dex]
        for z in nn_neighbors:
            for op in ops:
                dex = ops.index(op)
                s1, s2 = spin_op(op, z[0], self.n, self.unity), spin_op(op, z[1], self.n, self.unity)
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

    def two_point_correlations_c(self, total_time, dt, psi0, op_order, pairs=[]):

        dyn_data_real, dyn_data_imag = gen_m(len(pairs), total_time), gen_m(len(pairs), total_time)

        # Account for differences between examples from papers
        pseudo_constant = 1
        if self.paper in ['tachinno']:
            pseudo_constant = 4

        for t in range(total_time):
            t_ = t
            u = sl.expm(self.hamiltonian * dt * t_ * (1j))
            u_dag = sl.expm(self.hamiltonian * dt * t_ * (-1j))
            psi_dag = np.conj(psi0).transpose()
            for x in range(len(pairs)):
                si = spin_op(op_order[0], pairs[x][0], self.n, self.unity)
                sj = spin_op(op_order[1], pairs[x][1], self.n, self.unity)
                ket = u_dag.dot(si.dot(u.dot(sj.dot(psi0))))
                res = psi_dag.dot(ket).toarray()[0][0]
                dyn_data_real[x, t] = np.real(res) / pseudo_constant
                dyn_data_imag[x, t] = np.imag(res) / pseudo_constant

        return dyn_data_real, dyn_data_imag

    def dynamic_spin_susceptibility(self, total_time, dt, psi0, op_order, k_range, w_range):

        k_min, k_max, w_min, w_max = k_range[0], k_range[1], w_range[0], w_range[1]
        res, pairs = 300, []
        k_, w_ = np.arange(k_min, k_max, (k_max - k_min) / res), np.arange(w_min, w_max, (w_max - w_min) / res)
        k = np.array(k_.copy().tolist() * res).reshape(res, res).astype(complex)
        w = np.array(w_.copy().tolist() * res).reshape(res, res).T.astype(complex)

        for j in range(self.n):
            pairs.append((j, 0))

        tpc_real, tpc_imag = self.two_point_correlations_c(total_time, dt, psi0, op_order, pairs=pairs)
        tpc = tpc_real + 1j * tpc_imag

        dsf = np.zeros_like(k).astype(complex)
        # dsf will be complex since the operator s_a(t)s_b(0) will not be hermitian
        count = 0
        for j in range(self.n):
            exp_a = (np.exp(-1j * k * abs(j))) / self.n
            time_sum = np.zeros_like(w).astype(complex)
            for t in range(total_time):
                tpc_i = tpc[count, t]
                exp_b = np.exp(-1j * w * dt * t) # had to make this negative. is this because of w(J) ?
                time_sum += tpc_i * exp_b * dt
            count += 1
            dsf += exp_a * time_sum

        dsf_dagger = np.conj(dsf)
        spin_susceptibility = numpy.multiply(dsf_dagger, dsf)
        return spin_susceptibility, w, k


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
        qc.rx(-1 * math.pi / 2, 0)


def zz_operation(qc, a, b, delta):
    # Helper for time evolution operator
    qc.cx(a, b)
    qc.rz(2 * delta, b)
    qc.cx(a, b)

def three_cnot_evolution(qc, a, b, ancilla, j, t, dt, j2, trotter_steps, c, ising, a_constant, boundary):
    # Time evolving the system termwise by the elements in the hamiltonian
    nnn_term = j2
    if (abs(a - b) == 1 or boundary == 1):
        nnn_term = 1

    a_, b_ = a + ancilla, b + ancilla
    if ising:
        zz_operation(qc, a_, b_, j * t * dt * nnn_term / (c * trotter_steps))

    else:
        delta = j * t * dt * nnn_term / (c * trotter_steps)
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


# =================================================== Qiskit Simulations ============================================= >

class QuantumSim:

    def __init__(self, j=0.0, bg=0.0, a=1.0, n=0, j2=1, choice=False, open_chain=True, transverse=False, paper='',
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

                three_cnot_evolution(qc, x[0], x[1], ancilla, self.j, t, dt, self.j2, trotter_steps * pseudo_constant_a,
                                     1.0, self.ising, self.a, boundary)
                num_gates = num_gates + self.update_gates()

        return num_gates

    def second_order_trotter(self, qc, dt, t, ancilla):

        trotter_steps = 1
        if self.h_commutes:
            print("H commutes, trotter_steps = 1")
        else:
            t_ = t
            if t == 0:
                t_ += 1
                print("Second order trotter in progress..")
            trotter_steps = math.ceil(abs(self.j) * t_ * dt / self.epsilon)
            print('trotter steps:', trotter_steps, " t:", t)

        num_gates = 0

        # Address needed constants for particular paper
        pseudo_constant_a = 1.0
        mag_constant = 1.0
        if self.paper in ['joel']:
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

            for x in reversed(self.total_pairs[1:]):
                boundary = 0
                if x[0] == 0 and x[1] == self.n - 1 and self.open_chain == False:
                    boundary += 1
                three_cnot_evolution(qc, x[0], x[1], ancilla, self.j, t, dt, self.j2, trotter_steps * pseudo_constant_a,
                                     2.0, self.ising, self.a, boundary)
                num_gates = num_gates + self.update_gates()

            mid = self.total_pairs[0]
            three_cnot_evolution(qc, mid[0], mid[1], ancilla, self.j, t, dt, self.j2, trotter_steps * pseudo_constant_a,
                                 1.0, self.ising, self.a, False)
            num_gates = num_gates + self.update_gates()

            for x in self.total_pairs[1:]:
                boundary = 0
                if x[0] == 0 and x[1] == self.n - 1 and self.open_chain == False:
                    boundary += 1
                three_cnot_evolution(qc, x[0], x[1], ancilla, self.j, t, dt, self.j2, trotter_steps * pseudo_constant_a,
                                     2.0, self.ising, self.a, boundary)
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
                # noisy simulation
                result = execute(qc, backend=simulator, shots=50000, noise_model=nm, basis_gates=bg).result()
                counts = [result.get_counts(i) for i in range(len(result.results))]
                probs = sort_counts(counts[0], anc, 50000)
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
                # noisy simulation
                result = execute(qc, backend=simulator, shots=50000, noise_model=nm, basis_gates=bg).result()
                counts = [result.get_counts(i) for i in range(len(result.results))]
                measurements = sort_counts(counts[0], self.n, 50000)
                return measurements

    def two_point_correlations_q(self, trotter_alg, total_t, dt, alpha, beta, chosen_pairs, initialstate=0):
        # --------- #
        data_real_id, data_imag_id = gen_m(len(chosen_pairs), total_t), gen_m(len(chosen_pairs), total_t)
        data_real_noise, data_imag_noise = gen_m(len(chosen_pairs), total_t), gen_m(len(chosen_pairs), total_t)
        data_gates = gen_m(1, total_t)

        # Address needed constants for particular paper
        pseudo_constant = 1.0
        if self.paper in ['joel', 'tachinno']:
            pseudo_constant = 4.0

        for pair in chosen_pairs:
            for t in range(total_t):
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

    def dyn_spin_susceptibility_q(self, total_time, dt, psi0, op_order, k_range, w_range):

        k_min, k_max, w_min, w_max = k_range[0], k_range[1], w_range[0], w_range[1]
        res, pairs = 300, []
        k_, w_ = np.arange(k_min, k_max, (k_max - k_min) / res), np.arange(w_min, w_max, (w_max - w_min) / res)
        k = np.array(k_.copy().tolist() * res).reshape(res, res).astype(complex)
        w = np.array(w_.copy().tolist() * res).reshape(res, res).T.astype(complex)

        for j in range(self.n):
            pairs.append((j, 0))
        tpc_real, tpc_imag, gates = self.two_point_correlations_q(self.second_order_trotter, total_time, dt,
                                                                  op_order[0], op_order[1], pairs, initialstate=psi0)
        # using only the noisy data
        tpc_real = tpc_real[1]
        tpc_imag = tpc_imag[1]
        tpc = tpc_real + 1j * tpc_imag

        dsf = np.zeros_like(k).astype(complex)
        # dsf will be complex since the operator s_a(t)s_b(0) will not be hermitian
        count = 0
        for j in range(self.n):
            exp_a = (np.exp(-1j * k * abs(j))) / self.n
            time_sum = np.zeros_like(w).astype(complex)
            for t in range(total_time):
                tpc_i = tpc[count, t]
                exp_b = np.exp(-1j * w * dt * t) # had to make this negative. is this because of w(J) ?
                time_sum += tpc_i * exp_b * dt
            count += 1
            dsf += exp_a * time_sum

        dsf_dagger = np.conj(dsf)
        spin_susceptibility = numpy.multiply(dsf_dagger, dsf)
        return spin_susceptibility, w, k
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
        self.second = self.quantum_chain.second_order_trotter
        # --------- #


    def two_point_correlations(self, op_order='', total_t=0, dt=0, pairs=[], initstate=0):
        # --------- #
        alpha, beta = op_order[0], op_order[1]
        psi0 = init_spin_state(initstate, self.classical_chain.states)
        data_real_one, data_imag_one, data_gates_one = self.quantum_chain.two_point_correlations_q(self.first, total_t,
                                                                                                   dt, alpha, beta,
                                                                                                   pairs,
                                                                                                   initialstate=initstate)
        data_real_two, data_imag_two, data_gates_two = self.quantum_chain.two_point_correlations_q(self.second, total_t,
                                                                                                   dt, alpha, beta,
                                                                                                   pairs,
                                                                                                   initialstate=initstate)
        data_real_cl, data_imag_cl = self.classical_chain.two_point_correlations_c(total_t, dt, psi0, op_order,
                                                                                   pairs=pairs)
        j_ = self.classical_chain.j
        d_one, d_two = [data_real_one, data_imag_one, data_gates_one], [data_real_two, data_imag_two, data_gates_two]
        data_cl = [data_real_cl, data_imag_cl]
        two_point_correlations_plotter(alpha, beta, j_, dt, pairs, d_one, d_two, data_cl)
        # --------- #

    def dyn_spin_susceptibility(self, op_order='', total_t=0, dt=0, initstate=0, k_range=[], w_range=[]):

        psi0 = init_spin_state(initstate, self.classical_chain.states)
        data_cl, w, k = self.classical_chain.dynamic_spin_susceptibility(total_t, dt, psi0,
                                                                   op_order, k_range, w_range)
        data_q, w_, k_ = self.quantum_chain.dyn_spin_susceptibility_q(total_t, dt, initstate, op_order,
                                                                        k_range, w_range)
        dyn_spin_susceptibility_plotter(data_cl, data_q, w, k)

# ==================================== TESTERS ==================================================================== >

# --------------------------------------------------------------------------------------------------------------------------
# TACCHINO FIG 7 (Two Point Correlations)

#model = HeisenbergModel(j=-1, bg=20, a=1, n=3, j2=1, choice=False, open_chain=True, transverse=False, paper='tachinno',
#                        ising=False, epsilon=0.2, unity=True)
# auto, nearest, next-nearest two-point correlations
#model.two_point_correlations(op_order='xx', total_t=330, dt=.01, pairs=[(0, 0)], initstate=int('000', 2))
#model.two_point_correlations(op_order='xx', total_t=330, dt=.01, pairs=[(1, 0)], initstate=int('000', 2))
#model.two_point_correlations(op_order='xx', total_t=330, dt=.01, pairs=[(2, 0)], initstate=int('000', 2))

#model.dyn_spin_susceptibility(op_order='xx', total_t=4000, dt=.01, initstate=0, k_range=[-math.pi, math.pi], w_range=[-4, 16])


# ----------------------------------------------------------------------------------------------------------------------

#model = HeisenbergModel(j=-.84746, j2=1, bg=0, a=1, n=4, open_chain=False, unity=True, choice=False, paper='francis', epsilon=0.1)
#model.two_point_correlations(self, op_order='xx', total_t=600, dt=.01, pairs=[(0, 0), (1, 0), (2, 0)], initstate=0)
#model.dyn_spin_susceptibility(op_order='xx', total_t=600, dt=.01, initstate=0, k_range=[-math.pi, math.pi], w_range=[-4, 16])

# same example with twice as many sites

model = HeisenbergModel(j=-.84746, j2=1, bg=0, a=1, n=8, open_chain=False, unity=True, choice=False, paper='francis', epsilon=0.1)
#model.two_point_correlations(self, op_order='xx', total_t=600, dt=.01, pairs=[(0, 0), (1, 0), (2, 0)], initstate=0)
model.dyn_spin_susceptibility(op_order='xx', total_t=600, dt=.01, initstate=0, k_range=[-math.pi, math.pi], w_range=[-4, 16])
