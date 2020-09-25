# TwoDimHeisenberg.py

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
#colors = ['b', 'r', 'olive']

custom_cycler = (cycler(color=colors) + cycler(lw=[2] * len(colors)))
matplotlib.rcParams['figure.figsize'] = [10, 10]
plt.rc('font', size=12)
plt.rc('axes', titlesize=12)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('legend', fontsize=12)
plt.rc('figure', titlesize=12)


def set_up_axes_one(num_axes):
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



def set_up_axes_two(rows, cols):
    fig, axs = plt.subplots(rows, cols)
    for col in range(cols):
        for row in range(rows):
            ax = axs[row, col]
            ax.set_prop_cycle(custom_cycler)
            ax.margins(x=0)
    fig.tight_layout(pad=3.5)
    return fig, axs


def occ_plotter(chosen_states, j, n, total_time, dt, data_two, data_cl):

    data_two_id, data_two_noise = data_two[0], data_two[1]

    fig, axs = set_up_axes_one(2)
    x1 = [i * abs(j) * dt for i in range(total_time)]
    two, cl = axs[0], axs[1]
    for state in chosen_states:
        dex = chosen_states.index(state)
        label = np.binary_repr(state).zfill(n)
        two.plot(x1, data_two_id[dex, :].toarray().tolist()[0], label=label, linestyle="-")
        two.plot(x1, data_two_noise[dex, :].toarray().tolist()[0], label=label, linestyle=":")
        cl.plot(x1, data_cl[dex, :].toarray().tolist()[0], label=label)

    plot_dataset_byrows(two, "States", "Probability", 'Jt (Second-Order Trotter)')
    plot_dataset_byrows(cl, "States", "Probability", 'Jt (Exact)')
    fig.suptitle('Occupation Probabilities')
    plt.show()


def two_point_correlations_plotter(alpha, beta, j, dt, pairs, data_two, data_cl):
    real_two, imag_two = data_two[0], data_two[1]
    real_two_id, imag_two_id = real_two[0], imag_two[0]
    real_two_noise, imag_two_noise = real_two[1], imag_two[1]
    real_cl_data, im_cl_data = data_cl[0], data_cl[1]
    re_label = r'$Re \langle S_{\alpha}^{i}(t)S_{\beta}^{j}(0)\rangle $'
    im_label = r'$Im \langle S_{\alpha}^{i}(t)S_{\beta}^{j}(0)\rangle $'

    colors = ['g', 'k', 'maroon', 'mediumblue', 'slateblue', 'limegreen', 'b', 'r', 'olive']

    fig, axs = set_up_axes_two(2, 2)
    real2, im2 = axs[0, 0], axs[1, 0]
    real_cl, im_cl = axs[0, 1], axs[1, 1]
    scaler = abs(j) * dt
    p = [i * scaler for i in range(len(real_two_id.toarray()[0][:].tolist()))]
    pairsdex = 0
    for x in pairs:
        dx = pairs.index(x)
        real2.plot(p, real_two_id.toarray()[dx][:].tolist(), label=str(x), color=colors[pairsdex])
        real2.plot(p, real_two_noise.toarray()[dx][:].tolist(), label=str(x), linestyle=":", color=colors[pairsdex])
        real_cl.plot(p, real_cl_data.toarray()[dx][:].tolist(), label=str(x), color=colors[pairsdex])
        im_cl.plot(p, im_cl_data.toarray()[dx][:].tolist(), label=str(x), color=colors[pairsdex])
        im2.plot(p, imag_two_id.toarray()[dx][:].tolist(), label=str(x), color=colors[pairsdex])
        im2.plot(p, imag_two_noise.toarray()[dx][:].tolist(), label=str(x), linestyle=":", color=colors[pairsdex])
        pairsdex += 1

    plot_dataset_byrows(real2, "Site Pairs", re_label, "Jt (Second-Order Trotter)")
    plot_dataset_byrows(im2, "Site Pairs", im_label, "Jt (Second-Order Trotter)")
    plot_dataset_byrows(real_cl, "Site Pairs", re_label, "Jt (Exact)")
    plot_dataset_byrows(im_cl, "Site Pairs", im_label, "Jt (Exact)")

    fig.suptitle('Two-Point Correlations')
    plt.show()


def total_magnetization_plotter(j, total_t, dt, data_two, data_cl):
    data_two_id, data_two_noise = data_two[0], data_two[1]

    fig, axs = set_up_axes_one(2)
    x1 = [i * j * dt for i in range(total_t)]
    two, cl = axs[0], axs[1]
    two.plot(x1, data_two_id.toarray()[0][:].tolist(), linestyle='-')
    two.plot(x1, data_two_noise.toarray()[0][:].tolist(), linestyle=":")
    cl.plot(x1, data_cl.toarray()[0][:].tolist())

    two.set_ylabel('Total Magnetization')
    two.set_xlabel('Jt (Second-Order Trotter)')
    cl.set_ylabel('Total Magnetization')
    cl.set_xlabel('Jt (Exact)')
    fig.suptitle('Total Magnetization')
    plt.show()


def all_site_magnetization_plotter(n, j, dt, total_t, data_two, data_cl):
    # all_site_magnetization_plotter(n, j, dt, total_t, data_two, data_cl)
    data_two_id, data_two_noise = data_two[0], data_two[1]

    fig, axs = set_up_axes_one(2)
    x1 = [i * j * dt for i in range(total_t)]
    two, cl = axs[0], axs[1]
    for site in range(n):
        two.plot(x1, data_two_id.toarray()[site][:].tolist(), linestyle='-', label=site)
        two.plot(x1, data_two_noise.toarray()[site][:].tolist(), linestyle=":", label=site)
        cl.plot(x1, data_cl.toarray()[site][:].tolist(), label=site)

    plot_dataset_byrows(two, 'Sites', 'Magnetization', 'Jt (Second-Order Trotter)')
    plot_dataset_byrows(cl, 'Sites', 'Magnetization', 'Jt (Exact)')
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


def spin_op(operator, site, n):
    array = identity
    ops_list = ['x', 'y', 'z', '+', '-']
    pauli_ops = [sx, sy, sz, plus, minus]
    if site == (n - 1):
        array = pauli_ops[ops_list.index(operator)]
    for x in reversed(range(0, n - 1)):
        if x == site:
            array = sps.kron(array, pauli_ops[ops_list.index(operator)])
        else:
            array = sps.kron(array, identity)

    return array


def gen_pairs(n, l_type):
    nn, autos = [], []
    for p in range(n):
        autos.append((p, p))
    if l_type == "diamond":
        nn = nn + [(0, 1), (0, 3), (0, 5), (0, 7), (1, 2), (1, 4), (1, 6), (2, 3), (2, 5), (2, 7), 
                  (3, 4), (3, 6), (4, 5), (4, 7), (5, 6), (6, 7)]
    elif l_type == "cube": 
        nn = nn + [(0, 1), (0, 2), (0, 4), (1, 5), (1, 3), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), 
                  (6, 7)]
    return nn, autos


def gen_m(leng, steps):
    # Generate an empty data matrix
    return sps.lil_matrix(np.zeros([leng, steps]), dtype=complex)


def plot_dataset_byrows(ax, legend_title, y_label, x_label):
    ax.legend(loc='right', title=legend_title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)


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

    def __init__(self, j=0.0, bg=0.0, a=0.0, ising=False, transverse=False, l_type="diamond"):

        types = ["diamond", "cube"]
        sites = [8, 8]

        self.j = j  # ===============================================# coupling constant
        self.bg = bg  # =============================================# magnetic field strength
        self.a = a  # ===============================================# anisotropy constant jz/j
        self.n = sites[types.index(l_type)]  # ======================# sites
        self.states = 2 ** self.n  # ================================# number of basis states
        self.ising = ising  # =======================================# for ising model
        self.transverse = transverse  # =============================# whether transverse field ising model
        self.l_type = l_type # ======================================# lattice type
        self.hamiltonian = sps.lil_matrix(np.zeros([self.states, self.states], complex))
        neighbors, autos = gen_pairs(self.n, self.l_type)
        multipliers = [1, 1, self.a]

        ops = ['x', 'y', 'z']
        if ising:
            ops = ['z']
            multipliers = [self.a]

        for y in neighbors:
            for op in ops:
                dex = ops.index(op)
                s1, s2 = spin_op(op, y[1], self.n), spin_op(op, y[0], self.n)
                self.hamiltonian += s1.dot(s2) * self.j * multipliers[dex]
        for x in range(self.n):
            if self.bg != 0 and self.transverse == False:
                s_final = spin_op('z', x, self.n) * self.bg / 2
                self.hamiltonian += s_final
            elif self.bg != 0 and self.transverse == True:
                s_final = spin_op('x', x, self.n) * self.bg / 2
                self.hamiltonian += s_final

        self.hamiltonian = sps.csc_matrix(self.hamiltonian)

    def test_commuting_matrices(self):
        ops, multipliers = ['x', 'y', 'z'], [1, 1, self.a]
        neighbors, autos = gen_pairs(self.n, self.l_type) 
        if self.ising:
            ops = ['z']
            multipliers = [self.a]

        n, dex_terms = self.n, 0
        even_terms = sps.lil_matrix(np.zeros([2 ** n, 2 ** n], complex))
        odd_terms = sps.lil_matrix(np.zeros([2 ** n, 2 ** n], complex))
        
        for y in neighbors:
            for op in ops:
                dex = ops.index(op)
                s1, s2 = spin_op(op, y[1], self.n), spin_op(op, y[0], self.n)
                if dex_terms % 2 == 0:
                    even_terms += s1.dot(s2) * self.j * multipliers[dex]
                else:
                    odd_terms += s1.dot(s2) * self.j * multipliers[dex]
            dex_terms += 1

        commuting = True
        if not commutes(even_terms, odd_terms):
            commuting = False
        terms = even_terms + odd_terms
        for x in range(self.n):
            if self.bg != 0 and self.transverse == False:
                s_final = spin_op('z', x, self.n) * self.bg / 2
                if not commutes(terms, s_final):
                    commuting = False
            elif self.bg != 0 and self.transverse == True:
                s_final = spin_op('x', x, self.n) * self.bg / 2
                if not commutes(terms, s_final):
                    commuting = False

        return commuting

    def eig(self):
        h = self.hamiltonian.toarray()
        evals, evect = sl.eigh(h)[0], sl.eigh(h)[1]
        return evals, sps.csc_matrix(evect)

    def two_point_correlations_c(self, total_time, dt, psi0, op_order, pairs=[]):

        if pairs == []:
            # if wanting to generate all pairs, feed in none
            nn, auto = gen_pairs(self.n, True)
            pairs = nn + auto
        dyn_data_real, dyn_data_imag = gen_m(len(pairs), total_time), gen_m(len(pairs), total_time)

        for t in range(total_time):
            u = sl.expm(self.hamiltonian * dt * t * (-1j))
            u_dag = sl.expm(self.hamiltonian * dt * t * (1j))
            psi_dag = np.conj(psi0).transpose()
            for x in range(len(pairs)):
                si = spin_op(op_order[1], pairs[x][0], self.n)
                sj = spin_op(op_order[0], pairs[x][1], self.n)
                ket = u_dag.dot(si.dot(u.dot(sj.dot(psi0))))
                res = psi_dag.dot(ket).toarray()[0][0]
                dyn_data_real[x, t] = np.real(res)
                dyn_data_imag[x, t] = np.imag(res)
        return dyn_data_real, dyn_data_imag

    def occupation_probabilities_c(self, total_time=0, dt=0.0, initialstate=None, chosen_states=[]):

        psi0 = initialstate
        basis_matrix = sps.csc_matrix(np.eye(self.states))
        data = gen_m(len(chosen_states), total_time)
        for t in range(total_time):
            u = sl.expm(self.hamiltonian * dt * t * (-1j))
            psi_t = u.dot(psi0)
            index_ = 0
            for i in chosen_states:
                basis_bra = (basis_matrix[i, :])
                probability = (basis_bra.dot(psi_t)).toarray()[0][0]
                probability = ((np.conj(psi_t).transpose()).dot(np.conj(basis_bra).transpose())).toarray()[0][0] * probability
                data[index_, t] = probability
                index_ += 1
        return data

    def magnetization_per_site_c(self, total_time, dt, psi0, site):

        data = gen_m(1, total_time)
        for t in range(total_time):
            u = sl.expm(self.hamiltonian * dt * t * (-1j))
            bra = np.conj(u.dot(psi0).transpose())
            s_z = spin_op('z', site, self.n)
            ket = s_z.dot(u.dot(psi0))
            data[0, t] += (bra.dot(ket).toarray()[0][0])
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


def three_cnot_evolution(qc, a, b, ancilla, j, t, dt, trotter_steps, c, ising, a_constant):
    # Time evolving the system termwise by the elements in the hamiltonian
    
    

    a_, b_ = a + ancilla, b + ancilla
    if ising:
        zz_operation(qc, a_, b_, j * a_constant * t * dt / (c * trotter_steps))

    else:
        delta = j * t * dt  / (c * trotter_steps)
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

    def __init__(self, j=0.0, bg=0.0, a=1.0, transverse=False, ising=False, epsilon=0, l_type='diamond'):

        types, sites = ["diamond", "cube"], [8, 8]

        self.j = j  # ================================================= # coupling constant
        self.bg = bg  # ================================================# magnetic field strength
        self.n = sites[types.index(l_type)]  # =========================# number of spins
        self.states = 2 ** self.n  # ========================================# number of basis states
        self.a = a  # ==================================================# anisotropy jz / j
        self.transverse = transverse  # ================================# for transverse field ising model
        self.ising = ising  # ==========================================# Whether using the ising model
        self.epsilon = epsilon  # ======================================# Desired precision - use to find trotter steps
        self.l_type = l_type

        self.h_commutes = ClassicalSpinChain(j=self.j, bg=self.bg, a=self.a, l_type=self.l_type, 
                                             transverse=self.transverse, ising=self.ising).test_commuting_matrices()
        self.pairs_nn, autos = gen_pairs(self.n, self.l_type)  
        self.total_pairs = self.pairs_nn

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

    def second_order_trotter(self, qc, dt, t, ancilla):
        trotter_steps = 1
        if self.h_commutes:
            print("H commutes, trotter_steps = 1")
        else:
            t_ = t
            if t == 0:
                t_ = 1
                print("Second order trotter in progress..")
            trotter_steps = math.ceil(abs(self.j) * t_ * dt / self.epsilon)
            print('trotter steps:', trotter_steps, " t:", t)

        num_gates = 0

        for step in range(trotter_steps):
            for k in range(self.n):
                if self.bg != 0.0:
                    if self.transverse:
                        qc.rx(self.bg * dt * t / trotter_steps, k + ancilla)
                    else:
                        qc.rz(self.bg * dt * t / trotter_steps, k + ancilla)
                    num_gates = num_gates + 1

            for x in reversed(self.total_pairs[1:]):
                three_cnot_evolution(qc, x[0], x[1], ancilla, self.j, t, dt, trotter_steps,
                                     2.0, self.ising, self.a)
                num_gates = num_gates + self.update_gates()
            mid = self.total_pairs[0]
            three_cnot_evolution(qc, mid[0], mid[1], ancilla, self.j, t, dt, trotter_steps,
                                 1.0, self.ising, self.a)
            num_gates = num_gates + self.update_gates()
            for x in self.total_pairs[1:]:
                three_cnot_evolution(qc, x[0], x[1], ancilla, self.j, t, dt, trotter_steps,
                                     2.0, self.ising, self.a)
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

    def magnetization_per_site_q(self, t, dt, site, initialstate, trotter_alg):
        # --------- #
        qc_id, qc_noise = QuantumCircuit(self.n + 1, 1), QuantumCircuit(self.n + 1, 1)

        self.init_state(qc_id, True, initialstate)
        self.init_state(qc_noise, True, initialstate)
        qc_id.h(0)
        qc_noise.h(0)
   
        num_gates_id = trotter_alg(qc_id, dt, t, 1)
        num_gates_noise = trotter_alg(qc_noise, dt, t, 1)
        num_gates = num_gates_noise

        choose_control_gate('z', qc_id, 0, site + 1)
        choose_control_gate('z', qc_noise, 0, site + 1)
        qc_id.h(0)
        qc_noise.h(0)

        measurement_id = self.run_circuit(1, qc_id, False)
        measurement_noise = self.run_circuit(1, qc_noise, True)

        return measurement_id, measurement_noise, num_gates
        # --------- #

    def all_site_magnetization_q(self, trotter_alg, total_time=0, dt=0.0, initialstate=0):
        # --------- #
        data_id = gen_m(self.n, total_time)
        data_noise = gen_m(self.n, total_time)
        data_gates = gen_m(1, total_time)
   
        for t in range(total_time):
            num_gates_total = 0
            for site in range(self.n):
                m_id, m_noise, num_gates = self.magnetization_per_site_q(t, dt, site, initialstate, trotter_alg)
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

        for t in range(total_time):
            total_magnetization_id = 0
            total_magnetization_noise = 0
            num_gates_total = 0
            for site in range(self.n):
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

    def two_point_correlations_q(self, trotter_alg, total_t, dt, alpha, beta, chosen_pairs, initialstate=0):
        # --------- #
        data_real_id, data_imag_id = gen_m(len(chosen_pairs), total_t), gen_m(len(chosen_pairs), total_t)
        data_real_noise, data_imag_noise = gen_m(len(chosen_pairs), total_t), gen_m(len(chosen_pairs), total_t)
        data_gates = gen_m(1, total_t)

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
                    measurement_id = self.run_circuit(1, qc_id, False)

                    choose_control_gate(beta, qc_noise, 0, pair[1] + 1)
                    num_gates_noise = trotter_alg(qc_noise, dt, t, 1)
                    choose_control_gate(alpha, qc_noise, 0, pair[0] + 1)
                    real_or_imag_measurement(qc_noise, j)
                    measurement_noise = self.run_circuit(1, qc_noise, True)

                    num_gates = num_gates_noise
                    print(num_gates)
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

        for t in range(total_time):
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

    def __init__(self, j=1.0, bg=0.0, a=1.0, transverse=False, ising=False, epsilon=0, l_type="diamond"):
        # --------- #
        self.classical_chain = ClassicalSpinChain(j=j, bg=bg, a=a, ising=ising, transverse=transverse, l_type=l_type)
        self.quantum_chain = QuantumSim(j=j, bg=bg, a=a, transverse=transverse, ising=ising, epsilon=epsilon, l_type=l_type)

        self.second = self.quantum_chain.second_order_trotter
        # --------- #

    def all_site_magnetization(self, total_t=0, dt=0, initstate=0): # plotter done 
        # --------- #
        print("The code is doing stuff")
        num_states = self.classical_chain.states
        psi0 = init_spin_state(initstate, num_states)

        data_two, data_gates_two = self.quantum_chain.all_site_magnetization_q(self.second, total_time=total_t, dt=dt,
                                                                               initialstate=initstate)
        data_cl = self.classical_chain.all_site_magnetization_c(total_t, dt, psi0)
        n, j = self.classical_chain.n, self.classical_chain.j
        all_site_magnetization_plotter(n, j, dt, total_t, data_two, data_cl)
        # --------- #

    def total_magnetization(self, total_t=0, dt=0, initstate=0): # plotter done 
        # --------- #
        
        psi0 = init_spin_state(initstate, self.classical_chain.states)
        data_two, data_gates_two = self.quantum_chain.total_magnetization_q(self.first, total_t, dt,
                                                                            initialstate=initstate)
        data_cl = self.classical_chain.total_magnetization_c(total_t, dt, psi0)
        j_ = self.classical_chain.j
        total_magnetization_plotter(j_, total_t, dt, data_two, data_cl)
        # --------- #

    def two_point_correlations(self, op_order='', total_t=0, dt=0, pairs=[], initstate=0): # plotter done 
        # --------- #
        alpha, beta = op_order[0], op_order[1]
        psi0 = init_spin_state(initstate, self.classical_chain.states)
        data_real_two, data_imag_two, data_gates_two = self.quantum_chain.two_point_correlations_q(self.second, total_t,
                                                                                                   dt, alpha, beta,
                                                                                                   pairs,
                                                                                                   initialstate=initstate)
        data_real_cl, data_imag_cl = self.classical_chain.two_point_correlations_c(total_t, dt, psi0, op_order,
                                                                                   pairs=pairs)
        j_ = self.classical_chain.j
        d_two = [data_real_two, data_imag_two]
        data_cl = [data_real_cl, data_imag_cl]
        two_point_correlations_plotter(alpha, beta, j_, dt, pairs, d_two, data_cl)
        # --------- #

    def occupation_probabilities(self, total_t=0, dt=0, initstate=0, chosen_states=[]):
        # --------- #
        psi0 = init_spin_state(initstate, self.classical_chain.states)
        data_two, data_gates_two = self.quantum_chain.occupation_probabilities_q(self.second, total_t,
                                                                                 dt, initialstate=initstate,
                                                                                 chosen_states=chosen_states)
        data_cl = self.classical_chain.occupation_probabilities_c(total_t, dt, initialstate=psi0,
                                                                  chosen_states=chosen_states)
        n = self.classical_chain.n
        occ_plotter(chosen_states, self.classical_chain.j, n, total_t, dt, data_two, data_cl)
        # --------- #
        
model = HeisenbergModel(j=1, bg=0, a=0.5, epsilon=0.05, l_type="diamond") 
initstate=model.classical_chain.states - 2 
model.all_site_magnetization(total_t=200, dt=0.01, initstate=initstate)


