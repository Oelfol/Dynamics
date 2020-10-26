
# SpinChainsRM.py
# Code to compare readout-mitigated noisy results to noisy and exact results, and save all in csv files.
# Only incorporates ancilla-assisted measurements (no occupation probabilities)
# Code inherited from CompareTrotterAlgorithms.py. Only second order trotter formula is used, except for joel example.
# Only noisy data is used -- if need to check correctness, can change run argument.
# Removed: Next-nearest neighbors & equal-time correlations, since not in use
# October 2020


import numpy as np
import math
import warnings
import scipy.sparse as sps
import scipy.linalg as sl
from qiskit import QuantumCircuit, Aer, execute, IBMQ
import spinchainshelpers as sch
import IBUReadoutError as IBU
import ibmq_setup as ibmq
import joel_circuit as jc
warnings.filterwarnings('ignore')

shots = 50000
dev_name = 'ibmq_santiago'
setup = ibmq.ibmqSetup(sim=True, shots=shots, dev_name=dev_name)
device, nm, bg, cm = setup.get_noise_model()
simulator = setup.get_simulator()


# Initial Layouts for Joel example (minimize swap gates on ibmq_casablanca)
# Not fully implemented yet -- need more info to make sure ancilla is always qubit 2
il1 = [2, 1, 3, 0, 4, 5, 6]
il2 = [2, 0, 1, 3, 4, 5, 6]
il3 = [2, 5, 0, 1, 3, 4, 6]
il4 = [2, 4, 6, 0, 1, 3, 5]
il5 = [2, 4, 5, 6, 0, 1, 3]
il6 = [2, 3, 4, 5, 6, 0, 1]
initial_layouts = [il1, il2, il3, il4, il5, il6]

# Temp measure -- have to enable joel for that example
joel = True


def choose_RM(counts, num_qubits, RMfilename, RM=False):
    # choose whether to use readout mitigation
    # counts: counts directly after running the circuit
    # num_qubits: number of qubits measured
    if RM:
        probs = sch.sort_counts_no_div(counts[0], num_qubits)
        probs = IBU.unfold(RMfilename, shots, probs, num_qubits)
        return probs
    else:
        probs = sch.sort_counts(counts[0], num_qubits, shots)
        return probs


# ============================================ Classical Simulations ================================================= >


class ClassicalSpinChain:

    def __init__(self, j=0.0, j2=0.0, bg=0.0, a=0.0, n=0, open_chain=False, unity=False, choice=False, ising=False,
                 paper='', transverse=False):
        self.j = j  # ===============================================# coupling constant
        self.j2 = j2  # =============================================# how much to scale next-nearest coupling
        self.bg = bg  # =============================================# magnetic field strength
        self.a = a  # ===============================================# anisotropy constant jz/j
        self.n = n  # ===============================================# number of sites
        self.open_chain = open_chain  # =============================# whether open chain (bool)
        self.states = 2 ** n  # =====================================# number of basis states
        self.unity = unity  # =======================================# whether h-bar/2 == 1 (h-bar == 1 elsewise)
        self.choice = choice  # =====================================# whether next-nearest (bool)
        self.ising = ising  # =======================================# for ising model
        self.paper = paper  # =======================================# refer to settings for a specific paper
        self.transverse = transverse  # =============================# whether transverse field ising model

        # Generate Hamiltonian
        self.hamiltonian = sps.lil_matrix(np.zeros([2 ** n, 2 ** n], complex))
        neighbors, autos = sch.gen_pairs(self.n, False, self.open_chain)
        ops, multipliers = ['x', 'y', 'z'], [1, 1, self.a]
        if ising:
            ops, multipliers = ['z'], [self.a]

        for y in neighbors:
            for op in ops:
                s1, s2 = sch.spin_op(op, y[0], self.n, self.unity), sch.spin_op(op, y[1], self.n, self.unity)
                self.hamiltonian += s1.dot(s2) * self.j * multipliers[ops.index(op)]
        for x in range(self.n):
            if self.bg != 0 and self.transverse == False:
                self.hamiltonian += sch.spin_op('z', x, self.n, self.unity) * self.bg / 2
            elif self.bg != 0 and self.transverse == True:
                self.hamiltonian += sch.spin_op('x', x, self.n, self.unity) * self.bg / 2
        self.hamiltonian = sps.csc_matrix(self.hamiltonian)

    def test_commuting_matrices(self):
        ops, multipliers = ['x', 'y', 'z'], [1, 1, self.a]
        neighbors, autos = sch.gen_pairs(self.n, False, self.open_chain)
        if self.ising:
            ops, multipliers = ['z'], [self.a]

        n, dex_terms = self.n, 0
        even_terms = sps.lil_matrix(np.zeros([2 ** n, 2 ** n], complex))
        odd_terms = sps.lil_matrix(np.zeros([2 ** n, 2 ** n], complex))

        for y in neighbors:
            for op in ops:
                dex = ops.index(op)
                s1, s2 = sch.spin_op(op, y[1], self.n, self.unity), sch.spin_op(op, y[0], self.n, self.unity)
                if dex_terms % 2 == 0:
                    even_terms += s1.dot(s2) * self.j * multipliers[dex]
                else:
                    odd_terms += s1.dot(s2) * self.j * multipliers[dex]
            dex_terms += 1

        commuting = True
        if not sch.commutes(even_terms, odd_terms):
            commuting = False
        terms = even_terms + odd_terms
        for x in range(self.n):
            if self.bg != 0 and self.transverse == False:
                s_final = sch.spin_op('z', x, self.n, self.unity) * self.bg / 2
                if not sch.commutes(terms, s_final):
                    sch.commuting = False
            elif self.bg != 0 and self.transverse == True:
                s_final = sch.spin_op('x', x, self.n, self.unity) * self.bg / 2
                if not sch.commutes(terms, s_final):
                    commuting = False
        return commuting

    def two_point_correlations_c(self, total_time, dt, psi0, op_order, pairs=[]):
        if pairs == []:
            # if wanting to generate all pairs, feed in none
            nn, auto = sch.gen_pairs(self.n, True, self.open_chain)
            pairs = nn +  auto
        dyn_data_real, dyn_data_imag = sch.gen_m(len(pairs), total_time), sch.gen_m(len(pairs), total_time)

        pseudo_constant = 1
        if self.paper in ['tachinno']:
            pseudo_constant = 4

        for t in range(total_time):
            u, u_dag = sl.expm(self.hamiltonian * dt * t * (1j)), sl.expm(self.hamiltonian * dt * t * (-1j))
            psi_dag = np.conj(psi0).transpose()
            for x in range(len(pairs)):
                si = sch.spin_op(op_order[0], pairs[x][0], self.n, self.unity)
                sj = sch.spin_op(op_order[1], pairs[x][1], self.n, self.unity)
                ket = u_dag.dot(si.dot(u.dot(sj.dot(psi0))))
                res = psi_dag.dot(ket).toarray()[0][0]
                dyn_data_real[x, t] = np.real(res) / pseudo_constant
                dyn_data_imag[x, t] = np.imag(res) / pseudo_constant
        return dyn_data_real, dyn_data_imag

    def magnetization_per_site_c(self, total_time, dt, psi0, site):
        pseudo_constant, data = 1.0, sch.gen_m(1, total_time)
        if self.paper in ['tachinno']:
            pseudo_constant = 2.0
        for t in range(total_time):
            u = sl.expm(self.hamiltonian * dt * t * (-1j))
            bra = np.conj(u.dot(psi0).transpose())
            s_z = sch.spin_op('z', site, self.n, self.unity)
            ket = s_z.dot(u.dot(psi0))
            data[0, t] += (bra.dot(ket).toarray()[0][0]) / pseudo_constant
        return data

    def all_site_magnetization_c(self, total_time, dt, psi0):
        data = sch.gen_m(self.n, total_time)
        for site in range(self.n):
            site_data = self.magnetization_per_site_c(total_time, dt, psi0, site)
            data[site, :] += site_data
        return data

    def total_magnetization_c(self, total_time, dt, psi0):
        data = sch.gen_m(1, total_time)
        for site in range(self.n):
            site_data = self.magnetization_per_site_c(total_time, dt, psi0, site)
            data = data + site_data
        return data


# =================================================== Qiskit Simulations ============================================= >

class QuantumSim:

    def __init__(self, j=0.0, bg=0.0, a=1.0, n=0, open_chain=True, transverse=False, paper='',
                 ising=False, epsilon=0.0, RMfilename=''):
        self.j = j  # ================================================= # coupling constant
        self.bg = bg  # ================================================# magnetic field strength
        self.n = n  # ==================================================# number of spins
        self.states = 2 ** n  # ========================================# number of basis states
        self.a = a  # ==================================================# anisotropy jz / j
        self.open_chain = open_chain  # ================================# open chain = periodic boundary
        self.transverse = transverse  # ================================# for transverse field ising model
        self.paper = paper  # =========================# direct reference for all the unique aspects of different papers
        self.ising = ising  # ==========================================# Whether using the ising model
        self.unity = False  # ==========================================# Depends on paper
        self.epsilon = epsilon  # ======================================# Desired precision - use to find trotter steps
        if self.paper == 'francis':
            self.unity = True

        self.h_commutes = ClassicalSpinChain(j=self.j, bg=self.bg, a=self.a, n=self.n,
            open_chain=self.open_chain, unity=self.unity).test_commuting_matrices()
        self.total_pairs, autos = sch.gen_pairs(self.n, False, self.open_chain)
        self.RMfilename = RMfilename

    def init_state(self, qc, ancilla, initial_state):
        state_temp, anc, index = np.binary_repr(initial_state).zfill(self.n)[::-1], int(ancilla), 0
        for x in state_temp:
            if x == '1':
                qc.x(index + anc)
            index += 1

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

            for x in reversed(self.total_pairs[1:]):
                sch.three_cnot_evolution(qc, x, ancilla, self.j, t, dt, trotter_steps * pseudo_constant_a * 2.0,
                                         self.ising, self.a)
            mid = self.total_pairs[0]
            sch.three_cnot_evolution(qc, mid, ancilla, self.j, t, dt, trotter_steps * pseudo_constant_a,
                                 self.ising, self.a)
            for x in self.total_pairs[1:]:
                sch.three_cnot_evolution(qc, x, ancilla, self.j, t, dt, trotter_steps * pseudo_constant_a * 2.0,
                                     self.ising, self.a)

    def run_circuit(self, qc, site):
        qc.measure(0, 0)
        initial_layout = []
        if joel:
            initial_layout = initial_layouts[site]
        else:
            initial_layout = [2, 0, 1, 2, 3]
        result = execute(qc, backend=simulator, shots=shots, noise_model=nm, basis_gates=bg,
                         optimization_level=0, initial_layout=initial_layout).result()
        counts = [result.get_counts(i) for i in range(len(result.results))]
        probsRM = choose_RM(counts, 1, self.RMfilename, RM=True)
        probs = choose_RM(counts, 1, self.RMfilename, RM=False)
        return probsRM[0] - probsRM[1], probs[0] - probs[1]

    def magnetization_per_site_q(self, t, dt, site, initialstate, trotter_alg, hadamard=False):
        measurement_noise_rm, measurement_noise = 0, 0
        if joel:
            circuit = jc.magnetization(t, dt, site, initialstate, self.j, self.epsilon, self.a, self.n)
            measurement_noise_rm, measurement_noise = self.run_circuit(circuit, site)
        else:
            qc = QuantumCircuit(self.n + 1, 1)
            self.init_state(qc, True, initialstate)
            qc.h(0)
            if hadamard:  # -----> tachinno fig 5a
                qc.h(2)
            trotter_alg(qc, dt, t, 1)
            sch.choose_control_gate('z', qc, 0, site + 1)
            qc.h(0)
            measurement_noise_rm, measurement_noise = self.run_circuit(qc, site)
        pseudo_constant = 1
        if self.paper in ['joel', 'tachinno']:
            pseudo_constant = 2
        return measurement_noise_rm / pseudo_constant, measurement_noise / pseudo_constant

    def all_site_magnetization_q(self, trotter_alg, total_time=0, dt=0.0, initialstate=0, hadamard=False):
        data = sch.gen_m(self.n, total_time)
        data_rm = sch.gen_m(self.n, total_time)
        for t in range(total_time):
            for site in range(self.n):
                m_noise_rm, m_noise = self.magnetization_per_site_q(t, dt, site, initialstate, trotter_alg, hadamard=hadamard)
                data[site, t] += m_noise
                data_rm[site, t] += m_noise_rm
        return data_rm, data

    def total_magnetization_q(self, trotter_alg, total_time=0, dt=0.0, initialstate=0):
        data = sch.gen_m(1, total_time)
        data_rm = sch.gen_m(1, total_time)
        for t in range(total_time):
            total_magnetization_noise = 0
            total_magnetization_noise_rm = 0
            for site in range(self.n):
                m_noise_rm, m_noise = self.magnetization_per_site_q(t, dt, site, initialstate, trotter_alg)
                total_magnetization_noise += m_noise
                total_magnetization_noise_rm += m_noise_rm
            data[0, t] += total_magnetization_noise
            data_rm[0, t] += total_magnetization_noise_rm
        return data_rm, data

    def two_point_correlations_q(self, trotter_alg, total_t, dt, alpha, beta, chosen_pairs, initialstate=0):
        data_real, data_imag = sch.gen_m(len(chosen_pairs), total_t), sch.gen_m(len(chosen_pairs), total_t)
        data_real_rm, data_imag_rm = sch.gen_m(len(chosen_pairs), total_t), sch.gen_m(len(chosen_pairs), total_t)
        constant = 1.0
        if self.paper in ['joel', 'tachinno']:
            constant = 4.0
        for pair in chosen_pairs:
            for t in range(total_t):
                for j in range(2):
                    qc = QuantumCircuit(self.n + 1, 1)
                    self.init_state(qc, 1, initialstate)
                    qc.h(0)
                    sch.choose_control_gate(beta, qc, 0, pair[1] + 1)
                    trotter_alg(qc, dt, t, 1)
                    sch.choose_control_gate(alpha, qc, 0, pair[0] + 1)
                    sch.real_or_imag_measurement(qc, j)
                    measurement_noise_rm, measurement_noise = self.run_circuit(qc, None)
                    if j == 0:
                        data_real[chosen_pairs.index(pair), t] += measurement_noise / constant
                        data_real_rm[chosen_pairs.index(pair), t] += measurement_noise_rm / constant
                    elif j == 1:
                        data_imag[chosen_pairs.index(pair), t] += measurement_noise / constant
                        data_imag_rm[chosen_pairs.index(pair), t] += measurement_noise_rm / constant
        return data_real_rm, data_imag_rm, data_real, data_imag


# ========================================= Compare Classical & Quantum Simulations  =================================>


class HeisenbergModel():

    def __init__(self, j=0.0, bg=0.0, a=1.0, n=0, open_chain=True, transverse=False, paper='',
                 ising=False, epsilon=0.0, unity=False, RMfilename=''):
        self.classical_chain = ClassicalSpinChain(j=j, bg=bg, a=a, n=n, open_chain=open_chain, unity=unity,
                                                   ising=ising, paper=paper, transverse=transverse)
        self.quantum_chain = QuantumSim(j=j, bg=bg, a=a, n=n, open_chain=open_chain,
                            transverse=transverse, paper=paper, ising=ising, epsilon=epsilon, RMfilename=RMfilename)
        self.second = self.quantum_chain.second_order_trotter
        self.RMfilename = RMfilename

    def all_site_magnetization(self, total_t=0, dt=0.0, initstate=0, hadamard=False, filename=''):
        num_states = self.classical_chain.states
        psi0 = sps.csc_matrix(np.zeros((num_states, 1)))
        if hadamard:
            spins3 = self.classical_chain.states
            psi0 += sch.init_spin_state(0, spins3) - (sch.init_spin_state(2, spins3)
                                                      + sch.init_spin_state(3, spins3)) / math.sqrt(2)
        else:
            psi0 += sch.init_spin_state(initstate, num_states)

        data_two_RM, data_two_noRM = self.quantum_chain.all_site_magnetization_q(self.second, total_time=total_t, dt=dt,
                                                                initialstate=initstate, hadamard=hadamard)
        data_cl = self.classical_chain.all_site_magnetization_c(total_t, dt, psi0)
        # Write data to file to plot later
        sch.write_numpy_array(np.real(data_two_noRM.toarray()), filename + "_q_nRM.txt")
        sch.write_numpy_array(np.real(data_two_RM.toarray()), filename + "_q_RM.txt")
        sch.write_numpy_array(np.real(data_cl.toarray()), filename + "_cl.txt")
        vars = [self.classical_chain.n, self.classical_chain.j, total_t, dt]
        sch.write_data(vars, filename + "_vars.csv")

    def total_magnetization(self, total_t=0, dt=0.0, initstate=0, filename=''):
        psi0 = sch.init_spin_state(initstate, self.classical_chain.states)
        data_rm, data = self.quantum_chain.total_magnetization_q(self.second, total_t, dt, initialstate=initstate)
        data_cl = self.classical_chain.total_magnetization_c(total_t, dt, psi0)
        # Write data to file to plot later
        sch.write_numpy_array(np.real(data_cl.toarray()), filename + "_cl.txt")
        sch.write_numpy_array(np.real(data.toarray()), filename + "_q_nRM.txt")
        sch.write_numpy_array(np.real(data_rm.toarray()), filename + "_q_RM.txt")
        vars = [self.classical_chain.n, self.classical_chain.j, total_t, dt]
        sch.write_data(vars, filename + "_vars.csv")

    def two_point_correlations(self, op_order='', total_t=0, dt=0.0, pairs=[], initstate=0, filename=''):
        alpha, beta = op_order[0], op_order[1]
        psi0 = sch.init_spin_state(initstate, self.classical_chain.states)
        data_real_rm, data_imag_rm, data_real, data_imag = self.quantum_chain.two_point_correlations_q(self.second,
                    total_t, dt, alpha, beta, pairs, initialstate=initstate)
        data_real_cl, data_imag_cl = self.classical_chain.two_point_correlations_c(total_t, dt, psi0, op_order,
                                                                                   pairs=pairs)
        # Write data to file to plot later
        sch.write_numpy_array(np.real(data_real.toarray()), filename + "_q_real_nRM.txt")
        sch.write_numpy_array(np.real(data_real_cl.toarray()), filename + "_cl_real.txt")
        sch.write_numpy_array(np.real(data_imag.toarray()), filename + "_q_imag_nRM.txt")
        sch.write_numpy_array(np.real(data_imag_cl.toarray()), filename + "_cl_imag.txt")
        sch.write_numpy_array(np.real(data_real_rm.toarray()), filename + "_q_real_RM.txt")
        sch.write_numpy_array(np.real(data_imag_rm.toarray()), filename + "_q_image_RM.txt")
        vars = [self.classical_chain.n, self.classical_chain.j, total_t, dt]
        sch.write_data(vars, filename + "_vars.csv")
        sch.write_data(pairs, filename + "_pairs.csv")

# ==================================== TESTERS ==================================================================== >

# TACHINNO FIG 5a  (All Site Magnetization)


#model = HeisenbergModel(j=1, bg=0, n=2, j2=0, choice=False, paper='tachinno', epsilon=0.2,
#                        unity=True, RMfilename='RM_Arrays/santiago_RM_Oct20_AncillaQubit2.txt')
#model.all_site_magnetization(total_t=35, dt=0.1, initstate=0, hadamard=True, filename='tachinno_5a_Oct20')

# ---------------------------------------------------------------------------------------------------------------------

# JOEL FIG 2a  (All Site Magnetization)

model = HeisenbergModel(j=1, bg=0, a=0.5, n=6, open_chain=True, paper='joel', epsilon=0.2,
                        RMfilename='RM_Arrays/santiago_RM_Oct20_AncillaQubit2.txt')
initstate=model.classical_chain.states - 2
model.all_site_magnetization(total_t=80, dt=0.1, initstate=initstate, filename='joel_Oct25')

# --------------------------------------------------------------------------------------------------------------------------
# TACCHINO FIG 7 (Two Point Correlations)
#model = HeisenbergModel(j=-1, bg=20, a=1, n=3, j2=1, choice=False, open_chain=True, transverse=False, paper='tachinno',
#                 ising=False, epsilon=0.2, unity=True, RMfilename='RM_Arrays/santiago_RM_Oct20_AncillaQubit2.txt')
#model.two_point_correlations(op_order='xx', total_t=330, dt=.01, pairs=[(0,0)], initstate=int('000', 2),
#                             filename="tachinno_7a")
#model.two_point_correlations(op_order='xx', total_t=330, dt=.01, pairs=[(1,0)], initstate=int('000', 2), filename="tachinno_7b")
# model.two_point_correlations(op_order='xx', total_t=330, dt=.01, pairs=[(2,0)], initstate=int('000', 2))


# ----------------------------------------------------------------------------------------------------------------------
# TACCHINO FIG 5c (TOTAL MAGNETIZATION)
#model = HeisenbergModel(j=1, bg=2, a=1, n=2, j2=1, choice=False, open_chain=True, transverse=True, paper='tachinno',
#                 ising=True, epsilon=0.2, unity=True, RMfilename='RM_Arrays/santiago_RM_Oct20_AncillaQubit2.txt')
#model.total_magnetization(total_t=65, dt=0.1, initstate=0, filename="tachinno_5c_Oct20") # was 650, .01

# ----------------------------------------------------------------------------------------------------------------------

#model = HeisenbergModel(j=-.84746, j2=1, bg=0, a=1, n=4, open_chain=False, unity=True, choice=False, paper='francis', epsilon=0.1,
#                        RMfilename='RM_Arrays/santiago_RM_Oct20_AncillaQubit2.txt')
#model.two_point_correlations(op_order='xx', total_t=600, dt=.01, pairs=[(1, 1), (2, 1), (3, 1)], initstate=0,
#                             filename="francis")
