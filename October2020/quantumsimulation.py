# Code for quantum simulations of spin models using Qiskit
# Inherited from CompareTrotterAlgorithms to help organize code.
# Separate set of functions from error mitigation codes.
# Updated Oct. 2020

from qiskit import QuantumCircuit, execute
import numpy as np
import math

import classicalsimulation as cs
import spinchainshelpers as sch
import ibmq_setup as ibmq

shots = 50000
dev_name = 'ibmq_santiago'
setup = ibmq.ibmqSetup(sim=True, shots=shots, dev_name=dev_name)
device, nm, bg, cm = setup.get_noise_model()

class QuantumSim:

    def __init__(self, j=0.0, bg=0.0, a=1.0, n=0, open_chain=True, transverse=False, paper='',
                 ising=False, epsilon=0):

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

        self.h_commutes = cs.ClassicalSpinChain(j=self.j, bg=self.bg, a=self.a, n=self.n,
                                             open_chain=self.open_chain, unity=self.unity).test_commuting_matrices()
        self.pairs_nn, self.pairs_nnn, autos = sch.gen_pairs(self.n, False, self.open_chain)
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
                sch.three_cnot_evolution(qc, x, ancilla, self.j, t, dt,
                                         trotter_steps * pseudo_constant_a, self.ising, self.a)
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
                sch.three_cnot_evolution(qc, x, ancilla, self.j, t, dt,
                    trotter_steps * pseudo_constant_a * 2, self.ising, self.a)
                num_gates = num_gates + self.update_gates()
            mid = self.total_pairs[0]
            sch.three_cnot_evolution(qc, mid, ancilla, self.j, t, dt,
                   trotter_steps * pseudo_constant_a, self.ising, self.a)
            num_gates = num_gates + self.update_gates()
            for x in self.total_pairs[1:]:
                sch.three_cnot_evolution(qc, x, ancilla, self.j, t, dt,
                    trotter_steps * pseudo_constant_a * 2.0, self.ising, self.a)
                num_gates = num_gates + self.update_gates()

        return 0  #num_gates

    def run_circuit(self, anc, qc, noise):

        if anc == 1:
            # All ancilla-assisted measurements
            qc.measure(0, 0)
            if not noise:
                # ideal simulation
                result = execute(qc, backend=simulator, shots=50000).result()
                counts = [result.get_counts(i) for i in range(len(result.results))]
                probs = sch.sort_counts(counts[0], anc, 50000)
                return probs[0] - probs[1]
            else:
                # noisy simulation
                result = execute(qc, backend=simulator, shots=50000, noise_model=nm, basis_gates=bg).result()
                counts = [result.get_counts(i) for i in range(len(result.results))]
                probs = sch.sort_counts(counts[0], anc, 50000)
                return probs[0] - probs[1]

        else:
            # No ancilla (occupation probabilities)
            for x in range(self.n):
                qc.measure(x, x)

            if not noise:
                # ideal simulation
                result = execute(qc, backend=simulator, shots=50000).result()
                counts = [result.get_counts(i) for i in range(len(result.results))]
                measurements = sch.sort_counts(counts[0], self.n, 50000)
                return measurements
            else:
                # noisy simulation
                result = execute(qc, backend=simulator, shots=50000, noise_model=nm, basis_gates=bg).result()
                counts = [result.get_counts(i) for i in range(len(result.results))]
                measurements = sch.sort_counts(counts[0], self.n, 50000)
                return measurements

    def magnetization_per_site_q(self, t, dt, site, initialstate, trotter_alg, hadamard=False):
        qc_id, qc_noise = QuantumCircuit(self.n + 1, 1), QuantumCircuit(self.n + 1, 1)
        self.init_state(qc_id, True, initialstate)
        self.init_state(qc_noise, True, initialstate)
        qc_id.h(0)
        qc_noise.h(0)
        if hadamard:
            qc_id.h(2)  # -------------------------------------------------------------> tachinno fig 5a
            qc_noise.h(2)

        num_gates_id = trotter_alg(qc_id, dt, t, 1)
        num_gates_noise = trotter_alg(qc_noise, dt, t, 1)
        num_gates = num_gates_noise  # only need one
        sch.choose_control_gate('z', qc_id, 0, site + 1)
        sch.choose_control_gate('z', qc_noise, 0, site + 1)
        qc_id.h(0)
        qc_noise.h(0)
        measurement_id = self.run_circuit(1, qc_id, False)
        measurement_noise = self.run_circuit(1, qc_noise, True)

        # Address needed constants for particular paper
        pseudo_constant = 1
        if self.paper in ['joel', 'tachinno']:
            pseudo_constant = 2

        return measurement_id / pseudo_constant, measurement_noise / pseudo_constant, num_gates


    def all_site_magnetization_q(self, trotter_alg, total_time=0, dt=0.0, initialstate=0, hadamard=False):
        data_id = sch.gen_m(self.n, total_time)
        data_noise = sch.gen_m(self.n, total_time)
        data_gates = sch.gen_m(1, total_time)

        for t in range(total_time):
            num_gates_total = 0
            for site in range(self.n):
                m_id, m_noise, num_gates = self.magnetization_per_site_q(t, dt, site, initialstate, trotter_alg,
                                                                         hadamard=hadamard)
                data_id[site, t] += m_id
                data_noise[site, t] += m_noise
                num_gates_total += num_gates
            data_gates[0, t] += num_gates_total / self.n

        data = [data_id, data_noise]
        return data, data_gates


    def total_magnetization_q(self, trotter_alg, total_time=0, dt=0.0, initialstate=0):
        data_id = sch.gen_m(1, total_time)
        data_noise = sch.gen_m(1, total_time)
        data_gates = sch.gen_m(1, total_time)

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


    def two_point_correlations_q(self, trotter_alg, total_t, dt, alpha, beta, chosen_pairs, initialstate=0):
        # --------- #
        data_real_id, data_imag_id = sch.gen_m(len(chosen_pairs), total_t), sch.gen_m(len(chosen_pairs), total_t)
        data_real_noise, data_imag_noise = sch.gen_m(len(chosen_pairs), total_t), sch.gen_m(len(chosen_pairs), total_t)
        data_gates = sch.gen_m(1, total_t)

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

                    sch.choose_control_gate(beta, qc_id, 0, pair[1] + 1)
                    num_gates_id = trotter_alg(qc_id, dt, t, 1)
                    sch.choose_control_gate(alpha, qc_id, 0, pair[0] + 1)
                    sch.real_or_imag_measurement(qc_id, j)
                    measurement_id = self.run_circuit(1, qc_id, False) / pseudo_constant

                    sch.choose_control_gate(beta, qc_noise, 0, pair[1] + 1)
                    num_gates_noise = trotter_alg(qc_noise, dt, t, 1)
                    sch.choose_control_gate(alpha, qc_noise, 0, pair[0] + 1)
                    sch.real_or_imag_measurement(qc_noise, j)
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
        data_id = sch.gen_m(len(chosen_states), total_time)
        data_noise = sch.gen_m(len(chosen_states), total_time)
        data_gates = sch.gen_m(1, total_time)

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
