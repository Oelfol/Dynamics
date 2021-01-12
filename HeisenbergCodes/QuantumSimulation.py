###########################################################################
# QuantumSimulation.py
# Part of HeisenbergCodes
# Updated January '21
# Qiskit functions for creating initial state and computing expectation
# values.
#
# Current Qiskit:
# qiskit 0.21.0
# qiskit-terra 0.15.2
# qiskit-aer 0.6.1
# qiskit-ibmq-provider 0.9.0

# WIP
############################################################################

from qiskit import QuantumCircuit
import numpy as np
import ClassicalSimulation as cs
import HelpingFunctions as hf


class QuantumSim:

    def __init__(self, j=0.0, bg=0.0, a=1.0, n=0, open=True, trns=False, p='', ising=False, eps=0, dev_params=[]):

        ###################################################################################################
        # Params:
        # (j, coupling constant); (bg, magnetic field); (a, anisotropy jz/j);
        # (n, number of sites); (open, whether open-ended chain); (states, number of basis states)
        # (unity, whether h-bar/2 == 1 (h-bar == 1 elsewise)); (ising, for ising model);
        # (trns; transverse ising); (p, for settings related to examples from a specific paper 'p')
        # (eps, precision for trotter steps); (dev_params, for running circuit)
        ###################################################################################################

        self.j = j
        self.bg = bg
        self.n = n
        self.states = 2 ** n
        self.a = a
        self.open = open
        self.trns = trns
        self.p = p
        self.ising = ising
        self.unity = False
        self.eps = eps
        if self.p == 'francis':
            self.unity = True

        classical_h = cs.ClassicalSpinChain(j=self.j, bg=self.bg, a=self.a, n=self.n, open=self.open, unity=self.unity)
        self.h_commutes = classical_h.test_commuting_matrices()
        self.pairs_nn, autos = hf.gen_pairs(self.n, False, self.open)
        self.total_pairs = self.pairs_nn

        # For use with ibmq devices and noise models:
        self.device_params = dev_params

        # Address needed pseudospin constants for particular paper:
        self.spin_constant = 1
        if self.p in ['joel', 'tachinno']:
            self.spin_constant = 2

        # Params for trotter
        self.params = [self.h_commutes, self.j, self.eps, self.spin_constant, self.n, self.bg, self.trns,
                        self.total_pairs, self.ising, self.p, self.a]
    #####################################################################################

    def init_state(self, qc, ancilla, psi0):
        # Initialize a circuit in the desired spin state. Add a qubit if there is an ancilla measurement.
        state_temp, anc, index = np.binary_repr(psi0).zfill(self.n)[::-1], int(ancilla), 0
        for x in state_temp:
            if x == '1':
                qc.x(index + anc)
            index += 1
    #####################################################################################

    def magnetization_per_site_q(self, t, dt, site, psi0, trotter_alg, hadamard=False):
        qc_id = QuantumCircuit(self.n + 1, 1)
        self.init_state(qc_id, True, psi0)
        qc_id.h(0)
        if hadamard:
            qc_id.h(2)  # -------------------------------------------------------------> tachinno fig 5a

        trotter_alg(qc_id, dt, t, 1, self.params)
        hf.choose_control_gate('z', qc_id, 0, site + 1)
        qc_id.h(0)
        measurement_id = hf.run_circuit(1, qc_id, False, self.device_params, self.n)
        measurement_noise = hf.run_circuit(1, qc_id, True, self.device_params, self.n)

        return measurement_id / self.spin_constant, measurement_noise / self.spin_constant
    #####################################################################################

    def all_site_magnetization_q(self, trotter_alg, total_time=0, dt=0.0, psi0=0, hadamard=False):
        data_id = hf.gen_m(self.n, total_time)
        data_noise = hf.gen_m(self.n, total_time)
        for t in range(total_time):
            for site in range(self.n):
                m_id, m_noise = self.magnetization_per_site_q(t, dt, site, psi0, trotter_alg, hadamard=hadamard)
                data_id[site, t] += m_id
                data_noise[site, t] += m_noise

        return [data_id, data_noise]
    #####################################################################################

    def total_magnetization_q(self, trotter_alg, total_time=0, dt=0.0, psi0=0):
        data_id = hf.gen_m(1, total_time)
        data_noise = hf.gen_m(1, total_time)
        data_gates = hf.gen_m(1, total_time)

        for t in range(total_time):
            total_magnetization_id = 0
            total_magnetization_noise = 0
            num_gates_total = 0
            for site in range(self.n):
                measurement_id, measurement_noise = self.magnetization_per_site_q(t, dt, site, psi0, trotter_alg)
                total_magnetization_id += measurement_id
                total_magnetization_noise += measurement_noise

            data_id[0, t] += total_magnetization_id
            data_noise[0, t] += total_magnetization_noise
            data_gates[0, t] += num_gates_total / self.n

        return [data_id, data_noise]
    #####################################################################################

    def twoPtCorrelationsQ(self, trotter_alg, total_t, dt, alpha, beta, chosen_pairs, psi0=0):
        data_real_id, data_imag_id = hf.gen_m(len(chosen_pairs), total_t), hf.gen_m(len(chosen_pairs), total_t)
        data_real_noise, data_imag_noise = hf.gen_m(len(chosen_pairs), total_t), hf.gen_m(len(chosen_pairs), total_t)
        sc = (self.spin_constant * 2)

        for pair in chosen_pairs:
            for t in range(total_t):
                for j in range(2):
                    qc = QuantumCircuit(self.n + 1, 1)
                    self.init_state(qc, 1, psi0)
                    qc.h(0)
                    qc.barrier()
                    hf.choose_control_gate(beta, qc, 0, pair[1] + 1)
                    qc.barrier()
                    trotter_alg(qc, dt, t, 1, self.params)
                    qc.barrier()
                    hf.choose_control_gate(alpha, qc, 0, pair[0] + 1)
                    qc.barrier()
                    hf.real_or_imag_measurement(qc, j)
                    measurement_id = hf.run_circuit(1, qc, False, self.device_params, self.n) / sc
                    measurement_noise = hf.run_circuit(1, qc, True, self.device_params, self.n) / sc

                    if j == 0:
                        data_real_id[chosen_pairs.index(pair), t] += measurement_id
                        data_real_noise[chosen_pairs.index(pair), t] += measurement_noise
                    elif j == 1:
                        data_imag_id[chosen_pairs.index(pair), t] += measurement_id
                        data_imag_noise[chosen_pairs.index(pair), t] += measurement_noise

        data_real = [data_real_id, data_real_noise]
        data_imag = [data_imag_id, data_imag_noise]
        return data_real, data_imag
    #####################################################################################

    def occupation_probabilities_q(self, trotter_alg, total_time=0, dt=0.0, psi0=0, chosen_states=[]):
        data_id = hf.gen_m(len(chosen_states), total_time)
        data_noise = hf.gen_m(len(chosen_states), total_time)
        for t in range(total_time):
            qc = QuantumCircuit(self.n, self.n)
            qc.barrier()
            self.init_state(qc, 0, psi0)
            qc.barrier()
            trotter_alg(qc, dt, t, 0)
            measurements_id = hf.run_circuit(0, qc, False, self.device_params, self.n)
            measurements_noise = hf.run_circuit(0, qc, True, self.device_params, self.n)

            for x in chosen_states:
                data_noise[chosen_states.index(x), t] = measurements_noise[x]
            for x in chosen_states:
                data_id[chosen_states.index(x), t] = measurements_id[x]

        data = [data_id, data_noise]
        return data
