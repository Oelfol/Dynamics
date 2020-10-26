# TestingOldTrotterCode.py
# Improvements to CompareTrotterAlgorithms and cleanup.
# October2020

import numpy as np
import math
import warnings
import scipy.sparse as sps

import spinchainshelpers as sch
import quantumsimulation as qs
import plottinghelpers as ph
import classicalsimulation as cs
warnings.filterwarnings('ignore')


class HeisenbergModel():

    def __init__(self, j=0.0, bg=0.0, a=1.0, n=0, j2=1, open_chain=True, transverse=False, paper='',
                 ising=False, epsilon=0.0, unity=False):
        self.classical_chain = cs.ClassicalSpinChain(j=j, j2=j2, bg=bg, a=a, n=n, open_chain=open_chain, unity=unity,
                                                     ising=ising, paper=paper, transverse=transverse)
        self.quantum_chain = qs.QuantumSim(j=j, bg=bg, a=a, n=n, open_chain=open_chain,
                                        transverse=transverse, paper=paper, ising=ising, epsilon=epsilon)
        self.first = self.quantum_chain.first_order_trotter
        self.second = self.quantum_chain.second_order_trotter

    def all_site_magnetization(self, total_t=0, dt=0, initstate=0, hadamard=False):
        num_states = self.classical_chain.states
        psi0 = sps.csc_matrix(np.zeros((num_states, 1)))
        if hadamard:
            spins3 = self.classical_chain.states
            psi0 += sch.init_spin_state(0, spins3) \
                    - (sch.init_spin_state(2, spins3) + sch.init_spin_state(3, spins3)) / math.sqrt(2)
        else:
            psi0 += sch.init_spin_state(initstate, num_states)
        data_one, data_gates_one = self.quantum_chain.all_site_magnetization_q(self.first, total_time=total_t, dt=dt,
                                                                               initialstate=initstate,
                                                                               hadamard=hadamard)
        data_two, data_gates_two = self.quantum_chain.all_site_magnetization_q(self.second, total_time=total_t, dt=dt,
                                                                               initialstate=initstate,
                                                                               hadamard=hadamard)
        data_cl = self.classical_chain.all_site_magnetization_c(total_t, dt, psi0)
        n, j = self.classical_chain.n, self.classical_chain.j
        data_gates = [data_gates_one, data_gates_two]
        data = [data_one, data_two]
        ph.all_site_magnetization_plotter(n, j, dt, total_t, data, data_gates, data_cl)

    def total_magnetization(self, total_t=0, dt=0, initstate=0):
        psi0 = sch.init_spin_state(initstate, self.classical_chain.states)
        data_one, data_gates_one = self.quantum_chain.total_magnetization_q(self.first, total_t, dt,
                                                                            initialstate=initstate)
        data_two, data_gates_two = self.quantum_chain.total_magnetization_q(self.first, total_t, dt,
                                                                            initialstate=initstate)
        data_cl = self.classical_chain.total_magnetization_c(total_t, dt, psi0)
        data_gates = [data_gates_one, data_gates_two]
        data = [data_one, data_two]
        j_ = self.classical_chain.j
        ph.total_magnetization_plotter(j_, total_t, dt, data, data_gates, data_cl)

    def two_point_correlations(self, op_order='', total_t=0, dt=0, pairs=[], initstate=0):
        alpha, beta = op_order[0], op_order[1]
        psi0 = sch.init_spin_state(initstate, self.classical_chain.states)
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
        ph.two_point_correlations_plotter(alpha, beta, j_, dt, pairs, d_one, d_two, data_cl)

    def occupation_probabilities(self, total_t=0, dt=0, initstate=0, chosen_states=[]):
        psi0 = sch.init_spin_state(initstate, self.classical_chain.states)
        data_one, data_gates_one = self.quantum_chain.occupation_probabilities_q(self.first, total_t,
                                                                                 dt, initialstate=initstate,
                                                                                 chosen_states=chosen_states)
        data_two, data_gates_two = self.quantum_chain.occupation_probabilities_q(self.second, total_t,
                                                                                 dt, initialstate=initstate,
                                                                                 chosen_states=chosen_states)
        data_cl = self.classical_chain.occupation_probabilities_c(total_t, dt, initialstate=psi0,
                                                                  chosen_states=chosen_states)
        data_gates = [data_gates_one, data_gates_two]
        n = self.classical_chain.n
        data = [data_one, data_two]
        ph.occ_plotter(chosen_states, self.classical_chain.j, n, total_t, dt, data_gates, data, data_cl)

# ==================================== TESTERS ==================================================================== >

# TACHINNO FIG 5a  (All Site Magnetization)

# model = HeisenbergModel(j=1, bg=0, n=2, j2=0, choice=False, paper='tachinno', epsilon=0.2, unity=True)
# model.all_site_magnetization(total_t=35, dt=0.1, initstate=0, hadamard=True)

# ---------------------------------------------------------------------------------------------------------------------

# JOEL FIG 2a  (All Site Magnetization)

model = HeisenbergModel(j=1, bg=0, a=0.5, n=5, j2=0, open_chain=True, paper='joel', epsilon=0.2)
initstate=model.classical_chain.states - 2
model.all_site_magnetization(total_t=80, dt=0.1, initstate=initstate)

# ----------------------------------------------------------------------------------------------------------------------
# TACCHINO FIG 5b (Occupation Probabilities)

# model = HeisenbergModel(j=1, bg=20, a=1, n=3, j2=1, choice=False, open_chain=True, transverse=False, paper='tachinno',#
#                    ising=False, epsilon=0.2, unity=True)
# c = [int(x, 2) for x in ['100', '010', '111']]
# model.occupation_probabilities(total_t=350, dt=0.01, initstate=int('100', 2), chosen_states=c)


# --------------------------------------------------------------------------------------------------------------------------
# TACCHINO FIG 7 (Two Point Correlations)
# model = HeisenbergModel(j=-1, bg=20, a=1, n=3, j2=1, choice=False, open_chain=True, transverse=False, paper='tachinno',
#                 ising=False, epsilon=0.2, unity=True)
# model.two_point_correlations(op_order='xx', total_t=330, dt=.01, pairs=[(0,0)], initstate=int('000', 2))
# model.two_point_correlations(op_order='xx', total_t=330, dt=.01, pairs=[(1,0)], initstate=int('000', 2))
# model.two_point_correlations(op_order='xx', total_t=330, dt=.01, pairs=[(2,0)], initstate=int('000', 2))


# ----------------------------------------------------------------------------------------------------------------------
# TACCHINO FIG 5c (TOTAL MAGNETIZATION)
# model = HeisenbergModel(j=1, bg=2, a=1, n=2, j2=1, choice=False, open_chain=True, transverse=True, paper='tachinno',
#                 ising=True, epsilon=0.2, unity=True)
# model.total_magnetization(total_t=650, dt=0.01, initstate=0)

# ----------------------------------------------------------------------------------------------------------------------

# model = HeisenbergModel(j=-.84746, j2=1, bg=0, a=1, n=4, open_chain=False, unity=True, choice=False, paper='francis', epsilon=0.1)
# model.two_point_correlations(op_order='xx', total_t=600, dt=.01, pairs=[(1, 1), (2, 1), (3, 1)], initstate=0)
