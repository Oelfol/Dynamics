###########################################################################
# main.py
# Part of HeisenbergCodes
# Updated January '21
#
# Contains HeisenbergModel class
###########################################################################

import numpy as np
import math
import warnings
import scipy.sparse as sps

import HelpingFunctions as hf
import QuantumSimulation as qs
import PlottingFunctions as ph
import ClassicalSimulation as cs
import TimeEvolution as te
warnings.filterwarnings('ignore')



class HeisenbergModel():

    def __init__(self, j=0.0, bg=0.0, a=1.0, n=0, open=True, trns=False, p='', ising=False, eps=0.0, unity=False,
                 dev_params=[], RMfile=''):
        cc = cs.ClassicalSpinChain(j=j, bg=bg, a=a, n=n, open=open, unity=unity, ising=ising, trns=trns)
        qh = qs.QuantumSim(j=j, bg=bg, a=a, n=n, open=open, trns=trns, p=p, ising=ising, eps=eps, dev_params=dev_params,
                           RMfile=RMfile)
        self.classical_chain = cc
        self.quantum_chain = qh
        self.first = te.first_order_trotter
        self.second = te.second_order_trotter
        self.RMfile = RMfile

    #####################################################################################

    def all_site_magnetization(self, total_t=0, dt=0, psi0=0, hadamard=False):
        qchain, cchain = self.quantum_chain, self.classical_chain
        num_states = cchain.states
        psi0_ = sps.csc_matrix(np.zeros((num_states, 1)))

        if hadamard:
            spins3 = self.classical_chain.states
            psi0_ += hf.init_spin_state(0, spins3) \
                    - (hf.init_spin_state(2, spins3) + hf.init_spin_state(3, spins3)) / math.sqrt(2)
        else:
            psi0_ += hf.init_spin_state(psi0, num_states)

        data_one = qchain.all_site_magnetization_q(self.first, total_time=total_t, dt=dt, psi0=psi0, hadamard=hadamard)
        data_two = qchain.all_site_magnetization_q(self.second, total_time=total_t, dt=dt, psi0=psi0, hadamard=hadamard)
        data_cl = cchain.all_site_magnetization_c(total_t, dt, psi0_)

        n, j = cchain.n, cchain.j
        data = [data_one, data_two]
        ph.all_site_magnetization_plotter(n, j, dt, total_t, data, data_cl)
    #####################################################################################

    def total_magnetization(self, total_t=0, dt=0, psi0=0):
        psi0 = hf.init_spin_state(psi0, self.classical_chain.states)
        qchain, cchain = self.quantum_chain, self.classical_chain

        data_one = qchain.total_magnetization_q(self.first, total_t, dt, psi0=psi0)
        data_two = qchain.total_magnetization_q(self.first, total_t, dt, psi0=psi0)
        data_cl = cchain.total_magnetization_c(total_t, dt, psi0)

        data = [data_one, data_two]
        j_ = self.classical_chain.j
        ph.total_magnetization_plotter(j_, total_t, dt, data, data_cl)
    #####################################################################################

    def two_point_correlations(self, op_order='', total_t=0, dt=0, pairs=[], psi0=0):
        alpha, beta = op_order[0], op_order[1]
        psi0_ = hf.init_spin_state(psi0, self.classical_chain.states)

        qchain, cchain = self.quantum_chain, self.classical_chain
        #data_real_one, data_imag_one = qchain.twoPtCorrelationsQ(self.first, total_t, dt, alpha, beta, pairs, psi0=psi0)
        #data_real_two, data_imag_two = qchain.twoPtCorrelationsQ(self.second, total_t, dt, alpha, beta, pairs, psi0=psi0)
        data_real_cl, data_imag_cl = cchain.two_point_correlations_c(total_t, dt, psi0_, op_order, pairs=pairs)

        ## Temporary matrices
        data_real_one, data_imag_one = [[hf.gen_m(len(pairs), total_t), hf.gen_m(len(pairs), total_t)],[hf.gen_m(len(pairs), total_t), hf.gen_m(len(pairs), total_t)]]
        data_real_two, data_imag_two = [[hf.gen_m(len(pairs), total_t), hf.gen_m(len(pairs), total_t)],[hf.gen_m(len(pairs), total_t), hf.gen_m(len(pairs), total_t)]]
        ##

        j_ = cchain.j
        d_one, d_two = [data_real_one, data_imag_one], [data_real_two, data_imag_two]
        data_cl = [data_real_cl, data_imag_cl]
        ph.two_point_correlations_plotter(alpha, beta, j_, dt, pairs, d_one, d_two, data_cl)
    #####################################################################################

    def occupation_probabilities(self, total_t=0, dt=0, initstate=0, chosen_states=[]):
        psi0 = hf.init_spin_state(initstate, self.classical_chain.states)
        qchain, cchain = self.quantum_chain, self.classical_chain

        data_one = qchain.occupation_probabilities_q(self.first, total_t, dt, psi0=initstate,
                                                     chosen_states=chosen_states)
        data_two = qchain.occupation_probabilities_q(self.second, total_t, dt, psi0=initstate,
                                                     chosen_states=chosen_states)
        data_cl = cchain.occupation_probabilities_c(total_t, dt, psi0=psi0, chosen_states=chosen_states)

        n = self.classical_chain.n
        data = [data_one, data_two]
        ph.occ_plotter(chosen_states, self.classical_chain.j, n, total_t, dt, data, data_cl)


