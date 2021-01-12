###########################################################################
# ClassicalSimulation.py
# Part of HeisenbergCodes
# Updated January '21
#
# Classical functions for creating initial state and computing expectation
# values.
###########################################################################

import numpy as np
import scipy.sparse as sps
import HelpingFunctions as hf
import TimeEvolution as te


class ClassicalSpinChain:

    def __init__(self, j=0.0, bg=0.0, a=1.0, n=0, open=True, unity=False, ising=False, p='', trns =False):

        ###################################################################################################
        # Params:
        # (j, coupling constant); (bg, magnetic field); (a, anisotropy jz/j);
        # (n, number of sites); (open, whether open-ended chain); (states, number of basis states)
        # (unity, whether h-bar/2 == 1 (h-bar == 1 elsewise)); (ising, for ising model);
        # (trns ; transverse ising); (p, for settings related to examples from a specific paper 'p')
        ###################################################################################################
        self.j = j
        self.bg = bg
        self.a = a
        self.n = n
        self.open = open
        self.states = 2 ** n
        self.unity = unity
        self.ising = ising
        self.p = p
        self.trns = trns

        # Spin constant based on reference paper
        self.spin_constant = 1
        if self.p in ['tachinno']:
            self.spin_constant = 2

        # Create Hamiltonian matrix:
        self.hamiltonian = sps.lil_matrix(np.zeros([2 ** n, 2 ** n], complex))
        neighbors, autos = hf.gen_pairs(self.n, False, self.open)
        multipliers = [1, 1, self.a]

        # Used to test commutation for trotter algorithm:
        self.even_terms = sps.lil_matrix(np.zeros([2 ** n, 2 ** n], complex))
        self.odd_terms = sps.lil_matrix(np.zeros([2 ** n, 2 ** n], complex))

        ops = ['x', 'y', 'z']
        if ising:
            ops = ['z']
            multipliers = [self.a]

        # Pairwise terms
        dex_terms = 0
        for y in neighbors:
            for op in ops:
                dex = ops.index(op)
                s1, s2 = hf.spin_op(op, y[0], self.n, self.unity), hf.spin_op(op, y[1], self.n, self.unity)  # TODO these were not agreeing with commuting part. figure it out
                self.hamiltonian += s1.dot(s2) * self.j * multipliers[dex]
                if dex_terms % 2 == 0:
                    self.even_terms += s1.dot(s2) * self.j * multipliers[dex]
                else:
                    self.odd_terms += s1.dot(s2) * self.j * multipliers[dex]
            dex_terms += 1

        # Ising terms
        for x in range(self.n):
            if self.bg != 0 and self.trns == False:
                self.hamiltonian += hf.spin_op('z', x, self.n, self.unity) * self.bg / 2
            elif self.bg != 0 and self.trns== True:
                self.hamiltonian += hf.spin_op('x', x, self.n, self.unity) * self.bg / 2

        self.hamiltonian = sps.csc_matrix(self.hamiltonian)
    #####################################################################################

    def test_commuting_matrices(self):
        commuting = True
        if not hf.commutes(self.even_terms, self.odd_terms):
            commuting = False
        return commuting
    #####################################################################################

    def two_point_correlations_c(self, total_time, dt, psi0, op_order, pairs=[]):

        # if wanting to generate all pairs, feed in none
        if len(pairs) == 0:
            nn, auto = hf.gen_pairs(self.n, True, self.open)
            pairs = nn + auto
        dyn_data_real = hf.gen_m(len(pairs), total_time)
        dyn_data_imag = hf.gen_m(len(pairs), total_time)

        for t in range(total_time):
            u, u_dag, psi0_dag = te.classical_te(self.hamiltonian, dt, t, psi0)
            for x in range(len(pairs)):
                si = hf.spin_op(op_order[0], pairs[x][0], self.n, self.unity)
                sj = hf.spin_op(op_order[1], pairs[x][1], self.n, self.unity)
                ket = u_dag.dot(si.dot(u.dot(sj).dot(psi0)))
                res = psi0_dag.dot(ket).toarray()[0][0]
                dyn_data_real[x, t] = np.real(res) / (self.spin_constant*2)
                dyn_data_imag[x, t] = np.imag(res) / (self.spin_constant*2)
        return dyn_data_real, dyn_data_imag
    #####################################################################################

    def occupation_probabilities_c(self, total_time=0, dt=0.0, psi0=None, chosen_states=[]):
        basis_matrix = sps.csc_matrix(np.eye(self.states))
        data = hf.gen_m(len(chosen_states), total_time)
        for t in range(total_time):
            u, u_dag, psi0_dag = te.classical_te(self.hamiltonian, dt, t, psi0)
            psi_t = u.dot(psi0)
            for index_ in range(len(chosen_states)):
                i = chosen_states[index_]
                basis_bra = (basis_matrix[i, :])
                prob = (basis_bra.dot(psi_t)).toarray()[0][0]
                prob = ((np.conj(psi_t).transpose()).dot(np.conj(basis_bra).transpose())).toarray()[0][0] * prob
                data[index_, t] = prob
        return data
    #####################################################################################

    def magnetization_per_site_c(self, total_time, dt, psi0, site):
        data = hf.gen_m(1, total_time)
        for t in range(total_time):
            u, u_dag, psi0_dag = te.classical_te(self.hamiltonian, dt, t, psi0) # test this function again
            bra = np.conj(u.dot(psi0).transpose())
            s_z = hf.spin_op('z', site, self.n, self.unity)
            ket = s_z.dot(u.dot(psi0))
            data[0, t] += (bra.dot(ket).toarray()[0][0]) / self.spin_constant
        return data
    #####################################################################################

    def all_site_magnetization_c(self, total_time, dt, psi0):
        data = hf.gen_m(self.n, total_time)
        for site in range(self.n):
            data[site, :] += self.magnetization_per_site_c(total_time, dt, psi0, site)
        return data
    #####################################################################################

    def total_magnetization_c(self, total_time, dt, psi0):
        data = hf.gen_m(1, total_time)
        for site in range(self.n):
            data = data + self.magnetization_per_site_c(total_time, dt, psi0, site)
        return data
