# Code for classical simulations of spin models
# Updated Oct. 2020

import spinchainshelpers as sch
import numpy as np
import scipy.sparse as sps
import scipy.linalg as sl


class ClassicalSpinChain:

    def __init__(self, j=0.0, bg=0.0, a=0.0, n=0, open_chain=False, unity=False, ising=False,
                 paper='', transverse=False):

        self.j = j  # ===============================================# coupling constant
        self.bg = bg  # =============================================# magnetic field strength
        self.a = a  # ===============================================# anisotropy constant jz/j
        self.n = n  # ===============================================# sites
        self.open_chain = open_chain  # =============================# whether open chain
        self.states = 2 ** n  # =====================================# number of basis states
        self.unity = unity  # =======================================# whether h-bar/2 == 1 (h-bar == 1 elsewise)
        self.ising = ising  # =======================================# for ising model
        self.paper = paper  # =======================================# refer to settings for a specific paper
        self.transverse = transverse  # =============================# whether transverse field ising model
        self.hamiltonian = sps.lil_matrix(np.zeros([2 ** n, 2 ** n], complex))
        neighbors, autos = sch.gen_pairs(self.n, False, self.open_chain)
        multipliers = [1, 1, self.a]

        ops = ['x', 'y', 'z']
        if ising:
            ops = ['z']
            multipliers = [self.a]

        for y in neighbors:
            for op in ops:
                dex = ops.index(op)
                s1, s2 = sch.spin_op(op, y[0], self.n, self.unity), sch.spin_op(op, y[1], self.n, self.unity)
                self.hamiltonian += s1.dot(s2) * self.j * multipliers[dex]
        for x in range(self.n):
            if self.bg != 0 and self.transverse == False:
                s_final = sch.spin_op('z', x, self.n, self.unity) * self.bg / 2
                self.hamiltonian += s_final
            elif self.bg != 0 and self.transverse == True:
                s_final = sch.spin_op('x', x, self.n, self.unity) * self.bg / 2
                self.hamiltonian += s_final

        self.hamiltonian = sps.csc_matrix(self.hamiltonian)

    def test_commuting_matrices(self):
        ops, multipliers = ['x', 'y', 'z'], [1, 1, self.a]
        neighbors, nn_neighbors, autos = sch.gen_pairs(self.n, False, self.open_chain)
        if self.ising:
            ops = ['z']
            multipliers = [self.a]

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
                    commuting = False
            elif self.bg != 0 and self.transverse == True:
                s_final = sch.spin_op('x', x, self.n, self.unity) * self.bg / 2
                if not sch.commutes(terms, s_final):
                    commuting = False

        return commuting

    def two_point_correlations_c(self, total_time, dt, psi0, op_order, pairs=[]):

        if pairs == []:
            # if wanting to generate all pairs, feed in none
            nn, auto = sch.gen_pairs(self.n, True, self.open_chain)
            pairs = nn + auto
        dyn_data_real, dyn_data_imag = sch.gen_m(len(pairs), total_time), sch.gen_m(len(pairs), total_time)

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
                si = sch.spin_op(op_order[0], pairs[x][0], self.n, self.unity)
                sj = sch.spin_op(op_order[1], pairs[x][1], self.n, self.unity)
                ket = u_dag.dot(si.dot(u.dot(sj.dot(psi0))))
                res = psi_dag.dot(ket).toarray()[0][0]
                dyn_data_real[x, t] = np.real(res) / pseudo_constant
                dyn_data_imag[x, t] = np.imag(res) / pseudo_constant
        return dyn_data_real, dyn_data_imag

    def occupation_probabilities_c(self, total_time=0, dt=0.0, initialstate=None, chosen_states=[]):

        psi0 = initialstate
        basis_matrix = sps.csc_matrix(np.eye(self.states))
        data = sch.gen_m(len(chosen_states), total_time)
        for t in range(total_time):
            u = sl.expm(self.hamiltonian * dt * t * (-1j))
            psi_t = u.dot(psi0)
            index_ = 0
            for i in chosen_states:
                basis_bra = (basis_matrix[i, :])
                probability = (basis_bra.dot(psi_t)).toarray()[0][0]
                probability = ((np.conj(psi_t).transpose()).dot(np.conj(basis_bra).transpose())).toarray()[0][
                                  0] * probability
                data[index_, t] = probability
                index_ += 1
        return data

    def magnetization_per_site_c(self, total_time, dt, psi0, site):

        # Account for differences between examples from papers
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
