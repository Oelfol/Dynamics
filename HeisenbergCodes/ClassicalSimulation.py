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
import PlottingFunctions as pf


class ClassicalSpinChain:

    def __init__(self, j=0.0, bg=0.0, a=1.0, n=0, open=True, unity=False, ising=False, trns =False):

        ###################################################################################################
        # Params:
        # (j, coupling constant); (bg, magnetic field); (a, anisotropy jz/j);
        # (n, number of sites); (open, whether open-ended chain); (states, number of basis states)
        # (ising, for ising model); (trns ; transverse ising);  (unity, whether h-bar / 2 == 1)
        ###################################################################################################
        self.j = j
        self.bg = bg
        self.a = a
        self.n = n
        self.open = open
        self.states = 2 ** n
        self.unity = unity
        self.ising = ising
        self.trns = trns

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
                s1, s2 = hf.spin_op(op, y[0], self.n, self.unity), hf.spin_op(op, y[1], self.n, self.unity)
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
            print(t)
            u, u_dag, psi0_dag = te.classical_te(self.hamiltonian, dt, t, psi0)
            for x in range(len(pairs)):
                si = hf.spin_op(op_order[0], pairs[x][0], self.n, self.unity)
                sj = hf.spin_op(op_order[1], pairs[x][1], self.n, self.unity)
                ket = u_dag.dot(si.dot(u.dot(sj).dot(psi0)))
                res = psi0_dag.dot(ket).toarray()[0][0]
                dyn_data_real[x, t] = np.real(res)
                dyn_data_imag[x, t] = np.imag(res)

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
            u, u_dag, psi0_dag = te.classical_te(self.hamiltonian, dt, t, psi0)
            bra = np.conj(u.dot(psi0).transpose())
            s_z = hf.spin_op('z', site, self.n, self.unity)
            ket = s_z.dot(u.dot(psi0))
            data[0, t] += (bra.dot(ket).toarray()[0][0])
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

    #####################################################################################

    def dynamical_structure_factor(self, total_time, dt, psi0, alpha, beta, k_range, w_range):

        k_min, k_max, w_min, w_max = k_range[0], k_range[1], w_range[0], w_range[1]
        res, pairs = 300, []
        k_, w_ = np.arange(k_min, k_max, (k_max - k_min) / res), np.arange(w_min, w_max, (w_max - w_min) / res)
        k = np.array(k_.copy().tolist() * res).reshape(res, res).astype('float64')
        w = np.array(w_.copy().tolist() * res).reshape(res, res).T.astype('float64')
        for j in range(self.n):
            for p in range(self.n):
                pairs.append((j, p))

        tpc_real, tpc_imag = self.two_point_correlations_c(total_time, dt, psi0, [alpha, beta], pairs)
        dsf = np.zeros_like(k).astype('float64')

        tpc_real = tpc_real.toarray().astype('float64')
        tpc_imag = tpc_imag.toarray().astype('float64')

        count = 0
        for jk in range(len(pairs)):
            pair = pairs[jk]
            j = pair[0] - pair[1]
            print("the code is running!!")
            theta_one = - 1 * k * j
            time_sum = (np.zeros_like(w) / self.n).astype('float64')
            for t in range(total_time):
                tpc_r = tpc_real[count, t]
                tpc_i = tpc_imag[count, t]
                theta_two = w * t * dt
                theta = theta_one + theta_two
                time_sum += (np.cos(theta) * tpc_r * dt + np.sin(theta) * tpc_i * dt).astype('float64')
            count += 1
            dsf = dsf + time_sum

        dsf_mod = np.multiply(np.conj(dsf), dsf)
        pf.dyn_structure_factor_plotter(dsf_mod, w, k, False, self.j, k_range, res)
    #####################################################################################

    def gs_eigenvalue(self):

        H = self.hamiltonian.toarray()
        evals, evect = np.linalg.eigh(H)[0], np.linalg.eigh(H)[1]
        return evals[0]
