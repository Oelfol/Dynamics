###########################################################################
# TimeEvolution.py
# Part of HeisenbergCodes
# Updated January '21
#
# Time evolution codes including 1st/2nd-order Suzuki-Trotter
###########################################################################

import HelpingFunctions as hf
import math
import scipy.linalg as sl
import numpy as np


def classical_te(hamiltonian, dt, t, psi0):
    u = sl.expm(hamiltonian * dt * t * (-1j))
    u_dag = sl.expm(hamiltonian * dt * t * (1j))
    psi0_dag = np.conj(psi0).transpose()
    return u_dag, u, psi0_dag


def first_order_trotter(qc, dt, t, ancilla, params):

    [h_commutes, j, eps, sc, n, bg, trns, total_pairs, ising, paper, a, open_chain] = params 
    trotter_steps = 1
    if not h_commutes:
        t_ = t
        if t == 0:
            t_ = 1
            print("First order trotter in progress..")
        trotter_steps = math.ceil(abs(j) * t_ * dt / eps)
        print('trotter steps:', trotter_steps, " t:", t)

    # Address needed constants for particular paper
    pseudo_constant_a, mag_constant = 1.0, 1.0
    if paper == 'joel':
        pseudo_constant_a = 4.0
        mag_constant = 2.0

    # Find relevant pairs and see if they can be split evenly
    [even_pairs, odd_pairs] = hf.gen_even_odd_pairs(n, open_chain)
    no_bonds = hf.gen_num_bonds(n, open_chain)

    for step in range(trotter_steps):
        for k in range(n):
            if bg != 0.0:
                if trns:
                    qc.rx(bg * dt * t / (trotter_steps * mag_constant), k + ancilla)
                else:
                    qc.rz(bg * dt * t / (trotter_steps * mag_constant), k + ancilla)

        if no_bonds % 2 == 0:
            # It is possible to split to two noncommuting groups without redundant subroutines
            hf.grouped_three_cnot_evolution(qc, even_pairs, ancilla, j, t, dt, trotter_steps * pseudo_constant_a, ising, a)
            hf.grouped_three_cnot_evolution(qc, odd_pairs, ancilla, j, t, dt, trotter_steps * pseudo_constant_a, ising, a)
        else:
            # Do the normal routine
            for x in total_pairs:
                qc.barrier()
                hf.three_cnot_evolution(qc, x, ancilla, j, t, dt, trotter_steps * pseudo_constant_a, ising, a)



def second_order_trotter(qc, dt, t, ancilla, params):

    [h_commutes, j, eps, spin_constant, n, bg, trns, total_pairs, ising, paper, a, open_chain] = params
    trotter_steps = 1
    if not h_commutes:
        t_ = t
        if t == 0:
            t_ += 1
            print("Second order trotter in progress..")
        trotter_steps = math.ceil(abs(j) * t_ * dt / eps)
        print('trotter steps:', trotter_steps, " t:", t)

    # Address needed constants for particular paper
    pseudo_constant_a = 1.0
    mag_constant = 1.0
    if paper in ['joel']:
        pseudo_constant_a = 4.0
        mag_constant = 2.0

    # Find relevant pairs and see if they can be split evenly 
    [nn_even, nn_odd] = hf.gen_even_odd_pairs(n, open_chain) 
    no_bonds = hf.gen_num_bonds(n, open_chain)

    for step in range(trotter_steps):

        for k in range(n):
            if bg != 0.0:
                if trns:
                    qc.rx(bg * dt * t / (trotter_steps * mag_constant), k + ancilla)
                else:
                    qc.rz(bg * dt * t / (trotter_steps * mag_constant), k + ancilla)

        if no_bonds % 2 == 0: 
            hf.grouped_three_cnot_evolution(qc, nn_even, ancilla, j, t, dt, trotter_steps * pseudo_constant_a * 2.0, ising, a) 
            hf.grouped_three_cnot_evolution(qc, nn_odd, ancilla, j, t, dt, trotter_steps * pseudo_constant_a, ising, a)
            hf.grouped_three_cnot_evolution(qc, nn_even, ancilla, j, t, dt, trotter_steps * pseudo_constant_a * 2.0, ising, a)
        else: 
            for x in reversed(total_pairs[1:]):
                qc.barrier()
                hf.three_cnot_evolution(qc, x, ancilla, j, t, dt, trotter_steps * pseudo_constant_a * 2, ising, a)
            qc.barrier()
            mid = total_pairs[0]
            hf.three_cnot_evolution(qc, mid, ancilla, j, t, dt, trotter_steps * pseudo_constant_a, ising, a)
                   
            for x in total_pairs[1:]:
                qc.barrier()
                hf.three_cnot_evolution(qc, x, ancilla, j, t, dt, trotter_steps * pseudo_constant_a * 2.0, ising, a)

    return 
