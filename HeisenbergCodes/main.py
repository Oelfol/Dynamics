###########################################################################
# main.py
# Part of HeisenbergCodes
# Updated January '21
#
# A collection of calls for example spin chains and calculations.
# Noise model/device details are setup here as well
###########################################################################

###########################################################################
# Device List:
# 'ibmq_rome'
# 'ibmq_valencia'
# 'ibmq_16_melbourne'
# 'ibmq_armonk'
# 'ibmq_ourense'
# 'ibmq_qasm_simulator'
# 'ibmq_vigo'
# 'ibmqx2'
# 'ibmq_bogota'
# 'ibmq_santiago'
# 'ibmq_athens'
# 'ibmq_casablanca'
###########################################################################

import HeisenbergModel as hm
import IBMQSetup as ibmq
import QuantumSimulation as qs
import ClassicalSimulation as cs
import HelpingFunctions as hf
import math
import TimeEvolution as te

###########################################################################
# Choose to run noisy circuit on real hardware (real_sim), or noise model:
real_sim = False

# Set up device params here: (shots is always 50000 unless changed in IBMQSetup.)
# ibmq_ourense is down
device_name = 'ibmq_casablanca'
setup = ibmq.ibmqSetup(dev_name=device_name)
device, noise_model, basis_gates, coupling_map = setup.get_noise_model()
simulator = setup.get_simulator()
dparams = [device, noise_model, basis_gates, simulator, real_sim, coupling_map]
###########################################################################

# TODO obtain new readout arrays since ourense went down, so that hard-coded rm portions can be rewritten
# TODO SPSA calibration in TrialVQE
# TODO locate bug in TrialVQE for ferromagnet example
# TODO make RM optional
###################################################################################################
# Abbreviations:
# (j, coupling constant); (bg, magnetic field); (a, anisotropy jz/j);
# (n, number of sites); (open, whether open-ended chain); (states, number of basis states)
# (unity, whether h-bar/2 == 1 (h-bar == 1 elsewise)); (ising, for ising model);
# (trns; transverse ising); (eps, precision for trotter steps); (dev_params, for running circuit)
###################################################################################################

###################################################################################################
# Settings:
# h-bar / 2 == 1 (Francis) (set unity == True) #
# h-bar == 1 (Tachinno) (set unity == False)
# h-bar == 1 (Joel) (set unity == False)
###################################################################################################

# Testers
# ==================================== TESTERS ==================================================================== >

# TACHINNO FIG 5a  (All Site Magnetization) (ourense)

# model = hm.HeisenbergModel(j=1, bg=0, n=2, p='tachinno', eps=0.2, unity=False, dev_params=dparams,
#                                                                      RMfile='ourense_RM_Jan14_AncillaQubit2.txt')
# model.all_site_magnetization(total_t=35, dt=0.1, psi0=0, hadamard=True)

# ---------------------------------------------------------------------------------------------------------------------

# JOEL FIG 2a  (All Site Magnetization) (casablanca)
# STICKY REMINDER: enable joel in HelpingFunctions.py 

#model = hm.HeisenbergModel(j=1, bg=0, a=0.5, n=6, open=True, eps=0.23, dev_params=dparams, unity=False,  # was .23
#                                                             RMfile='RMArrays/casablanca_RM_Jan14_AncillaQubit2.txt')
#initstate=model.classical_chain.states - 2
#model.all_site_magnetization(total_t=40, dt=0.2, psi0=initstate) #(total_t=80, dt=0.1, psi0=initstate)


# ----------------------------------------------------------------------------------------------------------------------
# TACCHINO FIG 5b (Occupation Probabilities) (ourense)

# This example cannot run anymore without other fixes in helping_functions, due to RM 
# model = hm.HeisenbergModel(j=1, bg=20, n=3, open=True, trns=False, p='tachinno',#
#                    ising=False, eps=0.2, unity=True, dev_params=dparams)
# c = [int(x, 2) for x in ['100', '010', '111']]
# model.occupation_probabilities(total_t=350, dt=0.01, initstate=int('100', 2), chosen_states=c)


# --------------------------------------------------------------------------------------------------------------------------
# TACCHINO FIG 7 (Two Point Correlations) (ourense)
# model = hm.HeisenbergModel(j=-1, bg=20, n=3, open=True, trns=False, ising=False, eps=0.3, unity=False,
#                           dev_params=dparams, RMfile='RMArrays/ourense_RM_Jan14_AncillaQubit2.txt')
# model.two_point_correlations(op_order='xx', total_t=300, dt=.01, pairs=[(0,0)], psi0=int('000', 2))
# model.two_point_correlations(op_order='xx', total_t=30, dt=.01, pairs=[(1,0)], psi0=int('000', 2))
# model.two_point_correlations(op_order='xx', total_t=330, dt=.01, pairs=[(2,0)], psi0=int('000', 2))


# ----------------------------------------------------------------------------------------------------------------------
# TACCHINO FIG 5c (TOTAL MAGNETIZATION) (ourense)
# model = hm.HeisenbergModel(j=1, bg=2, n=2, open=True, trns=True, , ising=True, eps=0.2,
#               unity=True, dev_params=dparams, RMfile='RMArrays/ourense_RM_Jan14_AncillaQubit2.txt')
# model.total_magnetization(total_t=650, dt=0.01, psi0=0)

# ----------------------------------------------------------------------------------------------------------------------
# Francis (ourense)

# Correlation Functions
# model = hm.HeisenbergModel(j=-.84746, bg=0, n=4, a=1.0, open=False, unity=True, eps=0.17, dev_params=dparams,
#                           RMfile='RMArrays/ourense_RM_Jan14_AncillaQubit2.txt')
# model.two_point_correlations(op_order='xx', total_t=400, dt=.01, pairs=[(1, 1), (2, 1), (3, 1)], psi0=0)

# DSF : Ideal circuits MUST to be toggled off in twoPtCorrelationsQ or it will run too slow
# They stay in the code to show trotter error

# Francis ferromagnetic DSF (quantum sim) # set unity == false to compare with analytic curve
#model = qs.QuantumSim(j=-.84746, bg=0, a=1.0, n=4, open=False, unity=False, eps=0.18, dev_params=dparams,
#            RMfile='RMArrays/ourense_RM_Jan14_AncillaQubit2.txt')
#model.dynamical_structure_factor(te.first_order_trotter, 600, 0.01,'x', 'x', 0, [-2 * math.pi, 2 * math.pi],
#            [-0.5, 2])


# Francis ferromagnetic DSF (classical sim)  # set unity == false to compare with analytic curve
#model = cs.ClassicalSpinChain(j=-.84746, bg=0, a=1.0, n=4, open=False, unity=False)
#psi0 = hf.init_spin_state(0, 2 ** 4)
#model.dynamical_structure_factor(600, .01, psi0, 'x', 'x', [-2 * math.pi, 2 * math.pi], [-0.5, 2])

# VQE
model = hm.HeisenbergModel(j=-.84746, bg=0, n=4, a=1.0, open=False, unity=False, eps=0.17, dev_params=dparams,
                           RMfile='RMArrays/casablanca_RM_Jan14_AncillaQubit2.txt')
model.trial_vqe_hardware_efficient(num_updates=300, alpha=.3, gamma=.1, c=.01, a=1, noisy=False)


# =====================================================================================================================>

# Other experiments

# Correlation Functions
#model = hm.HeisenbergModel(j=-.84746, bg=10, n=4, a=10.0, open=False, unity=True, eps=0.17, dev_params=dparams,
#                           RMfile='RMArrays/casablanca_RM_Jan14_AncillaQubit2.txt')
#model.two_point_correlations(op_order='xx', total_t=400, dt=.01, pairs=[(1, 1), (2, 1), (3, 1)], psi0=0)

#ex1
#model = HeisenbergModel(j=1, bg=10, a=10, n=4, open=False, epsilon=0.17)
#model.two_point_correlations(total_t=160, dt=.01, initstate=int('000',2), op_order='xx', pairs=[(0, 0), (1, 0), (2, 0)])


