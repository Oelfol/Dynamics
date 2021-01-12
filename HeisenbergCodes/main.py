###########################################################################
# main.py
# Part of HeisenbergCodes
# Updated January '21
#
# A collection of calls for example spin chains and calculations.
# Noise model/device details are setup here as well

# WIP
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

###########################################################################
# Choose to run noisy circuit on real hardware (real_sim), or noise model:
real_sim = False

# Set up device params here: (shots is always 50000 unless changed in IBMQSetup.)
device_name = 'ibmq_athens'
setup = ibmq.ibmqSetup(dev_name=device_name)
device, noise_model, basis_gates, coupling_map, crosstalk_props, dev_name, provider = setup.get_noise_model()
simulator = setup.get_simulator()
dparams = [device, noise_model, basis_gates, simulator, crosstalk_props, real_sim, dev_name, provider]
###########################################################################

###################################################################################################
# Abbreviations:
# (j, coupling constant); (bg, magnetic field); (a, anisotropy jz/j);
# (n, number of sites); (open, whether open-ended chain); (states, number of basis states)
# (unity, whether h-bar/2 == 1 (h-bar == 1 elsewise)); (ising, for ising model);
# (trns; transverse ising); (p, for settings related to examples from a specific paper 'p')
# (eps, precision for trotter steps); (dev_params, for running circuit)
###################################################################################################

# Testers from TestingOldTrotterCode:
# 1/ 12 are verified without crosstalk / readout code incorporated
# TODO this week finish mitigation and verify
# TODO this week finish structure factor and verify
# ==================================== TESTERS ==================================================================== >

# TACHINNO FIG 5a  (All Site Magnetization)

#model = hm.HeisenbergModel(j=1, bg=0, n=2, p='tachinno', eps=0.2, unity=True, dev_params=dparams)
#model.all_site_magnetization(total_t=35, dt=0.1, psi0=0, hadamard=True)

# ---------------------------------------------------------------------------------------------------------------------

# JOEL FIG 2a  (All Site Magnetization)

#model = hm.HeisenbergModel(j=1, bg=0, a=0.5, n=5, open=True, p='joel', eps=0.4, dev_params=dparams)
#initstate=model.classical_chain.states - 2
#model.all_site_magnetization(total_t=80, dt=0.1, psi0=initstate)

# ----------------------------------------------------------------------------------------------------------------------
# TACCHINO FIG 5b (Occupation Probabilities)

#model = hm.HeisenbergModel(j=1, bg=20, n=3, open=True, trns=False, p='tachinno',#
#                    ising=False, eps=0.2, unity=True, dev_params=dparams)
#c = [int(x, 2) for x in ['100', '010', '111']]
#model.occupation_probabilities(total_t=350, dt=0.01, initstate=int('100', 2), chosen_states=c)


# --------------------------------------------------------------------------------------------------------------------------
# TACCHINO FIG 7 (Two Point Correlations)
#model = hm.HeisenbergModel(j=-1, bg=20, n=3, open=True, trns=False, p='tachinno',
#                 ising=False, eps=0.2, unity=True, dev_params=dparams)
##model.two_point_correlations(op_order='xx', total_t=330, dt=.01, pairs=[(0,0)], psi0=int('000', 2))
#model.two_point_correlations(op_order='xx', total_t=330, dt=.01, pairs=[(1,0)], psi0=int('000', 2))
#model.two_point_correlations(op_order='xx', total_t=330, dt=.01, pairs=[(2,0)], psi0=int('000', 2))


# ----------------------------------------------------------------------------------------------------------------------
# TACCHINO FIG 5c (TOTAL MAGNETIZATION)
# model = hm.HeisenbergModel(j=1, bg=2, n=2, open=True, trns=True, p='tachinno', ising=True, eps=0.2, unity=True, dev_params=dparams)
# model.total_magnetization(total_t=650, dt=0.01, psi0=0)

# ----------------------------------------------------------------------------------------------------------------------
# try changing this epsilon again TODO
# Would be good to choose either 1st or second order from here, TODO
model = hm.HeisenbergModel(j=-.84746, bg=0, n=4, a=1.0, open=False, unity=True, p='francis', eps=0.1, dev_params=dparams)
model.two_point_correlations(op_order='xx', total_t=10, dt=.01, pairs=[(1, 1)], psi0=0) #pairs=[(1, 1), (2, 1), (3, 1)]


# =====================================================================================================================>


# ============================================ RM TESTERS =============================================================>
# Calls for recording counts in ReadoutMitigation.py

#readout_mitigation_circuits_all(5, 50000, "santiago_RM_Oct20_AllQubits.txt")
# Used for Tachinno 5a
#readout_mitigation_circuits_ancilla(5, 100000, "santiago_RM_Oct20_AncillaQubit2.txt", qubit=2)
# ancilla qubit 2 --- > hardware qubit # 2

# Used for Joel 6-site problem:
#readout_mitigation_circuits_ancilla(6, 50000, "santiago_RM_Oct20_AncillaQubit2.txt", qubit=2)
# consider making this a 5-site problem
