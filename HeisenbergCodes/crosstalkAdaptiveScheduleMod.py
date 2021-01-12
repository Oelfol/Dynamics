# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
###############################################################################
# Notice of Modification:
# This code has been modified from its original version
# To correctly retrieve gate length values
# Original code also does not accept barriers in the circuit
# Do not include 'measured qubits' in input

# WIP: Code has to be fixed;
# Does not preserve trotter steps or keep barriers yet
##############################################################################

Crosstalk mitigation through adaptive instruction scheduling.
The scheduling algorithm is described in:
Prakash Murali, David C. Mckay, Margaret Martonosi, Ali Javadi Abhari,
Software Mitigation of Crosstalk on Noisy Intermediate-Scale Quantum Computers,
in International Conference on Architectural Support for Programming Languages
and Operating Systems (ASPLOS), 2020.
Please cite the paper if you use this pass.

The method handles crosstalk noise on two-qubit gates. This includes crosstalk
with simultaneous two-qubit and one-qubit gates. The method ignores
crosstalk between pairs of single qubit gates.

The method assumes that all qubits get measured simultaneously whether or not
they need a measurement. This assumption is based on current device properties
and may need to be revised for future device generations.
"""

import math
import operator
from itertools import chain, combinations


try:
    from z3 import Real, Bool, Sum, Implies, And, Or, Not, Optimize

    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library.standard_gates import U1Gate, U2Gate, U3Gate, CXGate
from qiskit.circuit.library.standard_gates import XGate, SXGate, RZGate, IGate
from qiskit.circuit import Measure
from qiskit.circuit.barrier import Barrier
from qiskit.transpiler.exceptions import TranspilerError
from qiskit import Aer, IBMQ
from qiskit.providers.aer import noise

NUM_PREC = 10
TWOQ_XTALK_THRESH = 1.1
ONEQ_XTALK_THRESH = 1.1


def cx_tuple(gate):
    """
    Representation for two-qubit gate
    Note: current implementation assumes that the CX error rates and
    crosstalk behavior are independent of gate direction
    Mod: Function moved out of class for clarity
    """
    physical_q_0 = gate.qargs[0].index
    physical_q_1 = gate.qargs[1].index
    r_0 = min(physical_q_0, physical_q_1)
    r_1 = max(physical_q_0, physical_q_1)
    return r_0, r_1


def singleq_tuple(gate):
    """
    Representation for single-qubit gate
    Mod: Function moved out of class for clarity
    """
    physical_q_0 = gate.qargs[0].index
    tup = (physical_q_0,)
    return tup


def gate_tuple(gate):
    """
    Representation for gate
    Mod: Function moved out of class for clarity
    """
    if len(gate.qargs) == 2:
        return cx_tuple(gate)
    else:
        return singleq_tuple(gate)


def r2f(val):
    """
    Convert Z3 Real to Python float
    """
    if val is None:
        return 0
    else:
        return float(val.as_decimal(16).rstrip('?'))


def floatfix(value):
    v1 = 0
    if value == 1.0:
        v1 += 0.9999
    else:
        v1 += value
    return v1


class CrosstalkAdaptiveSchedule(TransformationPass):
    """Crosstalk mitigation through adaptive instruction scheduling."""

    def __init__(self, backend_prop, crosstalk_prop, provider, weight_factor=0.5, measured_qubits=None, dev_name=''):
        """CrosstalkAdaptiveSchedule initializer.

        Args:
            backend_prop (BackendProperties): backend properties object
            crosstalk_prop (dict): crosstalk properties object
                crosstalk_prop[g1][g2] specifies the conditional error rate of
                g1 when g1 and g2 are executed simultaneously.
                g1 should be a two-qubit tuple of the form (x,y) where x and y are physical
                qubit ids. g2 can be either two-qubit tuple (x,y) or single-qubit tuple (x).
                We currently ignore crosstalk between pairs of single-qubit gates.
                Gate pairs which are not specified are assumed to be crosstalk free.

                Example::

                    crosstalk_prop = {(0, 1) : {(2, 3) : 0.2, (2) : 0.15},
                                                (4, 5) : {(2, 3) : 0.1},
                                                (2, 3) : {(0, 1) : 0.05, (4, 5): 0.05}}

                The keys of the crosstalk_prop are tuples for ordered tuples for CX gates
                e.g., (0, 1) corresponding to CX 0, 1 in the hardware.
                Each key has an associated value dict which specifies the conditional error rates
                with nearby gates e.g., ``(0, 1) : {(2, 3) : 0.2, (2) : 0.15}`` means that
                CNOT 0, 1 has an error rate of 0.2 when it is executed in parallel with CNOT 2,3
                and an error rate of 0.15 when it is executed in parallel with a single qubit
                gate on qubit 2.
            weight_factor (float): weight of gate error/crosstalk terms in the objective
                :math:`weight_factor*fidelities + (1-weight_factor)*decoherence errors`.
                Weight can be varied from 0 to 1, with 0 meaning that only decoherence
                errors are optimized and 1 meaning that only crosstalk errors are optimized.
                weight_factor should be tuned per application to get the best results.
            measured_qubits (list): a list of qubits that will be measured in a particular circuit.
                This arg need not be specified for circuits which already include measure gates.
                The arg is useful when a subsequent module such as state_tomography_circuits
                inserts the measure gates. If CrosstalkAdaptiveSchedule is made aware of those
                measurements, it is included in the optimization.
            dev_name(string): the device name (modification)
        Raises:
            ImportError: if unable to import z3 solver

        """
        super().__init__()
        self.backend_prop = backend_prop
        self.crosstalk_prop = crosstalk_prop
        self.weight_factor = weight_factor
        if measured_qubits is None:
            self.input_measured_qubits = []
        else:
            self.input_measured_qubits = measured_qubits

        # Some gates are not in this backends properties; saving in case
        """
        self.bp_u1_err = {}
        self.bp_u1_dur = {}
        self.bp_u2_err = {}
        self.bp_u2_dur = {}
        self.bp_u3_err = {}
        self.bp_u3_dur = {}
        """
        self.bp_x_err = {}
        self.bp_sx_err = {}
        self.bp_rz_err = {}
        self.bp_id_err = {}
        self.bp_x_dur = {}
        self.bp_sx_dur = {}
        self.bp_rz_dur = {}
        self.bp_id_dur = {}
        #
        self.bp_cx_err = {}
        self.bp_cx_dur = {}
        self.bp_t1_time = {}
        self.bp_t2_time = {}
        self.gate_id = {}
        self.gate_start_time = {}
        self.gate_duration = {}
        self.gate_fidelity = {}
        self.overlap_amounts = {}
        self.overlap_indicator = {}
        self.qubit_lifetime = {}
        self.dag_overlap_set = {}
        self.xtalk_overlap_set = {}
        self.opt = Optimize()
        self.measured_qubits = []
        self.measure_start = None
        self.last_gate_on_qubit = None
        self.first_gate_on_qubit = None
        self.fidelity_terms = []
        self.coherence_terms = []
        self.model = None
        self.dag = None
        # modifications:
        self.dev_name = dev_name
        self.provider = provider
        self.parse_backend_properties()

    def powerset(self, iterable):
        """
        Finds the set of all subsets of the given iterable
        This function is used to generate constraints for the Z3 optimization
        """
        l_s = list(iterable)
        return chain.from_iterable(combinations(l_s, r) for r in range(len(l_s) + 1))

    def get_gate_lengths_cnot(self, qid, name): # Retrieves
        # This function is a modification of the original code, for two-qubit gates (merge these functions later TODO)
        from qiskit.providers.aer import noise
        device = self.provider.get_backend(self.dev_name)
        properties = device.properties()
        gate_lengths = noise.device.parameters.gate_length_values(properties)
        gate_length = 0
        for tup in gate_lengths:
            if tup[0] == name:
                if tup[1][0] == qid[0] and tup[1][1] == qid[1]:
                    gate_length += tup[2]
        return gate_length

    def get_gate_lengths_sing(self, qid, name): # Retrieving
        # This function is a modification of the original code, for one-qubit gates (merge these functions later TODO)
        from qiskit.providers.aer import noise
        device = self.provider.get_backend(self.dev_name)
        properties = device.properties()
        gate_lengths = noise.device.parameters.gate_length_values(properties)
        gate_length = 0
        for tup in gate_lengths:
            if tup[0] == name and len(tup[1]) == 1:
                if tup[1][0] == qid:
                    gate_length += tup[2]
        return gate_length

    def get_gate_error_cnot(self, qid, name): # Retrieving
        # This function is a modification of the original code, for two-qubit gates
        device = self.provider.get_backend(self.dev_name)
        properties = device.properties()
        gate_err_list = noise.device.parameters.gate_error_values(properties)
        gate_err = 0
        for tup in gate_err_list:
            if tup[0] == name:
                if tup[1][0] == qid[0] and tup[1][1] == qid[1]:
                    gate_err += tup[2]
        return gate_err

    def get_gate_error_sing(self, qid, name): # Retrieving
        # This function is a modification of the original code, for single-qubit gates
        # Note: Rz error rate is not reported on backend for some reason.
        device = self.provider.get_backend(self.dev_name)
        properties = device.properties()
        gate_err_list = noise.device.parameters.gate_error_values(properties)
        gate_err = 0
        for tup in gate_err_list:
            if tup[0] == name:
                if tup[1][0] == qid:
                    gate_err += tup[2]
        return gate_err

    def parse_backend_properties(self):
        """
        This function assumes that gate durations and coherence times
        are in seconds in backend.properties()
        This function converts gate durations and coherence times to
        nanoseconds.
        """
        backend_prop = self.backend_prop
        scale = 10 ** 9
        for qid in range(len(backend_prop.qubits)):
            self.bp_t1_time[qid] = int(backend_prop.t1(qid) * scale)
            self.bp_t2_time[qid] = int(backend_prop.t2(qid) * scale)

            """ Some gates not needed for this backend; store here in case 
            self.bp_u1_dur[qid] = int(self.get_gate_lengths_sing(qid, 'u1') * scale)
            u1_err = floatfix(self.get_gate_error_sing(qid, 'u1'))
            self.bp_u1_err = round(u1_err, NUM_PREC)

            self.bp_u2_dur[qid] = int(self.get_gate_lengths_sing(qid, 'u2') * scale)
            u2_err = floatfix(self.get_gate_error_sing(qid, 'u2'))
            self.bp_u2_err = round(u2_err, NUM_PREC)

            self.bp_u3_dur[qid] = int(self.get_gate_lengths_sing(qid, 'u3') * scale)
            u3_err = floatfix(self.get_gate_error_sing(qid, 'u3'))
            self.bp_u3_err = round(u3_err, NUM_PREC)
            """

            self.bp_x_dur[qid] = int(self.get_gate_lengths_sing(qid, 'x') * scale)
            x_err = floatfix(self.get_gate_error_sing(qid, 'x'))
            self.bp_x_err[qid] = round(x_err, NUM_PREC)

            self.bp_sx_dur[qid] = int(self.get_gate_lengths_sing(qid, 'sx') * scale)
            sx_err = floatfix(self.get_gate_error_sing(qid, 'sx'))
            self.bp_sx_err[qid] = round(sx_err, NUM_PREC)

            self.bp_rz_dur[qid] = int(self.get_gate_lengths_sing(qid, 'rz') * scale)
            rz_err = floatfix(self.get_gate_error_sing(qid, 'rz'))
            self.bp_rz_err[qid] = round(rz_err, NUM_PREC)

            self.bp_id_dur[qid] = int(self.get_gate_lengths_sing(qid, 'id') * scale)
            id_err = floatfix(self.get_gate_error_sing(qid, 'id'))
            self.bp_id_err[qid] = round(id_err, NUM_PREC)

        for ginfo in backend_prop.gates:
            if ginfo.gate == 'cx':
                q_0 = ginfo.qubits[0]
                q_1 = ginfo.qubits[1]
                cx_tup = (min(q_0, q_1), max(q_0, q_1))
                self.bp_cx_dur[cx_tup] = int(self.get_gate_lengths_cnot(cx_tup, 'cx') * (scale))
                cx_err = floatfix(self.get_gate_error_cnot(cx_tup, 'cx'))
                self.bp_cx_err[cx_tup] = round(cx_err, NUM_PREC)

    def assign_gate_id(self, dag):
        """ ID for each gate"""
        idx = 0
        for gate in dag.gate_nodes():
            self.gate_id[gate] = idx
            idx += 1

    def extract_dag_overlap_sets(self, dag):
        """
        Gate A, B are overlapping if A is neither a descendant nor an ancestor of B.
        Currenty overlaps (A,B) are considered when A is a 2q gate and B is either 2q or 1q gate.
        """
        for gate in dag.two_qubit_ops():
            overlap_set = []
            descendants = dag.descendants(gate)
            ancestors = dag.ancestors(gate)
            for tmp_gate in dag.gate_nodes():
                if tmp_gate == gate:
                    continue
                if tmp_gate in descendants:
                    continue
                if tmp_gate in ancestors:
                    continue
                overlap_set.append(tmp_gate)
            self.dag_overlap_set[gate] = overlap_set

    def is_significant_xtalk(self, gate1, gate2):
        """
        Given two conditional gate error rates
        check if there is high crosstalk by comparing with independent error rates.
        """
        gate1_tup = gate_tuple(gate1)
        if len(gate2.qargs) == 2:
            gate2_tup = gate_tuple(gate2)
            independent_err_g_1 = self.bp_cx_err[gate1_tup]
            independent_err_g_2 = self.bp_cx_err[gate2_tup]
            rg_1 = self.crosstalk_prop[gate1_tup][gate2_tup] / independent_err_g_1
            rg_2 = self.crosstalk_prop[gate2_tup][gate1_tup] / independent_err_g_2
            if rg_1 > TWOQ_XTALK_THRESH or rg_2 > TWOQ_XTALK_THRESH:
                return True
        else:
            gate2_tup = gate_tuple(gate2)
            independent_err_g_1 = self.bp_cx_err[gate1_tup]
            rg_1 = self.crosstalk_prop[gate1_tup][gate2_tup] / independent_err_g_1
            if rg_1 > ONEQ_XTALK_THRESH:
                return True
        return False

    def extract_crosstalk_relevant_sets(self):
        """
        Extract the set of program gates which potentially have crosstalk noise
        """
        for gate in self.dag_overlap_set:
            self.xtalk_overlap_set[gate] = []
            tup_g = gate_tuple(gate)
            if tup_g not in self.crosstalk_prop:
                continue
            for par_g in self.dag_overlap_set[gate]:
                tup_par_g = gate_tuple(par_g)
                if tup_par_g in self.crosstalk_prop[tup_g]:
                    if self.is_significant_xtalk(gate, par_g):
                        if par_g not in self.xtalk_overlap_set[gate]:
                            self.xtalk_overlap_set[gate].append(par_g)

    def create_z3_vars(self):
        """
        Setup the variables required for Z3 optimization
        """
        for gate in self.dag.gate_nodes():
            t_var_name = 't_' + str(self.gate_id[gate])
            d_var_name = 'd_' + str(self.gate_id[gate])
            f_var_name = 'f_' + str(self.gate_id[gate])
            self.gate_start_time[gate] = Real(t_var_name)
            self.gate_duration[gate] = Real(d_var_name)
            self.gate_fidelity[gate] = Real(f_var_name)

        for gate in self.xtalk_overlap_set:
            self.overlap_indicator[gate] = {}
            self.overlap_amounts[gate] = {}

        for g_1 in self.xtalk_overlap_set:
            for g_2 in self.xtalk_overlap_set[g_1]:
                if len(g_2.qargs) == 2 and g_1 in self.overlap_indicator[g_2]:
                    self.overlap_indicator[g_1][g_2] = self.overlap_indicator[g_2][g_1]
                    self.overlap_amounts[g_1][g_2] = self.overlap_amounts[g_2][g_1]
                else:
                    # Indicator variable for overlap of g_1 and g_2
                    var_name1 = 'olp_ind_' + str(self.gate_id[g_1]) + '_' + str(self.gate_id[g_2])
                    self.overlap_indicator[g_1][g_2] = Bool(var_name1)
                    var_name2 = 'olp_amnt_' + str(self.gate_id[g_1]) + '_' + str(self.gate_id[g_2])
                    self.overlap_amounts[g_1][g_2] = Real(var_name2)

        active_qubits_list = []
        for gate in self.dag.gate_nodes():
            for q in gate.qargs:
                active_qubits_list.append(q.index)
        for active_qubit in list(set(active_qubits_list)):
            q_var_name = 'l_' + str(active_qubit)
            self.qubit_lifetime[active_qubit] = Real(q_var_name)

        meas_q = []
        for node in self.dag.op_nodes():
            if isinstance(node.op, Measure):
                meas_q.append(node.qargs[0].index)

        self.measured_qubits = list(set(self.input_measured_qubits).union(set(meas_q)))
        self.measure_start = Real('meas_start')

    def basic_bounds(self):
        """
        Basic variable bounds for optimization
        """
        for gate in self.gate_start_time:
            self.opt.add(self.gate_start_time[gate] >= 0)
        for gate in self.gate_duration:
            q_0 = gate.qargs[0].index
            dur = 0
            if isinstance(gate.op, XGate):
                dur = self.bp_x_dur[q_0]
            elif isinstance(gate.op, SXGate):
                dur = self.bp_sx_dur[q_0]
            elif isinstance(gate.op, RZGate):
                dur = self.bp_rz_dur[q_0]
            elif isinstance(gate.op, IGate):
                dur = self.bp_id_dur[q_0]
            elif isinstance(gate.op, CXGate):
                dur = self.bp_cx_dur[cx_tuple(gate)]

            self.opt.add(self.gate_duration[gate] == dur)

            """ #Keep: 
            if isinstance(gate.op, U1Gate):
                dur = self.bp_u1_dur[q_0]
                #self.opt.add(self.gate_duration[gate] == dur)
            elif isinstance(gate.op, U2Gate):
                dur = self.bp_u2_dur[q_0]
                #self.opt.add(self.gate_duration[gate] == dur)
            elif isinstance(gate.op, U3Gate):
                dur = self.bp_u3_dur[q_0]
                #self.opt.add(self.gate_duration[gate] == dur)
            """

    def scheduling_constraints(self):
        """
        DAG scheduling constraints optimization
        Sets overlap indicator variables
        """
        for gate in self.gate_start_time:
            for dep_gate in self.dag.successors(gate):
                if not dep_gate.type == 'op':
                    continue
                if isinstance(dep_gate.op, Measure):
                    continue
                if isinstance(dep_gate.op, Barrier):
                    continue
                fin_g = self.gate_start_time[gate] + self.gate_duration[gate]
                self.opt.add(self.gate_start_time[dep_gate] > fin_g)
        for g_1 in self.xtalk_overlap_set:
            for g_2 in self.xtalk_overlap_set[g_1]:
                if len(g_2.qargs) == 2 and self.gate_id[g_1] > self.gate_id[g_2]:
                    # Symmetry breaking: create only overlap variable for a pair
                    # of gates
                    continue
                s_1 = self.gate_start_time[g_1]
                f_1 = s_1 + self.gate_duration[g_1]
                s_2 = self.gate_start_time[g_2]
                f_2 = s_2 + self.gate_duration[g_2]
                # This constraint enforces full or zero overlap between two gates
                before = (f_1 < s_2)
                after = (f_2 < s_1)
                overlap1 = And(s_2 <= s_1, f_1 <= f_2)
                overlap2 = And(s_1 <= s_2, f_2 <= f_1)
                self.opt.add(Or(before, after, overlap1, overlap2))
                intervals_overlap = And(s_2 <= f_1, s_1 <= f_2)
                self.opt.add(self.overlap_indicator[g_1][g_2] == intervals_overlap)

    def fidelity_constraints(self):
        """
        Set gate fidelity based on gate overlap conditions
        """
        for gate in self.gate_start_time:
            q_0 = gate.qargs[0].index
            no_xtalk = False
            if gate not in self.xtalk_overlap_set:
                no_xtalk = True
            elif not self.xtalk_overlap_set[gate]:
                no_xtalk = True
            if no_xtalk:
                fid = 0
                if isinstance(gate.op, XGate):
                    fid = math.log(1.0 - self.bp_x_err[q_0])
                elif isinstance(gate.op, SXGate):
                    fid = math.log(1.0 - self.bp_sx_err[q_0])
                elif isinstance(gate.op, RZGate):
                    fid = math.log(1.0) #- self.bp_rz_err[q_0]) ##### Rz error is unreported on backend & breaks code
                elif isinstance(gate.op, IGate):
                    fid = math.log(1.0 - self.bp_id_err[q_0])
                elif isinstance(gate.op, CXGate):
                    fid = math.log(1.0 - self.bp_cx_err[cx_tuple(gate)])
                self.opt.add(self.gate_fidelity[gate] == round(fid, NUM_PREC))

                """ # Keep: 
                if isinstance(gate.op, U1Gate):
                    fid = math.log(1.0)
                    #self.opt.add(self.gate_fidelity[gate] == round(fid, NUM_PREC))
                elif isinstance(gate.op, U2Gate):
                    fid = math.log(1.0 - self.bp_u2_err[q_0])
                    #self.opt.add(self.gate_fidelity[gate] == round(fid, NUM_PREC))
                elif isinstance(gate.op, U3Gate):
                    fid = math.log(1.0 - self.bp_u3_err[q_0])
                    #self.opt.add(self.gate_fidelity[gate] == round(fid, NUM_PREC))
                elif isinstance(gate.op, CXGate):
                    fid = math.log(1.0 - self.bp_cx_err[cx_tuple(gate)])
                self.opt.add(self.gate_fidelity[gate] == round(fid, NUM_PREC))
                """

            else:
                comb = list(self.powerset(self.xtalk_overlap_set[gate]))
                xtalk_set = set(self.xtalk_overlap_set[gate])
                for item in comb:
                    on_set = item
                    off_set = [i for i in xtalk_set if i not in on_set]
                    clauses = []
                    for tmpg in on_set:
                        clauses.append(self.overlap_indicator[gate][tmpg])
                    for tmpg in off_set:
                        clauses.append(Not(self.overlap_indicator[gate][tmpg]))
                    err = 0
                    if not on_set:
                        err = self.bp_cx_err[cx_tuple(gate)]
                    elif len(on_set) == 1:
                        on_gate = on_set[0]
                        err = self.crosstalk_prop[gate_tuple(gate)][gate_tuple(on_gate)]
                    else:
                        err_list = []
                        for on_gate in on_set:
                            tmp_prop = self.crosstalk_prop[gate_tuple(gate)]
                            err_list.append(tmp_prop[gate_tuple(on_gate)])
                        err = max(err_list)
                    err = floatfix(err)
                    val = round(math.log(1.0 - err), NUM_PREC)
                    self.opt.add(Implies(And(*clauses), self.gate_fidelity[gate] == val))

    def coherence_constraints(self):
        """
        Set decoherence errors based on qubit lifetimes
        """
        self.last_gate_on_qubit = {}
        for gate in self.dag.topological_op_nodes():
            if isinstance(gate.op, Measure):
                continue
            if isinstance(gate.op, Barrier):
                continue
            if len(gate.qargs) == 1:
                q_0 = gate.qargs[0].index
                self.last_gate_on_qubit[q_0] = gate
            else:
                q_0 = gate.qargs[0].index
                q_1 = gate.qargs[1].index
                self.last_gate_on_qubit[q_0] = gate
                self.last_gate_on_qubit[q_1] = gate

        self.first_gate_on_qubit = {}
        for gate in self.dag.topological_op_nodes():
            if isinstance(gate.op, Measure):   # added
                continue
            if isinstance(gate.op, Barrier):   # added
                continue
            if len(gate.qargs) == 1:
                q_0 = gate.qargs[0].index
                if q_0 not in self.first_gate_on_qubit:
                    self.first_gate_on_qubit[q_0] = gate
            else:
                q_0 = gate.qargs[0].index
                q_1 = gate.qargs[1].index
                if q_0 not in self.first_gate_on_qubit:
                    self.first_gate_on_qubit[q_0] = gate
                if q_1 not in self.first_gate_on_qubit:
                    self.first_gate_on_qubit[q_1] = gate

        for q in self.last_gate_on_qubit:
            g_last = self.last_gate_on_qubit[q]
            g_first = self.first_gate_on_qubit[q]
            finish_time = self.gate_start_time[g_last] + self.gate_duration[g_last]
            start_time = self.gate_start_time[g_first]
            if q in self.measured_qubits:
                self.opt.add(self.measure_start >= finish_time)
                self.opt.add(self.qubit_lifetime[q] == self.measure_start - start_time)
            else:
                # All qubits get measured simultaneously whether or not they need a measurement
                self.opt.add(self.measure_start >= finish_time)
                self.opt.add(self.qubit_lifetime[q] == finish_time - start_time)

    def objective_function(self):
        """
        Objective function is a weighted combination of gate errors and decoherence errors
        """
        self.fidelity_terms = [self.gate_fidelity[gate] for gate in self.gate_fidelity]
        self.coherence_terms = []
        for q in self.qubit_lifetime:
            val = -self.qubit_lifetime[q] / min(self.bp_t1_time[q], self.bp_t2_time[q])
            self.coherence_terms.append(val)

        all_terms = []
        for item in self.fidelity_terms:
            all_terms.append(self.weight_factor * item)
        for item in self.coherence_terms:
            all_terms.append((1 - self.weight_factor) * item)
        self.opt.maximize(Sum(all_terms))

    def extract_solution(self):
        """
        Extract gate start and finish times from Z3 solution
        """
        self.model = self.opt.model()
        result = {}
        for tmpg in self.gate_start_time:
            start = r2f(self.model[self.gate_start_time[tmpg]])
            dur = r2f(self.model[self.gate_duration[tmpg]])
            result[tmpg] = (start, start + dur)
        return result

    def solve_optimization(self):
        """
        Setup and solve a Z3 optimization for finding the best schedule
        """
        self.opt = Optimize()
        self.create_z3_vars()
        self.basic_bounds()
        self.scheduling_constraints()
        self.fidelity_constraints()
        self.coherence_constraints()
        self.objective_function()

        # Solve step
        self.opt.check()

        # Extract the schedule computed by Z3
        result = self.extract_solution()
        return result

    def check_dag_dependency(self, gate1, gate2):
        """
        gate2 is a DAG dependent of gate1 if it is a descendant of gate1
        """
        return gate2 in self.dag.descendants(gate1)

    def check_xtalk_dependency(self, t_1, t_2):
        """
        Check if two gates have a crosstalk dependency.
        We do not consider crosstalk between pairs of single qubit gates.
        """
        g_1 = t_1[0]
        s_1 = t_1[1]
        f_1 = t_1[2]
        g_2 = t_2[0]
        s_2 = t_2[1]
        f_2 = t_2[2]
        if len(g_1.qargs) == 1 and len(g_2.qargs) == 1:
            return False, ()
        if s_2 <= f_1 and s_1 <= f_2:
            # Z3 says it's ok to overlap these gates,
            # so no xtalk dependency needs to be checked
            return False, ()
        else:
            # Assert because we are iterating in Z3 gate start time order,
            # so if two gates are not overlapping, then the second gate has to
            # start after the first gate finishes
            assert s_2 >= f_1
            # Not overlapping, but we care about this dependency
            if len(g_1.qargs) == 2 and len(g_2.qargs) == 2:
                if g_2 in self.xtalk_overlap_set[g_1]:
                    cx1 = cx_tuple(g_1)
                    cx2 = cx_tuple(g_2)
                    barrier = tuple(sorted([cx1[0], cx1[1], cx2[0], cx2[1]]))
                    return True, barrier
            elif len(g_1.qargs) == 1 and len(g_2.qargs) == 2:
                if g_1 in self.xtalk_overlap_set[g_2]:
                    singleq = gate_tuple(g_1)
                    cx1 = cx_tuple(g_2)
                    print(singleq, cx1)
                    barrier = tuple(sorted([singleq, cx1[0], cx1[1]]))
                    return True, barrier
            elif len(g_1.qargs) == 2 and len(g_2.qargs) == 1:
                if g_2 in self.xtalk_overlap_set[g_1]:
                    singleq = gate_tuple(g_2)
                    cx1 = cx_tuple(g_1)
                    barrier = tuple(sorted([singleq, cx1[0], cx1[1]]))
                    return True, barrier
            # Not overlapping, and we don't care about xtalk between these two gates
            return False, ()

    def filter_candidates(self, candidates, layer, layer_id, triplet):
        """
        For a gate G and layer L,
        L is a candidate layer for G if no gate in L has a DAG dependency with G,
        and if Z3 allows gates in L and G to overlap.
        """
        curr_gate = triplet[0]
        for prev_triplet in layer:
            prev_gate = prev_triplet[0]
            is_dag_dep = self.check_dag_dependency(prev_gate, curr_gate)
            is_xtalk_dep, _ = self.check_xtalk_dependency(prev_triplet, triplet)
            if is_dag_dep or is_xtalk_dep:
                # If there is a DAG dependency, we can't insert in any previous layer
                # If there is Xtalk dependency, we can (in general) insert in previous layers,
                # but since we are iterating in the order of gate start times,
                # we should only insert this gate in subsequent layers
                for i in range(layer_id + 1):
                    if i in candidates:
                        candidates.remove(i)
            return candidates

    def find_layer(self, layers, triplet):
        """
        Find the appropriate layer for a gate
        """
        candidates = list(range(len(layers)))
        for i, layer in enumerate(layers):
            candidates = self.filter_candidates(candidates, layer, i, triplet)
        if not candidates:
            return len(layers)
            # Open a new layer
        else:
            return max(candidates)
            # Latest acceptable layer, right-alignment

    def generate_barriers(self, layers):
        """
        For each gate g, see if a barrier is required to serialize it with
        some previously processed gate
        """
        barriers = []
        for i, layer in enumerate(layers):
            barriers.append(set())
            if i == 0:
                continue
            for t_2 in layer:
                for j in range(i):
                    prev_layer = layers[j]
                    for t_1 in prev_layer:
                        is_dag_dep = self.check_dag_dependency(t_1[0], t_2[0])
                        is_xtalk_dep, curr_barrier = self.check_xtalk_dependency(t_1, t_2)
                        if is_dag_dep:
                            # Don't insert a barrier since there is a DAG dependency
                            continue
                        if is_xtalk_dep:
                            # Insert a barrier for this layer
                            barriers[-1].add(curr_barrier)
        return barriers

    def create_updated_dag(self, layers, barriers):
        """
        Given a set of layers and barries, construct a new dag
        """

        new_dag = DAGCircuit()
        for qreg in self.dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in self.dag.cregs.values():
            new_dag.add_creg(creg)
        canonical_register = new_dag.qregs['q']
        for i, layer in enumerate(layers):
            curr_barriers = barriers[i]
            for b in curr_barriers:
                current_qregs = []
                for idx in b:
                    current_qregs.append(canonical_register[idx])
                new_dag.apply_operation_back(Barrier(len(b)), current_qregs, [])
            for triplet in layer:
                gate = triplet[0]
                new_dag.apply_operation_back(gate.op, gate.qargs, gate.cargs)

        for node in self.dag.op_nodes():
            if isinstance(node.op, Measure):
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

        # modification for original barriers
        #for node in self.dag.op_nodes():
        #    if isinstance(node.op, Barrier):
        #        new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

        return new_dag

    def enforce_schedule_on_dag(self, input_gate_times):
        """
        Z3 outputs start times for each gate.
        Some gates need to be serialized to implement the Z3 schedule.
        This function inserts barriers to implement those serializations
        """
        gate_times = []
        for key in input_gate_times:
            gate_times.append((key, input_gate_times[key][0], input_gate_times[key][1]))

        # Sort gates by start time
        sorted_gate_times = sorted(gate_times, key=operator.itemgetter(1))
        layers = []

        # Construct a set of layers. Each layer has a set of gates that
        # are allowed to fire in parallel according to Z3
        for triplet in sorted_gate_times:
            layer_idx = self.find_layer(layers, triplet)
            if layer_idx == len(layers):
                layers.append([triplet])
            else:
                layers[layer_idx].append(triplet)

        # Insert barries if necessary to enforce the above layers
        barriers = self.generate_barriers(layers)
        new_dag = self.create_updated_dag(layers, barriers)
        return new_dag

    def reset(self):
        """
        Reset variables
        """
        self.gate_id = {}
        self.gate_start_time = {}
        self.gate_duration = {}
        self.gate_fidelity = {}
        self.overlap_amounts = {}
        self.overlap_indicator = {}
        self.qubit_lifetime = {}
        self.dag_overlap_set = {}
        self.xtalk_overlap_set = {}
        self.measured_qubits = []
        self.measure_start = None
        self.last_gate_on_qubit = None
        self.first_gate_on_qubit = None
        self.fidelity_terms = []
        self.coherence_terms = []
        self.model = None

    def run(self, dag):
        """
        Main scheduling function
        """
        if not HAS_Z3:
            raise TranspilerError('z3-solver is required to use CrosstalkAdaptiveSchedule. '
                                  'To install, run "pip install z3-solver".')
        self.dag = dag

        # process input program
        self.assign_gate_id(self.dag)
        self.extract_dag_overlap_sets(self.dag)
        self.extract_crosstalk_relevant_sets()

        # setup and solve a Z3 optimization
        z3_result = self.solve_optimization()

        # post-process to insert barriers
        new_dag = self.enforce_schedule_on_dag(z3_result)
        self.reset()
        return new_dag
