# ibmq_setup.py
# Helper function for setting up noise model.

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import noise
from qiskit import Aer, IBMQ
IBMQ.disable_account()
# ========================================= IBM account and noise model setup ======================================== >

TOKEN = ''
provider = IBMQ.enable_account(TOKEN, hub='ibm-q', group='open', project='main')
simulator = Aer.get_backend('qasm_simulator')


class ibmqSetup():
    def __init__(self, sim=True, dev_name='', shots=50000):
        self.sim = sim
        self.shots = shots
        self.dev_name = dev_name

    def get_noise_model(self):
        # regular noise model from the backend

        device = provider.get_backend(self.dev_name)
        properties = device.properties()
        gate_lengths = noise.device.parameters.gate_length_values(properties)
        noise_model = NoiseModel.from_backend(properties, gate_lengths=gate_lengths)
        basis_gates = noise_model.basis_gates
        coupling_map = device.configuration().coupling_map
        return device, noise_model, basis_gates, coupling_map

    def get_device(self):
        return provider.get_backend(self.dev_name)

    def get_simulator(self):
        return Aer.get_backend('qasm_simulator')
