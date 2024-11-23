import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCMT, ZGate, XGate
from qiskit_aer import AerSimulator

class GroverSearchAlgorithm():
    def __init__(self, nr_possible_actions: int) -> None:
        self._nr_qubits = math.ceil(math.log2(nr_possible_actions))
 
        self._simulator = AerSimulator()
        self._simulator.set_option("shots", 1)

    # Code adapted from https://learning.quantum.ibm.com/tutorial/grovers-algorithm#step-1-map-classical-inputs-to-a-quantum-problem
    def _get_grover_oracle(self, good_state: str):
        """Build a Grover oracle for single good state

        Here we assume all input marked states have the same number of bits

        Parameters:
            good_state (str): Marked stat
            es of oracle

        Returns:
            QuantumCircuit: Quantum circuit representing Grover oracle
        """
        qc = QuantumCircuit(self._nr_qubits)
        
        # Flip target bit-string to match Qiskit bit-ordering
        good_state_reverted = good_state[::-1]
        # Find the indices of all the '0' elements in bit-string
        zero_inds = [ind for ind in range(self._nr_qubits) if good_state_reverted.startswith("0", ind)]
        # Add a multi-controlled Z-gate with pre- and post-applied X-gates (open-controls)
        # where the target bit-string has a '0' entry
        if (len(zero_inds) > 0):
            qc.x(zero_inds)
            
        qc.compose(MCMT(ZGate(), self._nr_qubits - 1, 1), inplace=True)

        if (len(zero_inds) > 0):
            qc.x(zero_inds)

        return qc.to_gate(label=" Oracle ")
    
    def _get_state_preparation_layer(self) -> QuantumCircuit:
        qc = QuantumCircuit(self._nr_qubits)

        for i in range(0, self._nr_qubits):
            qc.h(i)

        return qc.to_gate(label=" State Preparation ")
    
    def _get_amplification_layer(self) -> QuantumCircuit:
        qc = QuantumCircuit(self._nr_qubits)
        last_qubit_idx = self._nr_qubits - 1

        qc.h([i for i in range(0, self._nr_qubits)])
        qc.x([i for i in range(0, self._nr_qubits)])

        qc.h(last_qubit_idx)
        qc.compose(MCMT(XGate(), last_qubit_idx, 1), inplace=True)
        qc.h(last_qubit_idx)

        qc.x([i for i in range(0, self._nr_qubits)])
        qc.h([i for i in range(0, self._nr_qubits)])

        return qc.to_gate(label=" Amplification Layer ")
    
    def _transpile_circuit(self):
        self._grover_circuit_transpiled = transpile(self._grover_circuit, self._simulator)

        return self

    def get_optimal_iterations(self):
        return int(round(((np.pi / 4) * np.sqrt(2 ** self._nr_qubits))- 0.5))
    
    def build(self, iterations: int, good_action: int | None):
        self._grover_circuit = QuantumCircuit(self._nr_qubits)
        self._grover_circuit = self._grover_circuit.compose(self._get_state_preparation_layer())
        
        # Return the circuit only with state prep layer
        if good_action is None:
            self._transpile_circuit()
            return self
        
        good_action_bin = format(good_action, f'0{self._nr_qubits}b')

        for _ in range(0, iterations):
            self._grover_circuit = self._grover_circuit.compose(self._get_grover_oracle(good_action_bin))
            self._grover_circuit = self._grover_circuit.compose(self._get_amplification_layer())

        self._grover_circuit.measure_all()
        self._transpile_circuit()
        
        return self

    # Returns the binary string
    def run(self) -> str:
        result = self._simulator.run(self._grover_circuit_transpiled).result()

        return next(iter(result.get_counts()))
