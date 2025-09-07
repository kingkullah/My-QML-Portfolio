# 1. Import the necessary tools from modern Qiskit
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator  # This is the corrected import path
from qiskit.compiler import transpile

# 2. Create a 3-qubit quantum circuit with 3 classical bits for measurement
qc = QuantumCircuit(3, 3)

# 3. Create the entangled GHZ State
qc.h(0)      # Hadamard gate on the first qubit
qc.cx(0, 1)  # CNOT controlled by qubit 0, targeting qubit 1
qc.cx(1, 2)  # CNOT controlled by qubit 1, targeting qubit 2

# 4. Measure all three qubits
qc.measure([0, 1, 2], [0, 1, 2])

# 5. Choose the modern simulator
simulator = AerSimulator()

# 6. Prepare and run the circuit 1024 times
compiled_circuit = transpile(qc, simulator)
job = simulator.run(compiled_circuit, shots=1024)

# 7. Get the results and print the counts
result = job.result()
counts = result.get_counts(compiled_circuit)

print("GHZ state simulation results:")
print(counts)

# 8. Print the correct circuit diagram
print("\nCorrect GHZ Circuit Diagram:")
print(qc.draw(output='text'))