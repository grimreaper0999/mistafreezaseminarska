import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from fractions import Fraction
from math import floor, gcd, log


from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT, UnitaryGate
from qiskit.transpiler import CouplingMap, generate_preset_pass_manager
from qiskit.visualization import plot_histogram

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeAuckland

def M2mod15():
    """
    M2 (mod 15)
    """
    b = 2
    U = QuantumCircuit(4)

    U.swap(2, 3)
    U.swap(1, 2)
    U.swap(0, 1)

    U = U.to_gate()
    U.name = f"M_{b}"

    return U

# Get the M2 operator
M2 = M2mod15()

# Add it to a circuit and plot
circ = QuantumCircuit(4)
circ.compose(M2, inplace=True)
if input("Draw circuit M2mod15? ") in ["Y", "y", "yes"]:
    circ.decompose(reps=int(input("Number of decompositions: "))).draw(output="mpl", fold=-1)
    plt.show()
a = input("Enter to continue. ")

def controlled_M2mod15():
    """
    Controlled M2 (mod 15)
    """
    b = 2
    U = QuantumCircuit(4)

    U.swap(2, 3)
    U.swap(1, 2)
    U.swap(0, 1)

    U = U.to_gate()
    U.name = f"M_{b}"
    c_U = U.control()

    return c_U

# Get the controlled-M2 operator
controlled_M2 = controlled_M2mod15()

# Add it to a circuit and plot
circ = QuantumCircuit(5)
circ.compose(controlled_M2, inplace=True)
if input("Draw circuit controlled M2mod15? ") in ["Y", "y", "yes"]:
    circ.decompose(reps=int(input("Number of decompositions: "))).draw(output="mpl", fold=-1)
    plt.show()
a = input("Enter to continue. ")

def a2kmodN(a, k, N):
    """Compute a^{2^k} (mod N) by repeated squaring"""
    for _ in range(k):
        a = int(np.mod(a**2, N))
    return a

k_list = range(8)
b_list = [a2kmodN(2, k, 15) for k in k_list]

print(b_list)
a = input("Enter to continue. ")

def M4mod15():
    """
    M4 (mod 15)
    """
    b = 4
    U = QuantumCircuit(4)

    U.swap(1, 3)
    U.swap(0, 2)

    U = U.to_gate()
    U.name = f"M_{b}"

    return U

# Get the M4 operator
M4 = M4mod15()

# Add it to a circuit and plot
circ = QuantumCircuit(4)
circ.compose(M4, inplace=True)
if input(f"Draw circuit M4mod15? ") in ["Y", "y", "yes"]:
    circ.decompose(reps=int(input("Number of decompositions: "))).draw(output="mpl", fold=-1)
    plt.show()
a = input("Enter to continue. ")

def controlled_M4mod15():
    """
    Controlled M4 (mod 15)
    """
    b = 4
    U = QuantumCircuit(4)

    U.swap(1, 3)
    U.swap(0, 2)

    U = U.to_gate()
    U.name = f"M_{b}"
    c_U = U.control()

    return c_U

# Get the controlled-M4 operator
controlled_M4 = controlled_M4mod15()

# Add it to a circuit and plot
circ = QuantumCircuit(5)
circ.compose(controlled_M4, inplace=True)
if input(f"Draw circuit controlled M4mod15? ") in ["Y", "y", "yes"]:
    circ.decompose(reps=int(input("Number of decompositions: "))).draw(output="mpl", fold=-1)
    plt.show()
a = input("Enter to continue. ")

def mod_mult_gate(b, N):
    """
    Modular multiplication gate from permutation matrix.
    """
    if gcd(b, N) > 1:
        print(f"Error: gcd({b},{N}) > 1")
    else:
        n = floor(log(N - 1, 2)) + 1
        U = np.full((2**n, 2**n), 0)

        for x in range(N):
            U[b * x % N][x] = 1
        for x in range(N, 2**n):
            U[x][x] = 1

        G = UnitaryGate(U)
        G.name = f"M_{b}"
        return G

# Let's build M2 using the permutation matrix definition
M2_other = mod_mult_gate(2, 15)

# Add it to a circuit
circ = QuantumCircuit(4)
circ.compose(M2_other, inplace=True)
circ = circ.decompose()

# Transpile the circuit and get the depth
coupling_map = CouplingMap.from_line(4)
pm = generate_preset_pass_manager(coupling_map=coupling_map)

a = input("Enter to run the mod_mult_gate(2, 15) circuit. ")
transpiled_circ = pm.run(circ)

print(f"qubits: {circ.num_qubits}")
print(f"2q-depth: {transpiled_circ.depth(lambda x: x.operation.num_qubits==2)}")
print(f"2q-size: {transpiled_circ.size(lambda x: x.operation.num_qubits==2)}")
print(f"Operator counts: {transpiled_circ.count_ops()}")

if input(f"Draw transpiled circuit mod_mult_gate(2, 15)? ") in ["Y", "y", "yes"]:
    transpiled_circ.decompose().draw(output="mpl", fold=-1, style="clifford", idle_wires=False)
    plt.show()
a = input("Enter to continue. ")

# Order finding problem for N = 15 with a = 2
N = 15
a = 2

# Number of qubits
num_target = floor(log(N - 1, 2)) + 1  # for modular exponentiation operators
num_control = 2 * num_target  # for enough precision of estimation

# List of M_b operators in order
k_list = range(num_control)
b_list = [a2kmodN(2, k, 15) for k in k_list]

# Initialize the circuit
control = QuantumRegister(num_control, name="C")
target = QuantumRegister(num_target, name="T")
output = ClassicalRegister(num_control, name="out")
circuit = QuantumCircuit(control, target, output)

# Initialize the target register to the state |1>
circuit.x(num_control)

# Add the Hadamard gates and controlled versions of the
# multiplication gates
for k, qubit in enumerate(control):
    circuit.h(k)
    b = b_list[k]
    if b == 2:
        circuit.compose(
            M2mod15().control(), qubits=[qubit] + list(target), inplace=True
        )
    elif b == 4:
        circuit.compose(
            M4mod15().control(), qubits=[qubit] + list(target), inplace=True
        )
    else:
        continue  # M1 is the identity operator

# Apply the inverse QFT to the control register
circuit.compose(QFT(num_control, inverse=True), qubits=control, inplace=True)

# Measure the control register
circuit.measure(control, output)

if input(f"Draw full order 2 of 15 circuit? ") in ["Y", "y", "yes"]:
    circuit.draw("mpl", fold=-1)
    plt.show()
a = input("Enter to continue. ")


#service = QiskitRuntimeService()
backend = FakeAuckland() #service.backend("ibm_marrakesh")
pm = generate_preset_pass_manager(optimization_level=2, backend=backend)

a = input("Press enter to run the circuit on the FakeAuckland backend.")
transpiled_circuit = pm.run(circuit)

print(f"2q-depth: {transpiled_circuit.depth(lambda x: x.operation.num_qubits==2)}")
print(f"2q-size: {transpiled_circuit.size(lambda x: x.operation.num_qubits==2)}")
print(f"Operator counts: {transpiled_circuit.count_ops()}")

if input(f"Draw transpiled full order 2 of 15 circuit? ") in ["Y", "y", "yes"]:
    transpiled_circuit.draw(output="mpl", fold=-1, style="clifford", idle_wires=False)
    plt.show()
a = input("Enter to continue. ")

""" # Rows to be displayed in table
rows = []
# Corresponding phase of each bitstring
measured_phases = []

for output in counts:
    decimal = int(output, 2)  # Convert bitstring to decimal
    phase = decimal / (2**num_control)  # Find corresponding eigenvalue
    measured_phases.append(phase)
    # Add these values to the rows in our table:
    rows.append(
        [
            f"{output}(bin) = {decimal:>3}(dec)",
            f"{decimal}/{2 ** num_control} = {phase:.2f}",
        ]
    )

# Print the rows in a table
headers = ["Register Output", "Phase"]
df = pd.DataFrame(rows, columns=headers)
print(df)

# Rows to be displayed in a table
rows = []

for phase in measured_phases:
    frac = Fraction(phase).limit_denominator(15)
    rows.append(
        [phase, f"{frac.numerator}/{frac.denominator}", frac.denominator]
    )

# Print the rows in a table
headers = ["Phase", "Fraction", "Guess for r"]
df = pd.DataFrame(rows, columns=headers)
print(df) """

# Sampler primitive to obtain the probability distribution
sampler = Sampler(backend)
sampler.MAX_QUBITS_MEMORY = 27

# Turn on dynamical decoupling with sequence XpXm
sampler.options.dynamical_decoupling.enable = True
sampler.options.dynamical_decoupling.sequence_type = "XpXm"
# Enable gate twirling
sampler.options.twirling.enable_gates = True

pub = transpiled_circuit
job = sampler.run([pub], shots=1024)

result = job.result()[0]
counts = result.data["out"].get_counts()

plot_histogram(counts, figsize=(35, 5))

# Dictionary of bitstrings and their counts to keep
counts_keep = {}
# Threshold to filter
threshold = np.max(list(counts.values())) / 2

for key, value in counts.items():
    if value > threshold:
        counts_keep[key] = value

print(counts_keep)

a = 2
N = 15

FACTOR_FOUND = False
num_attempt = 0

while not FACTOR_FOUND:
    print(f"\nATTEMPT {num_attempt}:")
    # Here, we get the bitstring by iterating over outcomes
    # of a previous hardware run with multiple shots.
    # Instead, we can also perform a single-shot measurement
    # here in the loop.
    bitstring = list(counts_keep.keys())[num_attempt]
    num_attempt += 1
    # Find the phase from measurement
    decimal = int(bitstring, 2)
    phase = decimal / (2**num_control)  # phase = k / r
    print(f"Phase: theta = {phase}")

    # Guess the order from phase
    frac = Fraction(phase).limit_denominator(N)
    r = frac.denominator  # order = r
    print(f"Order of {a} modulo {N} estimated as: r = {r}")

    if phase != 0:
        # Guesses for factors are gcd(a^{r / 2} ± 1, 15)
        if r % 2 == 0:
            x = pow(a, r // 2, N) - 1
            d = gcd(x, N)
            if d > 1:
                FACTOR_FOUND = True
                print(f"*** Non-trivial factor found: {x} ***")