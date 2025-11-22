import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from fractions import Fraction
from math import floor, gcd, log
from Random.random import randint

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT, UnitaryGate
from qiskit.transpiler import CouplingMap, generate_preset_pass_manager
from qiskit.visualization import plot_histogram

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2
from qiskit.primitives import StatevectorSampler
from qiskit_ibm_runtime.fake_provider import FakeAuckland

N = 15
n = floor(log(N - 1, 2)) + 1

def a2kmodN(a, k):
    """Compute a^{2^k} (mod N) by repeated squaring"""
    for _ in range(k):
        a = int(np.mod(a**2, N))
    return a

def mod_mult_gate(b):
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
from qiskit.primitives import StatevectorSampler


def qua_order_subroutine(a):
    # Number of qubits
    num_target = n  # for modular exponentiation operators
    num_control = 2 * num_target  # for enough precision of estimation

    # List of M_b operators in order
    k_list = range(num_control)
    b_list = [a2kmodN(a, k) for k in k_list]

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
        if b != 1:
            circuit.compose(
                mod_mult_gate(b).control(), qubits=[qubit] + list(target), inplace=True
            )
        else:
            break  # M1 is the identity operator

    # Apply the inverse QFT to the control register
    circuit.compose(QFT(num_control, inverse=True), qubits=control, inplace=True)

    # Measure the control register
    circuit.measure(control, output)

    #service = QiskitRuntimeService()
    # TODO KATJA PROSIM PROSIM PROSIM PROBI USPOSOBIT TO STVAR Z BACKENDOM
    backend = FakeAuckland() #service.backend("ibm_marrakesh")
    pm = generate_preset_pass_manager(optimization_level=2, backend=backend)

    a = input("Press enter to run the circuit on the FakeAuckland backend.")
    transpiled_circuit = pm.run(circuit)

    print(f"2q-depth: {transpiled_circuit.depth(lambda x: x.operation.num_qubits==2)}")
    print(f"2q-size: {transpiled_circuit.size(lambda x: x.operation.num_qubits==2)}")
    print(f"Operator counts: {transpiled_circuit.count_ops()}")

    # Sampler primitive to obtain the probability distribution
    #sampler = SamplerV2(backend)
    sampler = StatevectorSampler(default_shots=1)
    #sampler.MAX_QUBITS_MEMORY = 27

    # Turn on dynamical decoupling with sequence XpXm
    #sampler.options.dynamical_decoupling.enable = True
    #sampler.options.dynamical_decoupling.sequence_type = "XpXm"
    # Enable gate twirling
    #sampler.options.twirling.enable_gates = True

    pub = transpiled_circuit
    job = sampler.run([pub], shots=1)

    result = job.result()[0]
    counts = result.data["out"].get_counts()

    # Dictionary of bitstrings and their counts to keep
    counts_keep = {}
    # Threshold to filter
    threshold = np.max(list(counts.values())) / 2

    for key, value in counts.items():
        if value > threshold:
            counts_keep[key] = value

    return counts_keep

FACTOR_FOUND = False

while not FACTOR_FOUND:

    a = randint(2, N-1)

    d = gcd(a, N)

    if d != 1:
        print(f"*** Non-trivial factor found: {d} ***")
    else:
        num_attempt = 0

        while not FACTOR_FOUND and num_attempt < len(list(counts_keep.keys())):

            # Here, we get the bitstring by iterating over outcomes
            # of a previous hardware run with multiple shots.
            # Instead, we can also perform a single-shot measurement
            # here in the loop.
            bitstring = list(counts_keep.keys())[num_attempt]
            num_attempt += 1

            # Find the phase from measurement
            decimal = int(bitstring, 2)
            phase = decimal / (2**num_control)  # phase = k / r

            # Guess the order from phase
            frac = Fraction(phase).limit_denominator(N)
            r = frac.denominator  # order = r

            if phase != 0:
                # Guesses for factors are gcd(a^{r / 2} ± 1, 15)
                if r % 2 == 0:
                    x = pow(a, r // 2, N) - 1
                    d = gcd(x, N)
                    if d > 1:
                        FACTOR_FOUND = True
                        print(f"*** Non-trivial factor found: {x} ***")