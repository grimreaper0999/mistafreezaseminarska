import cirq
import numpy as np
from fractions import Fraction
from math import gcd, floor, log
from random import randint
import matplotlib.pyplot as plt

# Problem setting
N = 15
n = floor(log(N - 1, 2)) + 1
num_target = n
num_control = 2 * num_target

# -------------------------------
# 1) a^(2^k) mod N
# -------------------------------
def a2kmodN(a, k):
    for _ in range(k):
        a = pow(a, 2, N)
    return a

# -------------------------------
# 2) Unitarno modularno množenje
# -------------------------------
def mod_mult_gate(b):
    """Return a Cirq gate that multiplies |x> -> |b*x mod N>."""
    dim = 2**n
    U = np.zeros((dim, dim), dtype=complex)

    for x in range(N):
        U[b * x % N][x] = 1.0

    # For basis states >= N → identity
    for x in range(N, dim):
        U[x][x] = 1.0

    return cirq.MatrixGate(U)

# -------------------------------
# 3) QFT−1 implementacija
# -------------------------------
def inverse_qft(qubits):
    circuit = cirq.Circuit()
    Nq = len(qubits)

    for i in range(Nq // 2):
        circuit.append(cirq.SWAP(qubits[i], qubits[Nq - i - 1]))

    for i in range(Nq):
        circuit.append(cirq.H(qubits[i]))
        for j in range(i + 1, Nq):
            angle = -np.pi / (2 ** (j - i))
            circuit.append(cirq.CZ(qubits[j], qubits[i]) ** (angle / np.pi))

    return circuit

# -------------------------------
# 4) Glavna podrutina Shorja
# -------------------------------
def quantum_order_subroutine(a):
    print(f"Running order-finding for a={a}")

    # Qubits
    control = cirq.LineQubit.range(num_control)
    target = cirq.LineQubit.range(num_control, num_control + num_target)
    
    circuit = cirq.Circuit()

    # Prepare |1> in target
    circuit.append(cirq.X(target[0]))

    # Hadamardi
    for q in control:
        circuit.append(cirq.H(q))

    # Controlled modular multiplication
    b_list = [a2kmodN(a, k) for k in range(num_control)]

    for k, q in enumerate(control):
        b = b_list[k]
        if b == 1:
            continue
        gate = mod_mult_gate(b).on(*target).controlled_by(q)
        circuit.append(gate)

    # Inverse QFT
    circuit += inverse_qft(control)

    # Measurement
    circuit.append(cirq.measure(*control, key='result'))

    # Simulation
    sim = cirq.Simulator()
    result = sim.run(circuit, repetitions=1000)
    counts = result.histogram(key='result')

    return counts

# -------------------------------
# 5) Glavna zanka za faktorje
# -------------------------------
FACTOR_FOUND = False

while not FACTOR_FOUND:
    a = randint(2, N - 1)
    d = gcd(a, N)

    if d != 1:
        print(f"Found classical factor: {d}")
        break

    counts = quantum_order_subroutine(a)

    # pretvori ključe v bitstring obliko
    keys = list(counts.keys())
    values = list(counts.values())

    # izbere najpogostejši izid
    best = keys[np.argmax(values)]

    phase = best / (2**num_control)
    frac = Fraction(phase).limit_denominator(N)
    r = frac.denominator

    print(f"Measured: {best}, phase={phase}, fraction={frac}, r={r}")

    if r % 2 == 0:
        x = pow(a, r // 2, N)
        d = gcd(x - 1, N)
        if 1 < d < N:
            print(f"Nontrivial factor: {d}")
            FACTOR_FOUND = True
