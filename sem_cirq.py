import cirq
import numpy as np
from fractions import Fraction
from math import floor, gcd, log
from random import randint
from matplotlib import axes
import matplotlib.pyplot as plt

N = 15
n = floor(log(N - 1, 2)) + 1

num_target = n
num_control = 2 * num_target

def a2kmodN(a, k):
    """Compute a^{2^k} (mod N) by repeated squaring"""
    for _ in range(k):
        a = int(np.mod(a**2, N))
    return a

def mod_mult_gate(b):
    """Modular multiplication gate using controlled unitary."""
    if gcd(b, N) > 1:
        print(f"Error: gcd({b},{N}) > 1")
    else:
        n = floor(log(N - 1, 2)) + 1
        U = np.full((2**n, 2**n), 0)
        for x in range(N):
            U[b * x % N][x] = 1
        for x in range(N, 2**n):
            U[x][x] = 1
        return U

def qua_order_subroutine(a):
    print("Running qua_order_subroutine with a:", a, "; num_target:", num_target, ", num_control:", num_control)

    # List of M_b operators in order
    k_list = range(num_control)
    b_list = [a2kmodN(a, k) for k in k_list]

    # Initialize the circuit
    qubits = [cirq.LineQubit(i) for i in range(num_control + num_target)]
    
    circuit = cirq.Circuit()

    # Initialize the control register to |1>
    circuit.append(cirq.X(qubits[-1]))

    # Add Hadamard gates and modular multiplication gates
    for k, qubit in enumerate(qubits[:num_control]):
        circuit.append(cirq.H(qubit))  # Apply Hadamard gate
        if b_list[k] == 1:
            break
            # Create modular multiplication using controlled gates (this would need implementation)
            # Example: controlled gates and U here need custom implementation

    # Apply inverse QFT (not directly available in Cirq, but you can manually implement it)
    # For simplicity, we'll omit QFT here, but you can implement QFT manually
    circuit.append(cirq.qft(*qubits[:num_control]))
    # Measure the control qubits
    circuit.append(cirq.measure(*qubits[:num_control]))

    # Simulate using Cirq's simulator
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=1000)
    return result.measurements['q(0),q(1),q(2),q(3),q(4),q(5),q(6),q(7)']
    # return result

FACTOR_FOUND = False
print("Initial num_target:", num_target, ", num_control:", num_control)

while not FACTOR_FOUND:
    a = randint(3, N - 1)
    d = gcd(a, N)

    print("Loop: got a:", a, "; d:", d)

    if d != 1:
        print(f"*** Non-trivial factor found: {d} ***")
        break
    else:
        num_attempt = 0

        counts_keep = qua_order_subroutine(a)
        # print(counts_keep)
        print(np.unique(counts_keep, return_counts=True, axis=0))
        # cirq.plot_state_histogram(data=counts_keep)
        funy = list(map(lambda bits: "".join(str(bits)), counts_keep))
        # print(funy)
        plt.hist(funy, color='skyblue', edgecolor='black', bins=2**8)
        plt.show()

        while not FACTOR_FOUND and num_attempt < len(counts_keep):
            bitstring = "".join(map(lambda x: str(x), counts_keep[num_attempt]))
            num_attempt += 1
            # Find the phase from measurement
            decimal = int(bitstring, 2)
            
            phase = decimal / (2 ** (2 * n))  # phase = k / r

            # Guess the order from phase
            frac = Fraction(phase).limit_denominator(N)
            r = frac.denominator  # order = r
            print("Loop - decimal:", decimal, "; phase:", phase, "; frac:", frac)

            if phase != 0 and r % 2 == 0:
                # Guesses for factors are gcd(a^{r / 2} ± 1, N)
                x = pow(a, r // 2, N) - 1
                if x == 0:
                    continue
                d = gcd(x, N)
                print("factor guesses - x:", x, "; d:", d)
                if d > 1:
                    FACTOR_FOUND = True
                    print(f"*** Non-trivial factor found: {d} ***")
