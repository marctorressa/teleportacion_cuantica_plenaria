"""
Teleportacion cuantica en Qiskit (circuito dinamico con feed-forward clasico).
Plenaria Mecanica Cuantica 2026-II  -  M. A. Torres, J. K. Ruiz Cortes
Uso:  python teleport_qiskit.py            (demo en vivo, ~5 s)
Requiere: pip install qiskit qiskit-aer
Convencion: q0 = estado a teleportar (Alice), q1 = mitad de Alice del par, q2 = Bob.
           m1 = resultado de q0 (tras H) -> corrige Z ;  m2 = resultado de q1 -> corrige X.
           Ojo: Qiskit imprime los bits como 'c1c0' (orden little-endian).
"""
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

def teleport_circuit(theta, phi, lam, verify=True):
    q = QuantumRegister(3, "q"); m = ClassicalRegister(2, "m"); out = ClassicalRegister(1, "bob")
    qc = QuantumCircuit(q, m, out)
    qc.u(theta, phi, lam, q[0])                     # 1. Alice prepara |psi> = U|0>
    qc.barrier()
    qc.h(q[1]); qc.cx(q[1], q[2])                   # 2. Par de Bell |Phi+> entre Alice (q1) y Bob (q2)
    qc.barrier()
    qc.cx(q[0], q[1]); qc.h(q[0])                   # 3. Medicion en base de Bell = CNOT + H + medida en Z
    qc.measure(q[0], m[0]); qc.measure(q[1], m[1])
    qc.barrier()
    with qc.if_test((m[1], 1)): qc.x(q[2])          # 4. Feed-forward: 2 bits clasicos -> correcciones
    with qc.if_test((m[0], 1)): qc.z(q[2])
    if verify:                                      # 5. Verificacion: deshacer U; si todo va bien, Bob mide 0 siempre
        qc.u(-theta, -lam, -phi, q[2])              # U(theta,phi,lam)^dagger = U(-theta,-lam,-phi)
        qc.measure(q[2], out[0])
    return qc

def p_success(qc, backend, shots=4000):
    counts = backend.run(transpile(qc, backend), shots=shots).result().get_counts()
    ok = sum(v for k, v in counts.items() if k.split()[0] == "0")   # registro 'bob' es el primero en la cadena
    return ok / shots, counts

if __name__ == "__main__":
    rng = np.random.default_rng(7)
    th, ph, la = rng.uniform(0, np.pi), rng.uniform(0, 2*np.pi), rng.uniform(0, 2*np.pi)
    qc = teleport_circuit(th, ph, la)
    print(qc.draw(output="text", fold=120))
    ideal = AerSimulator()
    p, counts = p_success(qc, ideal)
    print(f"\nSimulador ideal:   P(Bob recupera |psi>) = {p:.4f}")
    print("Conteos (formato 'bob m[1]m[0]'; m[0]=bit Z, m[1]=bit X): cada rama ~25%:", counts)

    # Ruido despolarizante en compuertas de 2 qubits -> la fidelidad cae
    print("\nRuido despolarizante en CX (prob. lambda):")
    for lam_ in (0.0, 0.02, 0.05, 0.1, 0.2):
        nm = NoiseModel(); nm.add_all_qubit_quantum_error(depolarizing_error(lam_, 2), ["cx"])
        p, _ = p_success(qc, AerSimulator(noise_model=nm))
        print(f"  lambda = {lam_:4.2f}  ->  P_exito = {p:.3f}")
