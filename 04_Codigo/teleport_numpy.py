"""
Simulacion de teleportacion cuantica con matrices densidad (solo NumPy).
Plenaria Mecanica Cuantica 2026-II  -  M. A. Torres, J. K. Ruiz Cortes
Genera las figuras de la plenaria en ../fig/
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
FIG = Path(__file__).resolve().parent.parent / "fig"; FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(2026)
plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False})
I2 = np.eye(2); X = np.array([[0, 1], [1, 0]]); Z = np.diag([1, -1]); Y = 1j * X @ Z
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
k0, k1 = np.array([1, 0]), np.array([0, 1])
def kron(*a):
    out = np.array([[1.0]])
    for m in a: out = np.kron(out, m)
    return out
def ket2dm(v): return np.outer(v, v.conj())
phi_plus = (np.kron(k0, k0) + np.kron(k1, k1)) / np.sqrt(2)
P_PHI = ket2dm(phi_plus)

def werner(p):            # rho_W = p|Phi+><Phi+| + (1-p) I/4
    return p * P_PHI + (1 - p) * np.eye(4) / 4

def amp_damp(rho2q, g):   # amortiguamiento de amplitud sobre el qubit de Bob (qubit 2 del par)
    K0 = np.array([[1, 0], [0, np.sqrt(1 - g)]]); K1 = np.array([[0, np.sqrt(g)], [0, 0]])
    return sum(kron(I2, K) @ rho2q @ kron(I2, K).conj().T for K in (K0, K1))

def CNOT(c, t, n=3):
    P0, P1 = ket2dm(k0), ket2dm(k1)
    a = [I2] * n; a[c] = P0
    b = [I2] * n; b[c] = P1; b[t] = X
    return kron(*a) + kron(*b)

def teleport(psi, resource, theta=np.pi):
    """Protocolo completo. Devuelve rho de Bob tras correcciones.
    theta=pi -> correcciones Pauli estandar (X^m2 Z^m1 salvo fase global).
    theta != pi -> correcciones RX(theta), RZ(theta) (variante 'susurro cuantico')."""
    rho = np.kron(ket2dm(psi), resource)              # qubits: 0 (Alice, psi), 1 (Alice), 2 (Bob)
    U = kron(H, I2, I2) @ CNOT(0, 1)
    rho = U @ rho @ U.conj().T
    RX = np.cos(theta/2)*I2 - 1j*np.sin(theta/2)*X
    RZ = np.cos(theta/2)*I2 - 1j*np.sin(theta/2)*Z
    out = np.zeros((2, 2), complex)
    for m1 in (0, 1):
        for m2 in (0, 1):
            Pm = kron(ket2dm([k0, k1][m1]), ket2dm([k0, k1][m2]), I2)
            r = Pm @ rho @ Pm
            rB = np.einsum("ijkijl->kl", r.reshape(2, 2, 2, 2, 2, 2))   # traza parcial sobre 0,1
            C = np.linalg.matrix_power(RZ, m1) @ np.linalg.matrix_power(RX, m2)
            out += C @ rB @ C.conj().T
    return out, rho

def haar(n):
    v = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def avg_fid(resource, n=400, theta=np.pi):
    return np.mean([np.real(s.conj() @ teleport(s, resource, theta)[0] @ s) for s in haar(n)])

if __name__ == "__main__":
    # 1) Verificacion: estado ideal -> fidelidad 1 ; no-senalizacion (sin bits clasicos rho_B = I/2)
    psi = haar(1)[0]
    rho_out, rho_after = teleport(psi, P_PHI)
    print("F ideal =", np.real(psi.conj() @ rho_out @ psi))
    rB_nobits = np.einsum("ijkijl->kl", rho_after.reshape(2,2,2,2,2,2))
    print("rho_B sin informacion clasica =\n", np.round(rB_nobits, 6))

    # 2) Medir-y-preparar clasico: F = 2/3
    s = haar(20000)
    c = np.abs(s[:, 0])**2
    print("F clasica (medir-preparar en Z) =", np.mean(c**2 + (1 - c)**2))

    # 3) Fidelidad vs parametro de Werner
    ps = np.linspace(0, 1, 21)
    Fnum = [avg_fid(werner(p), 300) for p in ps]
    pp = np.linspace(0, 1, 200)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(pp, (1 + pp) / 2, color="#1f4e79", lw=2, label=r"Teoría: $\bar F=(1+p)/2$")
    ax.plot(ps, Fnum, "o", color="#c0504d", ms=5, label="Simulación (300 estados de Haar)")
    ax.axhline(2/3, ls="--", color="gray"); ax.text(0.02, 0.675, r"límite clásico $2/3$", color="gray")
    ax.axvline(1/3, ls=":", color="#2e7d32"); ax.text(0.345, 0.52, r"$p=1/3$" "\n(entrelazado)", color="#2e7d32", fontsize=9)
    ax.axvline(1/np.sqrt(2), ls=":", color="#6a1b9a"); ax.text(0.72, 0.52, r"$p=1/\sqrt{2}$" "\n(viola CHSH)", color="#6a1b9a", fontsize=9)
    ax.set_xlabel(r"Parámetro de Werner $p$"); ax.set_ylabel(r"Fidelidad promedio $\bar F$")
    ax.set_ylim(0.48, 1.02); ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fidelidad_werner.pdf"); fig.savefig(FIG / "fidelidad_werner.png", dpi=200)

    # 4) Amortiguamiento de amplitud en el qubit de Bob
    gs = np.linspace(0, 1, 21)
    Fg = [avg_fid(amp_damp(P_PHI, g), 300) for g in gs]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    gg = np.linspace(0, 1, 200)
    ax.plot(gg, 2/3 + np.sqrt(1-gg)/3 - gg/6, color="#1f4e79", lw=2, label=r"Teoría: $2/3+\sqrt{1-\gamma}/3-\gamma/6$")
    ax.plot(gs, Fg, "o", color="#c0504d", ms=4, label="Simulación")
    ax.axvline(2*(np.sqrt(2)-1), ls=":", color="#2e7d32"); ax.text(0.70, 0.9, r"$\gamma^*=2(\sqrt{2}-1)$", color="#2e7d32", fontsize=9)
    ax.axhline(2/3, ls="--", color="gray"); ax.text(0.02, 0.675, r"límite clásico $2/3$", color="gray")
    ax.set_xlabel(r"Probabilidad de decaimiento $\gamma$ (qubit de Bob)"); ax.set_ylabel(r"$\bar F$")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / "fidelidad_damping.pdf")
    g_cross = gs[np.argmax(np.array(Fg) < 2/3)]
    print("Amortiguamiento: F cae bajo 2/3 hacia gamma ~", g_cross, " F(gamma=1)=", Fg[-1])

    # 5) Variante 'susurro cuantico' (Miranda & Oshiro): correcciones RX(theta), RZ(theta)
    th = np.linspace(0, 2*np.pi, 41)
    Ft = [avg_fid(P_PHI, 250, t) for t in th]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    tt = np.linspace(0, 2*np.pi, 300); sn = np.sin(tt/2)**2
    ax.plot(tt/np.pi, (3 + 2*sn + sn**2)/6, color="#1f4e79", lw=2, label=r"Teoría: $(3+2s^2+s^4)/6,\ s=\sin(\theta/2)$")
    ax.plot(th/np.pi, Ft, "o", color="#c0504d", ms=4, label="Simulación"); ax.legend(frameon=False, fontsize=9, loc="lower center")
    ax.axhline(2/3, ls="--", color="gray")
    ax.set_xlabel(r"Ángulo de corrección $\theta/\pi$"); ax.set_ylabel(r"$\bar F$")
    ax.set_title(r"Correcciones $R_X(\theta),R_Z(\theta)$: $\theta=\pi$ recupera el protocolo estándar", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "susurro_theta.pdf")
    print("Susurro: F(0)=%.3f  F(pi/2)=%.3f  F(pi)=%.3f" % (Ft[0], Ft[10], Ft[20]))

    # 6) Perdidas en fibra: limite PLOB vs distancia
    L = np.linspace(1, 600, 400); eta = 10 ** (-0.2 * L / 10)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.semilogy(L, -np.log2(1 - eta), color="#1f4e79", lw=2, label=r"Límite PLOB $-\log_2(1-\eta)$")
    ax.semilogy(L, eta, "--", color="#c0504d", label=r"Transmitancia $\eta=10^{-0.02L}$")
    ax.set_xlabel("Distancia en fibra $L$ (km), 0.2 dB/km"); ax.set_ylabel("ebits por uso de canal")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / "plob.pdf")
    print("eta(100 km)=%.1e  eta(500 km)=%.1e" % (10**-2, 10**-10))
