"""
Animacion: estado de Bob en la esfera de Bloch antes y despues de las correcciones.
Fase 1: aparecen los 4 estados posibles de Bob (uno por resultado de Alice) -> su promedio es el centro (I/2).
Fase 2: cada rama recibe su correccion Pauli y converge al estado original.
Salida: ../video/bloch_teleportacion.mp4 y ../fig/bloch_ramas.pdf
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
plt.rcParams.update({"font.family": "serif"})
th, ph = 1.05, 0.9                                  # estado de ejemplo
r = np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])
# X^b Z^a |psi> en Bloch:  Z -> (-x,-y,z) ; X -> (x,-y,-z)
def Zb(v): return v*np.array([-1, -1, 1])
def Xb(v): return v*np.array([1, -1, -1])
branches = {"00": r, "01": Xb(r), "10": Zb(r), "11": Xb(Zb(r))}   # clave = m1 m2 ; Bob tiene X^{m2} Z^{m1}|psi>
corr = {"00": "I", "01": "X", "10": "Z", "11": "ZX"}
cols = {"00": "#1f4e79", "01": "#c0504d", "10": "#2e7d32", "11": "#6a1b9a"}

def sphere(ax):
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    ax.plot_wireframe(np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v), color="#bbbbbb", lw=0.3)
    for a, lab in zip(np.eye(3), ["x", "y", r"$|0\rangle$"]):
        ax.plot(*np.c_[-a, a], color="#999999", lw=0.6); ax.text(*(1.15*a), lab, fontsize=10)
    ax.text(0, 0, -1.25, r"$|1\rangle$", fontsize=10)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1); ax.set_box_aspect((1, 1, 1), zoom=1.35); ax.axis("off")

def slerp(a, b, t):
    om = np.arccos(np.clip(a@b, -1, 1))
    if om < 1e-6: return a
    if abs(om-np.pi) < 1e-6:                          # antipodas: girar por un eje perpendicular
        perp = np.cross(a, [0.3, 0.7, 0.2]); perp /= np.linalg.norm(perp)
        return np.cos(np.pi*t)*a + np.sin(np.pi*t)*perp
    return (np.sin((1-t)*om)*a + np.sin(t*om)*b)/np.sin(om)

# ---- figura estatica
fig = plt.figure(figsize=(6, 5)); ax = fig.add_subplot(projection="3d"); sphere(ax)
ax.quiver(0, 0, 0, *r, color="k", lw=2.5, arrow_length_ratio=0.08)
ax.text(*(1.1*r), r"$|\psi\rangle$", fontsize=12)
for k, v in branches.items():
    if k == "00": continue
    ax.quiver(0, 0, 0, *v, color=cols[k], lw=2, arrow_length_ratio=0.08)
    ax.text(*(0.75*v), f"$m_1m_2={k}$", color=cols[k], fontsize=9)
ax.scatter(0, 0, 0, color="orange", s=40); ax.view_init(18, 35)
ax.set_title("Estados de Bob antes de la corrección\n(promedio = centro: $\\rho_B=I/2$)", fontsize=11)
fig.savefig(ROOT/"fig"/"bloch_ramas.pdf", bbox_inches="tight", pad_inches=0.3); plt.close(fig)

# ---- animacion
fps, T1, T2, T3 = 30, 4.0, 3.0, 5.0
N1, N2, N3 = int(T1*fps), int(T2*fps), int(T3*fps)
fig = plt.figure(figsize=(12.8, 7.2), dpi=100); ax = fig.add_axes([0.05, 0.0, 0.9, 0.88], projection="3d")
def frame(i):
    ax.cla(); sphere(ax); ax.view_init(18, 35 + 0.15*i)
    ax.quiver(0, 0, 0, *r, color="k", lw=1.2, alpha=0.35, arrow_length_ratio=0.08)
    if i < N1:
        t = i/N1; msg = "Alice mide en la base de Bell: Bob queda en uno de 4 estados (cada uno con prob. 1/4)"
        for k, v in branches.items():
            ax.quiver(0, 0, 0, *(min(1, 1.6*t)*v), color=cols[k], lw=2.2, arrow_length_ratio=0.08)
    elif i < N1+N2:
        msg = "Sin los 2 bits clásicos, Bob solo ve la mezcla: promedio = centro de la esfera (I/2)"
        for k, v in branches.items(): ax.quiver(0, 0, 0, *v, color=cols[k], lw=1.5, alpha=0.5, arrow_length_ratio=0.08)
        ax.scatter(0, 0, 0, color="orange", s=120)
    else:
        t = min(1, (i-N1-N2)/(0.8*N3)); msg = "Llegan los bits $(m_1,m_2)$: Bob aplica $Z^{m_1}X^{m_2}$ y todas las ramas convergen a $|\\psi\\rangle$"
        for k, v in branches.items():
            w = slerp(v, r, t); ax.quiver(0, 0, 0, *w, color=cols[k], lw=2.2, arrow_length_ratio=0.08)
            ax.text(*(1.1*w), corr[k], color=cols[k], fontsize=11)
    fig.suptitle(msg, fontsize=14, y=0.95)
anim = FuncAnimation(fig, frame, frames=N1+N2+N3)
anim.save(ROOT/"video"/"bloch_teleportacion.mp4", writer=FFMpegWriter(fps=fps, codec="libx264", extra_args=["-preset", "veryfast", "-pix_fmt", "yuv420p"]))
print("ok")
