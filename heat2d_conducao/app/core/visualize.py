"""
visualize.py
------------
Rotinas de visualização do simulador de condução de calor 2D:

    - plot_malha              : pré-visualização da malha + condições de contorno
    - plot_temperatura        : campo de temperatura 2D (contorno preenchido)
    - plot_fluxo_calor        : magnitude do fluxo (mapa de cores) + vetores (quiver)
    - plot_superficie3d       : T(x,y) como superfície 3D
    - plot_convergencia       : histórico de resíduo (semilog) do solver iterativo
    - animar_transiente       : animação GIF do campo de T ao longo do tempo

Paleta e estilo pensados para apresentação didática (cores contrastantes,
fontes maiores, grid discreto).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (necessário para projeção 3d)

from .mesh import Mesh2D
from .bc import ContornosRetangulo
import textwrap


def _titulo_quebrado(texto, largura=42):
    """Quebra títulos longos em múltiplas linhas para não serem cortados pela colorbar."""
    return "\n".join(textwrap.wrap(texto, largura))

# --- estilo visual global (didático, cores fortes, boa legibilidade) ---
COR_BC = {
    "dirichlet": "#d62728",   # vermelho - T prescrita
    "neumann": "#1f77b4",     # azul - fluxo prescrito
    "conveccao": "#ff7f0e",   # laranja - convecção
    "simetria": "#2ca02c",    # verde - simetria
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "font.size": 11,
    "grid.color": "#dddddd",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
})


def plot_malha(mesh: Mesh2D, contornos: ContornosRetangulo, figsize=(7, 5.5)):
    """Pré-visualização da malha cartesiana com as condições de contorno indicadas por cor."""
    fig, ax = plt.subplots(figsize=figsize)

    # linhas da malha
    for xi in mesh.x:
        ax.plot([xi, xi], [0, mesh.Ly], color="#cccccc", lw=0.6, zorder=1)
    for yi in mesh.y:
        ax.plot([0, mesh.Lx], [yi, yi], color="#cccccc", lw=0.6, zorder=1)

    # nós
    ax.scatter(mesh.X, mesh.Y, s=10, color="#444444", zorder=2)

    # contorno colorido por tipo de BC (linhas grossas nas 4 faces)
    lw = 5
    ax.plot([0, 0], [0, mesh.Ly], color=COR_BC[contornos.esquerda.tipo], lw=lw,
            label=f"Esquerda: {contornos.esquerda.rotulo()}", solid_capstyle="butt")
    ax.plot([mesh.Lx, mesh.Lx], [0, mesh.Ly], color=COR_BC[contornos.direita.tipo], lw=lw,
            label=f"Direita: {contornos.direita.rotulo()}", solid_capstyle="butt")
    ax.plot([0, mesh.Lx], [0, 0], color=COR_BC[contornos.inferior.tipo], lw=lw,
            label=f"Inferior: {contornos.inferior.rotulo()}", solid_capstyle="butt")
    ax.plot([0, mesh.Lx], [mesh.Ly, mesh.Ly], color=COR_BC[contornos.superior.tipo], lw=lw,
            label=f"Superior: {contornos.superior.rotulo()}", solid_capstyle="butt")

    ax.set_xlim(-0.06 * mesh.Lx, 1.06 * mesh.Lx)
    ax.set_ylim(-0.06 * mesh.Ly, 1.06 * mesh.Ly)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(_titulo_quebrado(f"Pré-visualização da malha ({mesh.Nx}×{mesh.Ny} nós, "
                 f"dx={mesh.dx:.4g} m, dy={mesh.dy:.4g} m)"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=True, fontsize=9)
    fig.tight_layout()
    return fig


def plot_temperatura(mesh: Mesh2D, T_campo: np.ndarray, titulo="Campo de Temperatura",
                      cmap="turbo", n_niveis=40, unidade="°C", figsize=(7, 5.5)):
    """Campo de temperatura 2D como contorno preenchido (mapa de cores)."""
    fig, ax = plt.subplots(figsize=figsize)
    cf = ax.contourf(mesh.X, mesh.Y, T_campo, levels=n_niveis, cmap=cmap)
    ax.contour(mesh.X, mesh.Y, T_campo, levels=10, colors="black", linewidths=0.3, alpha=0.4)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(f"Temperatura [{unidade}]")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.set_title(_titulo_quebrado(titulo))
    fig.tight_layout()
    return fig


def plot_fluxo_calor(mesh: Mesh2D, T_campo: np.ndarray, k: float,
                      titulo="Fluxo de Calor (Lei de Fourier)", cmap="inferno",
                      passo_quiver=None, figsize=(7, 5.5)):
    """Magnitude do fluxo de calor |q| (mapa de cores) + vetores de direção (quiver)."""
    from . import solver as sv
    qx, qy, q_mag = sv.fluxo_calor(mesh, T_campo, k)

    fig, ax = plt.subplots(figsize=figsize)
    cf = ax.pcolormesh(mesh.X, mesh.Y, q_mag, cmap=cmap, shading="auto")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("|q| [W/m²]")

    # subamostragem dos vetores para não poluir a figura
    if passo_quiver is None:
        passo_quiver = max(1, min(mesh.Nx, mesh.Ny) // 15)
    sl = (slice(None, None, passo_quiver), slice(None, None, passo_quiver))
    ax.quiver(mesh.X[sl], mesh.Y[sl], qx[sl], qy[sl], color="white",
              scale_units="xy", angles="xy", width=0.003, alpha=0.9)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.set_title(_titulo_quebrado(titulo))
    fig.tight_layout()
    return fig


def plot_superficie3d(mesh: Mesh2D, T_campo: np.ndarray, titulo="Campo de Temperatura (3D)",
                       cmap="turbo", unidade="°C", figsize=(8, 6)):
    """Superfície 3D T(x,y)."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(mesh.X, mesh.Y, T_campo, cmap=cmap, linewidth=0,
                            antialiased=True, rcount=min(mesh.Ny, 120), ccount=min(mesh.Nx, 120))
    fig.colorbar(surf, ax=ax, shrink=0.6, label=f"T [{unidade}]")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel(f"T [{unidade}]")
    ax.set_title(_titulo_quebrado(titulo))
    fig.tight_layout()
    return fig


def plot_convergencia(hist, metodo="", titulo=None, figsize=(7, 4.5)):
    """Histórico de resíduo relativo (escala log) ao longo das iterações."""
    fig, ax = plt.subplots(figsize=figsize)
    hist = np.asarray(hist)
    ax.semilogy(np.arange(1, len(hist) + 1), hist, color="#1f77b4", lw=1.6)
    ax.set_xlabel("Iteração")
    ax.set_ylabel(r"Resíduo relativo  $\|\,A T - b\,\| \,/\, \|b\|$")
    ax.set_title(_titulo_quebrado(titulo or f"Convergência do método: {metodo}"))
    ax.grid(True, which="both")
    fig.tight_layout()
    return fig


def animar_transiente(mesh: Mesh2D, frames_T, tempos, framerate=10, cmap="turbo",
                       titulo="Evolução transiente de T", unidade="°C",
                       vmin=None, vmax=None, caminho_saida="transiente.gif", figsize=(7, 5.5)):
    """
    Cria uma animação (GIF) do campo de temperatura ao longo do tempo.

    frames_T : lista/array de campos (Ny,Nx), um por instante salvo
    tempos   : lista de tempos [s] correspondentes a cada frame
    framerate: quadros por segundo da animação (definido pelo usuário)
    """
    frames_T = np.asarray(frames_T)
    if vmin is None:
        vmin = frames_T.min()
    if vmax is None:
        vmax = frames_T.max()

    fig, ax = plt.subplots(figsize=figsize)
    cf = ax.contourf(mesh.X, mesh.Y, frames_T[0], levels=40, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(f"Temperatura [{unidade}]")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    txt_tempo = ax.set_title(_titulo_quebrado(f"{titulo}  |  t = {tempos[0]:.3g} s"))

    def atualizar(frame_idx):
        ax.clear()
        ax.contourf(mesh.X, mesh.Y, frames_T[frame_idx], levels=40, cmap=cmap,
                    vmin=vmin, vmax=vmax)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
        ax.set_title(_titulo_quebrado(f"{titulo}  |  t = {tempos[frame_idx]:.3g} s"))
        return []

    anim = animation.FuncAnimation(fig, atualizar, frames=len(frames_T),
                                    interval=1000.0 / framerate, blit=False)
    anim.save(caminho_saida, writer=animation.PillowWriter(fps=framerate))
    plt.close(fig)
    return caminho_saida
