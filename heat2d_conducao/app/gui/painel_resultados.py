"""
painel_resultados.py
---------------------
Painel direito: CTkTabview com uma aba por tipo de visualização, cada uma
com um FigureCanvasTkAgg (matplotlib embutido no Tkinter).

Abas:
    Malha         -- pré-visualização da malha + condições de contorno
    Temperatura   -- contorno preenchido do campo T
    Fluxo de Calor-- |q| (mapa de cores) + vetores (Lei de Fourier)
    Superfície 3D -- T(x,y) em 3D
    Convergência  -- resíduo x iteração (atualizado AO VIVO durante a solução)
    Animação      -- reprodução do transiente com FPS definido pelo usuário
"""

import customtkinter as ctk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from app.core import visualize as vz
from app.core import solver as sv


class AbaCanvas(ctk.CTkFrame):
    """Aba genérica contendo um único FigureCanvasTkAgg."""

    def __init__(self, master, figsize=(6.5, 5.2), **kwargs):
        super().__init__(master, **kwargs)
        self.fig = plt.figure(figsize=figsize)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._msg_vazio()

    def _msg_vazio(self, texto="Ainda não há resultados nesta aba."):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, texto, ha="center", va="center", color="0.55", fontsize=12,
                transform=ax.transAxes, wrap=True)
        ax.axis("off")
        self.canvas.draw()

    def mostrar_figura(self, fig_nova):
        """Substitui a figura interna pela figura recém-gerada (de visualize.py)."""
        plt.close(self.fig)
        self.fig = fig_nova
        self.canvas.figure = self.fig
        self.canvas.get_tk_widget().pack_forget()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()


class AbaConvergencia(AbaCanvas):
    """Aba especial: desenha o resíduo em tempo real, ponto a ponto, durante a solução."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.reiniciar()

    def reiniciar(self, metodo=""):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_yscale("log")
        self.ax.set_xlabel("Iteração")
        self.ax.set_ylabel(r"Resíduo relativo $\|AT-b\|/\|b\|$")
        self.ax.set_title(f"Convergência{' -- ' + metodo if metodo else ''}")
        self.ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        (self._linha,) = self.ax.plot([], [], color="#1f77b4", lw=1.6)
        self._its, self._res = [], []
        self.fig.tight_layout()
        self.canvas.draw()

    def adicionar_ponto(self, it, res):
        self._its.append(it)
        self._res.append(max(res, 1e-300))  # evita log(0)
        self._linha.set_data(self._its, self._res)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()


class AbaAnimacao(AbaCanvas):
    """Aba especial: reproduz os quadros do transiente no FPS escolhido pelo usuário."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.frames = None
        self.tempos = None
        self.fps = 10
        self._playing = False
        self._idx = 0
        self._vmin = self._vmax = None

        self.barra = ctk.CTkFrame(self, fg_color="transparent", height=1)
        self.barra.pack(fill="x", pady=(4, 0))
        self.btn_play = ctk.CTkButton(self.barra, text="▶ Reproduzir", width=110,
                                       command=self.alternar_play)
        self.btn_play.pack(side="left", padx=6)
        self.slider = ctk.CTkSlider(self.barra, from_=0, to=1, number_of_steps=1,
                                     command=self._on_slider)
        self.slider.pack(side="left", fill="x", expand=True, padx=6)
        self.lbl_tempo = ctk.CTkLabel(self.barra, text="t = -- s", width=90)
        self.lbl_tempo.pack(side="left", padx=6)

    def carregar(self, frames, tempos, fps, mesh, cmap="turbo"):
        self.frames = frames
        self.tempos = tempos
        self.fps = max(1, fps)
        self.mesh = mesh
        self.cmap = cmap
        self._vmin = min(f.min() for f in frames)
        self._vmax = max(f.max() for f in frames)
        self.slider.configure(from_=0, to=len(frames) - 1, number_of_steps=max(1, len(frames) - 1))
        self.slider.set(0)
        self._idx = 0
        self._desenhar_frame(0)

    def _desenhar_frame(self, idx):
        if not self.frames:
            return
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        cf = ax.contourf(self.mesh.X, self.mesh.Y, self.frames[idx], levels=40,
                          cmap=self.cmap, vmin=self._vmin, vmax=self._vmax)
        self.fig.colorbar(cf, ax=ax, label="Temperatura [°C]")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")
        ax.set_title(f"t = {self.tempos[idx]:.3g} s")
        self.fig.tight_layout()
        self.canvas.draw()
        self.lbl_tempo.configure(text=f"t = {self.tempos[idx]:.3g} s")

    def _on_slider(self, valor):
        self._idx = int(round(valor))
        self._desenhar_frame(self._idx)

    def alternar_play(self):
        self._playing = not self._playing
        self.btn_play.configure(text="⏸ Pausar" if self._playing else "▶ Reproduzir")
        if self._playing:
            self._tick()

    def _tick(self):
        if not self._playing or not self.frames:
            return
        self._idx = (self._idx + 1) % len(self.frames)
        self.slider.set(self._idx)
        self._desenhar_frame(self._idx)
        self.after(int(1000 / self.fps), self._tick)


class PainelResultados(ctk.CTkTabview):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.add("Malha")
        self.add("Temperatura")
        self.add("Fluxo de Calor")
        self.add("Superfície 3D")
        self.add("Convergência")
        self.add("Animação")

        self.aba_malha = AbaCanvas(self.tab("Malha"))
        self.aba_malha.pack(fill="both", expand=True)

        self.aba_temp = AbaCanvas(self.tab("Temperatura"))
        self.aba_temp.pack(fill="both", expand=True)

        self.aba_fluxo = AbaCanvas(self.tab("Fluxo de Calor"))
        self.aba_fluxo.pack(fill="both", expand=True)

        self.aba_3d = AbaCanvas(self.tab("Superfície 3D"))
        self.aba_3d.pack(fill="both", expand=True)

        self.aba_conv = AbaConvergencia(self.tab("Convergência"))
        self.aba_conv.pack(fill="both", expand=True)

        self.aba_anim = AbaAnimacao(self.tab("Animação"))
        self.aba_anim.pack(fill="both", expand=True)

    # -----------------------------------------------------------------
    def mostrar_malha(self, mesh, contornos):
        fig = vz.plot_malha(mesh, contornos)
        self.aba_malha.mostrar_figura(fig)
        self.set("Malha")

    def mostrar_campo_final(self, mesh, T_campo, k, cmap="turbo"):
        self.aba_temp.mostrar_figura(vz.plot_temperatura(mesh, T_campo, cmap=cmap))
        self.aba_fluxo.mostrar_figura(vz.plot_fluxo_calor(mesh, T_campo, k))
        self.aba_3d.mostrar_figura(vz.plot_superficie3d(mesh, T_campo, cmap=cmap))

    def preparar_convergencia(self, metodo):
        self.aba_conv.reiniciar(metodo)
        self.set("Convergência")

    def ponto_convergencia(self, it, res):
        self.aba_conv.adicionar_ponto(it, res)

    def carregar_animacao(self, frames, tempos, fps, mesh, cmap="turbo"):
        self.aba_anim.carregar(frames, tempos, fps, mesh, cmap)
