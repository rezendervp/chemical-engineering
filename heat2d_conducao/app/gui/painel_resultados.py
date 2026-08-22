"""
painel_resultados.py
---------------------
Painel direito: CTkTabview com uma aba por tipo de visualização, cada uma
com um FigureCanvasTkAgg (matplotlib embutido no Tkinter).

Abas:
    Malha         -- pré-visualização da malha + condições de contorno
    Temperatura   -- contorno preenchido do campo T
    Fluxo de Calor-- |q| (mapa de cores) + vetores (Lei de Fourier)
    Superfície 3D -- T(x,y) em 3D, com toolbar de zoom/pan (rotação por
                     clique+arraste já é nativa do matplotlib)
    Convergência  -- resíduo x iteração (permanente) OU resíduo x passo de
                     tempo (transiente implícito). Atualizado AO VIVO.
    Animação      -- reprodução do transiente com FPS definido pelo usuário
    Perfis        -- extração de linha horizontal/vertical e ponto de
                     amostragem (valor x iteração/tempo)

NOTA IMPORTANTE sobre o bug de "fantasma" (ghost image) corrigido aqui:
    Reatribuir apenas `canvas.figure = nova_figura` e chamar `.draw()` NÃO
    é suficiente no backend TkAgg no Windows -- o widget Tk (um Label
    interno com uma imagem bitmap) pode manter pixels da figura anterior
    onde a nova figura não cobre exatamente a mesma área (ex.: colorbar
    de largura diferente). A correção é DESTRUIR o widget Tk do canvas
    antigo e criar um FigureCanvasTkAgg novo a cada troca de figura.
"""

import customtkinter as ctk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from app.core import visualize as vz
from app.core import solver as sv


class AbaCanvas(ctk.CTkFrame):
    """Aba genérica contendo um único FigureCanvasTkAgg, recriado a cada figura nova."""

    def __init__(self, master, figsize=(6.5, 5.2), com_toolbar=False, **kwargs):
        super().__init__(master, **kwargs)
        self.figsize = figsize
        self.com_toolbar = com_toolbar
        self.fig = plt.figure(figsize=figsize)
        self._frame_canvas = ctk.CTkFrame(self, fg_color="transparent", height=1)
        self._frame_canvas.pack(fill="both", expand=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self._frame_canvas)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._toolbar = None
        if self.com_toolbar:
            self._toolbar = NavigationToolbar2Tk(self.canvas, self._frame_canvas)
            self._toolbar.update()
        self._msg_vazio()

    def _msg_vazio(self, texto="Ainda não há resultados nesta aba."):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, texto, ha="center", va="center", color="0.55", fontsize=12,
                transform=ax.transAxes, wrap=True)
        ax.axis("off")
        self.canvas.draw()

    def mostrar_figura(self, fig_nova):
        """
        Substitui a figura interna pela figura recém-gerada, RECRIANDO o
        widget do canvas do zero -- evita o bug de "fantasma" (pixels da
        figura anterior sobrepostos) observado no Windows.
        """
        self.canvas.get_tk_widget().destroy()
        if self._toolbar is not None:
            self._toolbar.destroy()
            self._toolbar = None
        plt.close(self.fig)

        self.fig = fig_nova
        self.canvas = FigureCanvasTkAgg(self.fig, master=self._frame_canvas)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        if self.com_toolbar:
            self._toolbar = NavigationToolbar2Tk(self.canvas, self._frame_canvas)
            self._toolbar.update()
            for ax in self.fig.axes:
                if hasattr(ax, "mouse_init"):
                    ax.mouse_init()
        self.canvas.draw()
        self.update_idletasks()


class AbaConvergencia(AbaCanvas):
    """
    Aba especial: desenha o resíduo em tempo real.

    Dois modos:
        "iteracao"    -- regime permanente: x = número da iteração do solver
        "passo_tempo" -- regime transiente implícito: x = passo de tempo
                         (y = resíduo final do solve linear daquele passo)
    O esquema explícito não gera resíduo (não há solve linear), então nesse
    caso a aba mostra uma mensagem explicativa em vez de um gráfico vazio.
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self._modo = None

    def reiniciar(self, metodo="", modo="iteracao"):
        self._modo = modo
        self.canvas.get_tk_widget().destroy()
        plt.close(self.fig)
        self.fig = plt.figure(figsize=self.figsize)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self._frame_canvas)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_yscale("log")
        if modo == "iteracao":
            self.ax.set_xlabel("Iteração")
            self.ax.set_ylabel(r"Resíduo relativo $\|AT-b\|/\|b\|$")
            self.ax.set_title(f"Convergência{' -- ' + metodo if metodo else ''}")
        else:
            self.ax.set_xlabel("Passo de tempo (n)")
            self.ax.set_ylabel(r"Resíduo final do solve linear no passo")
            self.ax.set_title(f"Convergência por passo de tempo{' -- ' + metodo if metodo else ''}")
        self.ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        (self._linha,) = self.ax.plot([], [], color="#1f77b4", lw=1.6, marker=".", markersize=4)
        self._its, self._res = [], []
        self.fig.tight_layout()
        self.canvas.draw()

    def sem_residuo(self, motivo):
        """Usado quando o esquema não gera resíduo (ex.: transiente explícito)."""
        self._msg_vazio(motivo)

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
        self._after_id = None  # necessário para cancelar o loop de reprodução

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

    def limpar(self):
        """Para a reprodução e limpa os dados -- chamado ao iniciar um novo caso
        (essencial: sem isso, uma animação de um caso anterior continua rodando
        indefinidamente mesmo depois de trocar pra regime permanente)."""
        self._parar()
        self.frames = None
        self.tempos = None
        self._idx = 0
        self.slider.configure(from_=0, to=1, number_of_steps=1)
        self.slider.set(0)
        self.lbl_tempo.configure(text="t = -- s")
        self._msg_vazio("Sem animação (regime permanente, ou ainda não resolvido).")

    def _parar(self):
        self._playing = False
        self.btn_play.configure(text="▶ Reproduzir")
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None

    def carregar(self, frames, tempos, fps, mesh, cmap="turbo"):
        self._parar()
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
        self.canvas.get_tk_widget().destroy()
        plt.close(self.fig)
        self.fig = plt.figure(figsize=self.figsize)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self._frame_canvas)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

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
        if not self.frames:
            return
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
        self._after_id = self.after(int(1000 / self.fps), self._tick)


class AbaPerfis(ctk.CTkFrame):
    """
    Extração de perfis (corte horizontal/vertical) e ponto de amostragem.

    - Perfil: escolhe uma linha horizontal (y fixo) ou vertical (x fixo) do
      campo de temperatura final e plota T ao longo da linha.
    - Amostragem: mostra a evolução de T num ponto (x,y) escolhido --
      ao longo das ITERAÇÕES (regime permanente) ou do TEMPO (transiente).
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.mesh = None
        self.T_campo = None

        f_ctrl = ctk.CTkFrame(self, fg_color="transparent", height=1)
        f_ctrl.pack(fill="x", padx=8, pady=8)

        ctk.CTkLabel(f_ctrl, text="Corte:").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.opt_corte = ctk.CTkOptionMenu(f_ctrl, values=["Horizontal (y fixo)", "Vertical (x fixo)"],
                                            width=170)
        self.opt_corte.grid(row=0, column=1, sticky="w", padx=(0, 8))
        ctk.CTkLabel(f_ctrl, text="Valor [m]:").grid(row=0, column=2, sticky="w", padx=(0, 4))
        self.ent_valor = ctk.CTkEntry(f_ctrl, width=80)
        self.ent_valor.insert(0, "0.0")
        self.ent_valor.grid(row=0, column=3, sticky="w", padx=(0, 8))
        self.btn_perfil = ctk.CTkButton(f_ctrl, text="Extrair perfil", width=120,
                                         command=self._extrair_perfil)
        self.btn_perfil.grid(row=0, column=4, sticky="w", padx=(0, 16))

        ctk.CTkLabel(f_ctrl, text="Amostra em x,y [m]:").grid(row=1, column=0, columnspan=2,
                                                               sticky="w", pady=(6, 0))
        self.ent_xa = ctk.CTkEntry(f_ctrl, width=70)
        self.ent_xa.insert(0, "0.0")
        self.ent_xa.grid(row=1, column=2, sticky="w", pady=(6, 0))
        self.ent_ya = ctk.CTkEntry(f_ctrl, width=70)
        self.ent_ya.insert(0, "0.0")
        self.ent_ya.grid(row=1, column=3, sticky="w", pady=(6, 0))
        self.btn_amostra = ctk.CTkButton(f_ctrl, text="Amostrar ponto", width=120,
                                          command=self._amostrar_ponto)
        self.btn_amostra.grid(row=1, column=4, sticky="w", pady=(6, 0), padx=(0, 16))

        self.aba_canvas = AbaCanvas(self, figsize=(6.5, 4.6))
        self.aba_canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self._callback_amostra = None  # fornecido pelo main_window

    def carregar_resultado(self, mesh, T_campo, callback_amostra):
        self.mesh = mesh
        self.T_campo = T_campo
        self._callback_amostra = callback_amostra

    def _extrair_perfil(self):
        if self.mesh is None or self.T_campo is None:
            return
        try:
            valor = float(self.ent_valor.get())
        except ValueError:
            return
        mesh = self.mesh
        if self.opt_corte.get().startswith("Horizontal"):
            j = int(np.clip(round(valor / mesh.dy), 0, mesh.Ny - 1))
            y_real = mesh.y[j]
            perfil = self.T_campo[j, :]
            eixo = mesh.x
            titulo = f"Perfil horizontal em y = {y_real:.4g} m"
            xlabel = "x [m]"
        else:
            i = int(np.clip(round(valor / mesh.dx), 0, mesh.Nx - 1))
            x_real = mesh.x[i]
            perfil = self.T_campo[:, i]
            eixo = mesh.y
            titulo = f"Perfil vertical em x = {x_real:.4g} m"
            xlabel = "y [m]"

        fig = plt.figure(figsize=(6.5, 4.6))
        ax = fig.add_subplot(111)
        ax.plot(eixo, perfil, color="#d62728", lw=1.8, marker=".", markersize=4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Temperatura [°C]")
        ax.set_title(titulo)
        ax.grid(True, linestyle="--", linewidth=0.5)
        fig.tight_layout()
        self.aba_canvas.mostrar_figura(fig)

    def _amostrar_ponto(self):
        if self._callback_amostra is None:
            return
        try:
            xa = float(self.ent_xa.get())
            ya = float(self.ent_ya.get())
        except ValueError:
            return
        self._callback_amostra(xa, ya, self.aba_canvas)


class PainelResultados(ctk.CTkTabview):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.add("Malha")
        self.add("Temperatura")
        self.add("Fluxo de Calor")
        self.add("Superfície 3D")
        self.add("Convergência")
        self.add("Animação")
        self.add("Perfis")

        self.aba_malha = AbaCanvas(self.tab("Malha"))
        self.aba_malha.pack(fill="both", expand=True)

        self.aba_temp = AbaCanvas(self.tab("Temperatura"))
        self.aba_temp.pack(fill="both", expand=True)

        self.aba_fluxo = AbaCanvas(self.tab("Fluxo de Calor"))
        self.aba_fluxo.pack(fill="both", expand=True)

        self.aba_3d = AbaCanvas(self.tab("Superfície 3D"), com_toolbar=True)
        self.aba_3d.pack(fill="both", expand=True)

        self.aba_conv = AbaConvergencia(self.tab("Convergência"))
        self.aba_conv.pack(fill="both", expand=True)

        self.aba_anim = AbaAnimacao(self.tab("Animação"))
        self.aba_anim.pack(fill="both", expand=True)

        self.aba_perfis = AbaPerfis(self.tab("Perfis"))
        self.aba_perfis.pack(fill="both", expand=True)

    # -----------------------------------------------------------------
    def mostrar_malha(self, mesh, contornos):
        fig = vz.plot_malha(mesh, contornos)
        self.aba_malha.mostrar_figura(fig)
        self.set("Malha")

    def mostrar_campo_final(self, mesh, T_campo, k, cmap="turbo"):
        self.aba_temp.mostrar_figura(vz.plot_temperatura(mesh, T_campo, cmap=cmap))
        self.aba_fluxo.mostrar_figura(vz.plot_fluxo_calor(mesh, T_campo, k))
        self.aba_3d.mostrar_figura(vz.plot_superficie3d(mesh, T_campo, cmap=cmap))

    def preparar_convergencia(self, metodo, modo="iteracao"):
        self.aba_conv.reiniciar(metodo, modo=modo)
        self.set("Convergência")

    def convergencia_sem_residuo(self, motivo):
        self.aba_conv.sem_residuo(motivo)

    def ponto_convergencia(self, it, res):
        self.aba_conv.adicionar_ponto(it, res)

    def carregar_animacao(self, frames, tempos, fps, mesh, cmap="turbo"):
        self.aba_anim.carregar(frames, tempos, fps, mesh, cmap)

    def limpar_animacao(self):
        self.aba_anim.limpar()

    def preparar_perfis(self, mesh, T_campo, callback_amostra):
        self.aba_perfis.carregar_resultado(mesh, T_campo, callback_amostra)
