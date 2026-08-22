"""
main_window.py
---------------
Janela principal do simulador. Orquestra:
    - PainelEntrada (esquerda)   -- coleta de parâmetros
    - PainelResultados (direita) -- abas de visualização
    - Fluxo de confirmação da malha antes de simular
    - SimulacaoWorker rodando em thread separada
    - Polling periódico da fila de progresso (queue) para atualizar a GUI
      com segurança (nunca se mexe em widgets a partir de outra thread)
"""

import customtkinter as ctk
from tkinter import messagebox, filedialog
import queue
import os

from .painel_entrada import PainelEntrada
from .painel_resultados import PainelResultados
from .worker import SimulacaoWorker
from app.core import visualize as vz

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


class JanelaPrincipal(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Condução de Calor 2D -- Simulador Didático (DEQ-UEM)")
        self.geometry("1360x860")
        self.minsize(1100, 700)

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---- coluna esquerda: entrada + botões de ação ----
        col_esq = ctk.CTkFrame(self, fg_color="transparent", height=1)
        col_esq.grid(row=0, column=0, sticky="nsw", padx=(8, 4), pady=8)
        col_esq.grid_rowconfigure(0, weight=1)

        self.painel_entrada = PainelEntrada(col_esq, width=380)
        self.painel_entrada.grid(row=0, column=0, sticky="nsew")

        f_botoes = ctk.CTkFrame(col_esq, fg_color="transparent", height=1)
        f_botoes.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        f_botoes.grid_columnconfigure((0, 1), weight=1)

        self.btn_previsualizar = ctk.CTkButton(f_botoes, text="Pré-visualizar malha",
                                                command=self.previsualizar_malha)
        self.btn_previsualizar.grid(row=0, column=0, sticky="ew", padx=(0, 3), pady=2)

        self.btn_resolver = ctk.CTkButton(f_botoes, text="▶ Resolver", fg_color="#2e7d32",
                                           hover_color="#1b5e20", command=self.iniciar_simulacao)
        self.btn_resolver.grid(row=0, column=1, sticky="ew", padx=(3, 0), pady=2)

        self.btn_cancelar = ctk.CTkButton(f_botoes, text="✕ Cancelar", fg_color="#c0392b",
                                           hover_color="#922b21", command=self.cancelar_simulacao,
                                           state="disabled")
        self.btn_cancelar.grid(row=1, column=0, sticky="ew", padx=(0, 3), pady=2)

        self.btn_exportar = ctk.CTkButton(f_botoes, text="⭳ Exportar resultados",
                                           command=self.exportar_resultados, state="disabled")
        self.btn_exportar.grid(row=1, column=1, sticky="ew", padx=(3, 0), pady=2)

        self.lbl_status = ctk.CTkLabel(col_esq, text="Pronto.", anchor="w",
                                        text_color="gray50", font=ctk.CTkFont(size=11))
        self.lbl_status.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        self.barra_progresso = ctk.CTkProgressBar(col_esq, mode="indeterminate")
        self.barra_progresso.grid(row=3, column=0, sticky="ew", pady=(2, 0))

        # ---- coluna direita: resultados ----
        self.painel_resultados = PainelResultados(self)
        self.painel_resultados.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)

        # ---- estado interno ----
        self.fila = queue.Queue()
        self.worker = None
        self.ultimo_resultado = None
        self.ultimos_params = None

    # -----------------------------------------------------------------
    def previsualizar_malha(self):
        try:
            mesh = self.painel_entrada.montar_mesh()
            contornos = self.painel_entrada.montar_contornos()
        except Exception as e:
            messagebox.showerror("Erro nos parâmetros", str(e))
            return
        self.painel_resultados.mostrar_malha(mesh, contornos)
        return mesh, contornos

    def iniciar_simulacao(self):
        try:
            params = self.painel_entrada.coletar_parametros()
        except Exception as e:
            messagebox.showerror("Erro nos parâmetros", str(e))
            return

        # exige confirmação visual da malha antes de rodar (pedido explícito do usuário)
        self.painel_resultados.mostrar_malha(params["mesh"], params["contornos"])
        confirmado = messagebox.askyesno(
            "Confirmar malha",
            f"Malha: {params['mesh'].Nx}×{params['mesh'].Ny} nós "
            f"(dx={params['mesh'].dx:.4g} m, dy={params['mesh'].dy:.4g} m).\n\n"
            "A malha e as condições de contorno estão corretas na aba 'Malha'?\n"
            "Deseja iniciar a simulação?")
        if not confirmado:
            return

        self.ultimos_params = params
        self.painel_resultados.preparar_convergencia(
            self.painel_entrada.opt_solver.get() if params["regime"] == "permanente"
            or params.get("esquema") == "implicito" else "explícito (sem solver linear)")

        self.btn_resolver.configure(state="disabled")
        self.btn_previsualizar.configure(state="disabled")
        self.btn_cancelar.configure(state="normal")
        self.btn_exportar.configure(state="disabled")
        self.barra_progresso.start()

        self.worker = SimulacaoWorker(params, self.fila)
        self.worker.start()
        self.after(50, self._poll_fila)

    def cancelar_simulacao(self):
        if self.worker is not None:
            self.worker.cancelar()
        self.btn_cancelar.configure(state="disabled")
        self.lbl_status.configure(text="Cancelando...")

    # -----------------------------------------------------------------
    def _poll_fila(self):
        try:
            while True:
                msg = self.fila.get_nowait()
                tipo = msg[0]
                if tipo == "status":
                    self.lbl_status.configure(text=msg[1])
                elif tipo == "convergencia":
                    _, it, res = msg
                    self.painel_resultados.ponto_convergencia(it, res)
                    self.lbl_status.configure(text=f"Iteração {it}  |  resíduo = {res:.3e}")
                elif tipo == "frame_transiente":
                    pass  # frames completos chegam em "concluido"; status já é atualizado
                elif tipo == "concluido":
                    self._finalizar(msg[1])
                    return
                elif tipo == "cancelado":
                    self.lbl_status.configure(text="Simulação cancelada.")
                    self._resetar_botoes()
                    return
                elif tipo == "erro":
                    self.lbl_status.configure(text="Erro na simulação.")
                    messagebox.showerror("Erro na simulação", msg[1])
                    self._resetar_botoes()
                    return
        except queue.Empty:
            pass
        self.after(50, self._poll_fila)

    def _finalizar(self, resultado):
        self.ultimo_resultado = resultado
        params = self.ultimos_params
        mesh = params["mesh"]
        k = params["material"].k

        self.painel_resultados.mostrar_campo_final(mesh, resultado["T_campo"], k)

        if resultado["tipo"] == "transiente":
            self.painel_resultados.carregar_animacao(
                resultado["frames"], resultado["tempos"], params["fps"], mesh)
            self.painel_resultados.set("Animação")
        else:
            self.painel_resultados.set("Temperatura")

        self.lbl_status.configure(text="Concluído.")
        self._resetar_botoes()
        self.btn_exportar.configure(state="normal")

    def _resetar_botoes(self):
        self.btn_resolver.configure(state="normal")
        self.btn_previsualizar.configure(state="normal")
        self.btn_cancelar.configure(state="disabled")
        self.barra_progresso.stop()

    # -----------------------------------------------------------------
    def exportar_resultados(self):
        if self.ultimo_resultado is None:
            return
        pasta = filedialog.askdirectory(title="Escolha a pasta para exportar os resultados")
        if not pasta:
            return

        params = self.ultimos_params
        mesh = params["mesh"]
        k = params["material"].k
        resultado = self.ultimo_resultado

        try:
            vz.plot_malha(mesh, params["contornos"]).savefig(
                os.path.join(pasta, "malha.png"), dpi=150)
            vz.plot_temperatura(mesh, resultado["T_campo"]).savefig(
                os.path.join(pasta, "temperatura.png"), dpi=150)
            vz.plot_fluxo_calor(mesh, resultado["T_campo"], k).savefig(
                os.path.join(pasta, "fluxo_calor.png"), dpi=150)
            vz.plot_superficie3d(mesh, resultado["T_campo"]).savefig(
                os.path.join(pasta, "superficie3d.png"), dpi=150)

            import numpy as np
            np.savetxt(os.path.join(pasta, "campo_temperatura.csv"), resultado["T_campo"],
                       delimiter=",", header="Campo de temperatura [Ny x Nx], graus C",
                       comments="# ")

            if resultado["tipo"] == "transiente":
                vz.animar_transiente(mesh, resultado["frames"], resultado["tempos"],
                                      framerate=params["fps"],
                                      caminho_saida=os.path.join(pasta, "transiente.gif"))

            messagebox.showinfo("Exportação concluída", f"Resultados salvos em:\n{pasta}")
        except Exception as e:
            messagebox.showerror("Erro ao exportar", str(e))
