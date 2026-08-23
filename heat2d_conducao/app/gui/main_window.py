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
import json
import numpy as np

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
        self.geometry("1400x880")
        self.minsize(1150, 720)

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

        self.btn_salvar_cfg = ctk.CTkButton(f_botoes, text="💾 Salvar setup", fg_color="gray40",
                                             hover_color="gray30", command=self.salvar_configuracao)
        self.btn_salvar_cfg.grid(row=2, column=0, sticky="ew", padx=(0, 3), pady=2)

        self.btn_carregar_cfg = ctk.CTkButton(f_botoes, text="📂 Carregar setup", fg_color="gray40",
                                               hover_color="gray30", command=self.carregar_configuracao)
        self.btn_carregar_cfg.grid(row=2, column=1, sticky="ew", padx=(3, 0), pady=2)

        self.lbl_status = ctk.CTkLabel(col_esq, text="Pronto.", anchor="w",
                                        text_color="gray50", font=ctk.CTkFont(size=11))
        self.lbl_status.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        # barra DETERMINÍSTICA (0 a 1) -- mostra progresso real, não um "vai e volta"
        self.barra_progresso = ctk.CTkProgressBar(col_esq, mode="determinate")
        self.barra_progresso.set(0.0)
        self.barra_progresso.grid(row=3, column=0, sticky="ew", pady=(2, 0))

        # ---- coluna direita: resultados ----
        self.painel_resultados = PainelResultados(self)
        self.painel_resultados.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)
        # o callback de amostragem pós-hoc (aba Perfis) só funciona pra transiente
        # (usa os frames já guardados); ligamos aqui para ter acesso a self.ultimo_resultado
        self.painel_resultados.aba_perfis._callback_amostra = self._amostrar_ponto_pos_hoc

        # ---- estado interno ----
        self.fila = queue.Queue()
        self.worker = None
        self.ultimo_resultado = None
        self.ultimos_params = None
        self._amostra_eixo = []
        self._amostra_valores = []

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
        self._amostra_eixo = []
        self._amostra_valores = []

        # PARA qualquer animação de um caso anterior antes de rodar o novo --
        # sem isso, uma animação tocando continua rodando indefinidamente
        # mesmo depois de mudar para um caso novo (bug reportado)
        self.painel_resultados.limpar_animacao()

        # configura a aba de convergência de acordo com regime/esquema.
        # A formulação é sempre "transiente" (ver solver.residuo_pde): em
        # permanente, Δt é só um artifício de relaxação e o resíduo evolui
        # ITERAÇÃO a iteração; em transiente, Δt é físico e o resíduo evolui
        # no TEMPO -- mesma fórmula ‖A·T+F‖ nos dois casos, calculável para
        # qualquer esquema (explícito incluso, pois não depende de solver).
        if params["regime"] == "permanente":
            self.painel_resultados.preparar_convergencia(
                self.painel_entrada.opt_solver.get(), modo="iteracao")
        else:
            metodo_label = ("Euler explícito" if params["esquema"] == "explicito"
                             else f"Euler implícito ({self.painel_entrada.opt_solver.get()})")
            self.painel_resultados.preparar_convergencia(metodo_label, modo="tempo")

        self.btn_resolver.configure(state="disabled")
        self.btn_previsualizar.configure(state="disabled")
        self.btn_cancelar.configure(state="normal")
        self.btn_exportar.configure(state="disabled")
        self.barra_progresso.set(0.0)

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
        # processa no máximo N mensagens por tick do event loop -- se a fila
        # tiver milhares de mensagens acumuladas (ex.: transiente com muitos
        # passos), processar TODAS de uma vez trava a GUI (Tk fica "Not
        # Responding" até esvaziar a fila). Processar em lotes menores
        # mantém a interface responsiva mesmo em simulações longas.
        MAX_MSGS_POR_TICK = 40
        try:
            for _ in range(MAX_MSGS_POR_TICK):
                msg = self.fila.get_nowait()
                tipo = msg[0]
                if tipo == "status":
                    self.lbl_status.configure(text=msg[1])
                elif tipo == "progresso":
                    self.barra_progresso.set(float(msg[1]))
                elif tipo == "convergencia":
                    _, it, res = msg
                    self.painel_resultados.ponto_convergencia(it, res)
                elif tipo == "convergencia_passo":
                    _, n, res = msg
                    self.painel_resultados.ponto_convergencia(n, res)
                elif tipo == "amostra":
                    _, x_eixo, valor = msg
                    self._amostra_eixo.append(x_eixo)
                    self._amostra_valores.append(valor)
                elif tipo == "frame_transiente":
                    pass  # frames completos chegam em "concluido"
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
        self.after(30, self._poll_fila)

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
            self.painel_resultados.limpar_animacao()
            self.painel_resultados.set("Temperatura")

        # aba Perfis: perfil final sempre disponível; histórico de amostragem
        # (se o usuário habilitou) é plotado automaticamente
        self.painel_resultados.preparar_perfis(mesh, resultado["T_campo"],
                                                self._amostrar_ponto_pos_hoc)
        if self._amostra_eixo:
            xlabel = "Iteração" if resultado["tipo"] == "permanente" else "Tempo [s]"
            xy = params.get("amostra_xy_real", ("?", "?"))
            titulo = f"T no ponto (x={xy[0]:.4g}, y={xy[1]:.4g}) m"
            self._plotar_amostra(self._amostra_eixo, self._amostra_valores, xlabel, titulo)
            self.painel_resultados.set("Perfis")

        self.lbl_status.configure(text="Concluído.")
        self.barra_progresso.set(1.0)
        self._resetar_botoes()
        self.btn_exportar.configure(state="normal")

    def _resetar_botoes(self):
        self.btn_resolver.configure(state="normal")
        self.btn_previsualizar.configure(state="normal")
        self.btn_cancelar.configure(state="disabled")

    # -----------------------------------------------------------------
    def _plotar_amostra(self, eixo, valores, xlabel, titulo):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6.5, 4.6))
        ax = fig.add_subplot(111)
        ax.plot(eixo, valores, color="#2e7d32", lw=1.6, marker=".", markersize=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Temperatura [°C]")
        ax.set_title(titulo)
        ax.grid(True, linestyle="--", linewidth=0.5)
        fig.tight_layout()
        self.painel_resultados.aba_perfis.aba_canvas.mostrar_figura(fig)

    def _amostrar_ponto_pos_hoc(self, xa, ya, aba_canvas):
        """
        Amostragem PÓS-HOC (depois da simulação já ter rodado), usada pelos
        botões da aba Perfis. Só funciona para regime TRANSIENTE, pois só
        nesse caso os quadros completos ficam guardados em memória
        (para o permanente, guardar o campo inteiro em toda iteração
        consumiria memória demais -- por isso a amostragem no permanente
        precisa ser habilitada ANTES de rodar, na seção 7 do painel de entrada).
        """
        if self.ultimo_resultado is None or self.ultimos_params is None:
            messagebox.showinfo("Sem resultados", "Rode uma simulação primeiro.")
            return
        mesh = self.ultimos_params["mesh"]
        if not (0 <= xa <= mesh.Lx and 0 <= ya <= mesh.Ly):
            messagebox.showerror("Ponto inválido", "Coordenadas fora do domínio.")
            return
        i = int(np.clip(round(xa / mesh.dx), 0, mesh.Nx - 1))
        j = int(np.clip(round(ya / mesh.dy), 0, mesh.Ny - 1))

        if self.ultimo_resultado["tipo"] == "transiente":
            valores = [campo[j, i] for campo in self.ultimo_resultado["frames"]]
            tempos = self.ultimo_resultado["tempos"]
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(6.5, 4.6))
            ax = fig.add_subplot(111)
            ax.plot(tempos, valores, color="#2e7d32", lw=1.6, marker=".", markersize=3)
            ax.set_xlabel("Tempo [s]")
            ax.set_ylabel("Temperatura [°C]")
            ax.set_title(f"T no ponto (x={mesh.x[i]:.4g}, y={mesh.y[j]:.4g}) m "
                          f"-- {len(tempos)} quadros salvos")
            ax.grid(True, linestyle="--", linewidth=0.5)
            fig.tight_layout()
            aba_canvas.mostrar_figura(fig)
        else:
            valor = self.ultimo_resultado["T_campo"][j, i]
            messagebox.showinfo(
                "Amostragem pós-hoc indisponível no permanente",
                f"T no ponto (x={mesh.x[i]:.4g}, y={mesh.y[j]:.4g}) = {valor:.4g} °C "
                f"(valor final convergido).\n\n"
                "Para ver a evolução desse valor ao longo das ITERAÇÕES, habilite "
                "'Amostragem de ponto' na seção 7 do painel de entrada ANTES de "
                "clicar em Resolver, e rode novamente.")

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

            np.savetxt(os.path.join(pasta, "campo_temperatura.csv"), resultado["T_campo"],
                       delimiter=",", header="Campo de temperatura [Ny x Nx], graus C",
                       comments="# ")

            if resultado["tipo"] == "transiente":
                vz.animar_transiente(mesh, resultado["frames"], resultado["tempos"],
                                      framerate=params["fps"],
                                      caminho_saida=os.path.join(pasta, "transiente.gif"))

            if self._amostra_eixo:
                arr = np.column_stack([self._amostra_eixo, self._amostra_valores])
                np.savetxt(os.path.join(pasta, "amostra_ponto.csv"), arr, delimiter=",",
                           header="eixo(iteracao_ou_tempo),temperatura_C", comments="# ")

            messagebox.showinfo("Exportação concluída", f"Resultados salvos em:\n{pasta}")
        except Exception as e:
            messagebox.showerror("Erro ao exportar", str(e))

    # -----------------------------------------------------------------
    def salvar_configuracao(self):
        caminho = filedialog.asksaveasfilename(
            title="Salvar configuração", defaultextension=".json",
            filetypes=[("Configuração JSON", "*.json")])
        if not caminho:
            return
        try:
            cfg = self.painel_entrada.exportar_config()
            with open(caminho, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
            self.lbl_status.configure(text=f"Configuração salva em {os.path.basename(caminho)}.")
        except Exception as e:
            messagebox.showerror("Erro ao salvar configuração", str(e))

    def carregar_configuracao(self):
        caminho = filedialog.askopenfilename(
            title="Carregar configuração", filetypes=[("Configuração JSON", "*.json")])
        if not caminho:
            return
        try:
            with open(caminho, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.painel_entrada.importar_config(cfg)
            self.lbl_status.configure(text=f"Configuração carregada de {os.path.basename(caminho)}.")
        except Exception as e:
            messagebox.showerror("Erro ao carregar configuração", str(e))
