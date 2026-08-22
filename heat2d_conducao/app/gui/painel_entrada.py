"""
painel_entrada.py
------------------
Painel esquerdo (rolável) com todos os parâmetros de entrada do problema:
geometria, material, malha, condições de contorno, regime (permanente/
transiente) e método de solução.

Este módulo só COLETA e VALIDA os parâmetros -- a orquestração (rodar a
simulação, atualizar resultados) fica em main_window.py.
"""

import customtkinter as ctk
from app.core.mesh import Mesh2D
from app.core.bc import ContornosRetangulo
from app.core import solver as sv
from .widgets import FrameBC, FrameMaterial


class PainelEntrada(ctk.CTkScrollableFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, label_text="Parâmetros de entrada", **kwargs)
        self.grid_columnconfigure(0, weight=1)

        self._secao("1. Geometria")
        f_geo = ctk.CTkFrame(self, fg_color="transparent", height=1)
        f_geo.pack(fill="x", padx=10, pady=(0, 8))
        f_geo.grid_columnconfigure((1, 3), weight=1)
        ctk.CTkLabel(f_geo, text="Lx [m]:").grid(row=0, column=0, sticky="w")
        self.ent_Lx = ctk.CTkEntry(f_geo, width=90)
        self.ent_Lx.insert(0, "0.40")
        self.ent_Lx.grid(row=0, column=1, sticky="ew", padx=(4, 10))
        ctk.CTkLabel(f_geo, text="Ly [m]:").grid(row=0, column=2, sticky="w")
        self.ent_Ly = ctk.CTkEntry(f_geo, width=90)
        self.ent_Ly.insert(0, "0.25")
        self.ent_Ly.grid(row=0, column=3, sticky="ew", padx=(4, 0))

        self._secao("2. Material")
        self.frame_material = FrameMaterial(self, fg_color="transparent", height=1)
        self.frame_material.pack(fill="x", padx=10, pady=(0, 8))

        self._secao("3. Malha (cartesiana ortogonal, espaçamento uniforme por direção)")
        f_malha = ctk.CTkFrame(self, fg_color="transparent", height=1)
        f_malha.pack(fill="x", padx=10, pady=(0, 8))
        f_malha.grid_columnconfigure((1, 3), weight=1)
        ctk.CTkLabel(f_malha, text="Nx (nós em x):").grid(row=0, column=0, sticky="w")
        self.ent_Nx = ctk.CTkEntry(f_malha, width=90)
        self.ent_Nx.insert(0, "41")
        self.ent_Nx.grid(row=0, column=1, sticky="ew", padx=(4, 10))
        ctk.CTkLabel(f_malha, text="Ny (nós em y):").grid(row=0, column=2, sticky="w")
        self.ent_Ny = ctk.CTkEntry(f_malha, width=90)
        self.ent_Ny.insert(0, "26")
        self.ent_Ny.grid(row=0, column=3, sticky="ew", padx=(4, 0))
        self.lbl_dxdy = ctk.CTkLabel(f_malha, text="", text_color="gray60", font=ctk.CTkFont(size=11))
        self.lbl_dxdy.grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))
        for ent in (self.ent_Lx, self.ent_Ly, self.ent_Nx, self.ent_Ny):
            ent.bind("<KeyRelease>", lambda e: self._atualizar_labels())

        self._secao("4. Condições de contorno")
        self.bc_esquerda = FrameBC(self, "Esquerda", "#ff7f0e", fg_color=("gray92", "gray20"), height=1)
        self.bc_esquerda.pack(fill="x", padx=10, pady=3)
        self.bc_direita = FrameBC(self, "Direita", "#9467bd", fg_color=("gray92", "gray20"), height=1)
        self.bc_direita.pack(fill="x", padx=10, pady=3)
        self.bc_inferior = FrameBC(self, "Inferior", "#d62728", fg_color=("gray92", "gray20"), height=1)
        self.bc_inferior.pack(fill="x", padx=10, pady=3)
        self.bc_superior = FrameBC(self, "Superior", "#1f77b4", fg_color=("gray92", "gray20"), height=1)
        self.bc_superior.pack(fill="x", padx=10, pady=(3, 8))

        self._secao("5. Regime")
        f_regime = ctk.CTkFrame(self, fg_color="transparent", height=1)
        f_regime.pack(fill="x", padx=10, pady=(0, 4))
        self.seg_regime = ctk.CTkSegmentedButton(f_regime, values=["Permanente", "Transiente"],
                                                  command=self._on_regime_change)
        self.seg_regime.set("Permanente")
        self.seg_regime.pack(fill="x")

        f_tini = ctk.CTkFrame(self, fg_color="transparent", height=1)
        f_tini.pack(fill="x", padx=10, pady=(4, 4))
        ctk.CTkLabel(f_tini, text="T inicial / chute [°C]:").pack(side="left")
        self.ent_Tini = ctk.CTkEntry(f_tini, width=90)
        self.ent_Tini.insert(0, "25.0")
        self.ent_Tini.pack(side="left", padx=(6, 0))

        # sub-painel exclusivo do regime transiente
        self.frame_transiente = ctk.CTkFrame(self, fg_color=("gray90", "gray17"), height=1)
        self._montar_frame_transiente()

        self._secao("6. Método de solução")
        self.frame_solver = ctk.CTkFrame(self, fg_color="transparent", height=1)
        self.frame_solver.pack(fill="x", padx=10, pady=(0, 8))
        self.frame_solver.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self.frame_solver, text="Solver:").grid(row=0, column=0, sticky="w")
        self.opt_solver = ctk.CTkOptionMenu(
            self.frame_solver, values=["Jacobi", "Gauss-Seidel", "SOR", "Direto"],
            command=self._on_solver_change)
        self.opt_solver.set("SOR")
        self.opt_solver.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        self.lbl_omega = ctk.CTkLabel(self.frame_solver, text="ω (SOR):")
        self.ent_omega = ctk.CTkEntry(self.frame_solver, width=90)
        self.ent_omega.insert(0, "1.85")

        ctk.CTkLabel(self.frame_solver, text="Tolerância:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.ent_tol = ctk.CTkEntry(self.frame_solver, width=90)
        self.ent_tol.insert(0, "1e-7")
        self.ent_tol.grid(row=2, column=1, sticky="w", padx=(6, 0), pady=(4, 0))

        ctk.CTkLabel(self.frame_solver, text="Máx. iterações:").grid(row=3, column=0, sticky="w", pady=(4, 0))
        self.ent_maxiter = ctk.CTkEntry(self.frame_solver, width=90)
        self.ent_maxiter.insert(0, "30000")
        self.ent_maxiter.grid(row=3, column=1, sticky="w", padx=(6, 0), pady=(4, 0))

        self._on_solver_change(self.opt_solver.get())
        self._atualizar_labels()

    # -----------------------------------------------------------------
    def _secao(self, texto):
        lbl = ctk.CTkLabel(self, text=texto, font=ctk.CTkFont(weight="bold", size=13),
                            anchor="w", justify="left", wraplength=340)
        lbl.pack(fill="x", padx=10, pady=(12, 4))

    def _montar_frame_transiente(self):
        f = self.frame_transiente
        for w in f.winfo_children():
            w.destroy()
        f.grid_columnconfigure((1, 3), weight=1)

        ctk.CTkLabel(f, text="Esquema:").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 2))
        self.seg_esquema = ctk.CTkSegmentedButton(f, values=["Explícito", "Implícito"],
                                                   command=lambda v: self._atualizar_labels())
        self.seg_esquema.set("Explícito")
        self.seg_esquema.grid(row=0, column=1, columnspan=3, sticky="ew", padx=8, pady=(8, 2))

        self.lbl_dtcrit = ctk.CTkLabel(f, text="", text_color="#e07b00", font=ctk.CTkFont(size=11, weight="bold"))
        self.lbl_dtcrit.grid(row=1, column=0, columnspan=4, sticky="w", padx=8)

        ctk.CTkLabel(f, text="Δt [s]:").grid(row=2, column=0, sticky="w", padx=8, pady=2)
        self.ent_dt = ctk.CTkEntry(f, width=90)
        self.ent_dt.insert(0, "1.0")
        self.ent_dt.grid(row=2, column=1, sticky="ew", padx=(4, 8), pady=2)
        btn_dtcrit = ctk.CTkButton(f, text="usar 80% do Δt crítico", width=140,
                                    command=self._usar_dt_critico)
        btn_dtcrit.grid(row=2, column=2, columnspan=2, sticky="ew", padx=(0, 8), pady=2)

        ctk.CTkLabel(f, text="t final [s]:").grid(row=3, column=0, sticky="w", padx=8, pady=2)
        self.ent_tfinal = ctk.CTkEntry(f, width=90)
        self.ent_tfinal.insert(0, "60.0")
        self.ent_tfinal.grid(row=3, column=1, sticky="ew", padx=(4, 8), pady=2)

        ctk.CTkLabel(f, text="Nº quadros salvos:").grid(row=3, column=2, sticky="w", padx=(0, 4))
        self.ent_nframes = ctk.CTkEntry(f, width=70)
        self.ent_nframes.insert(0, "80")
        self.ent_nframes.grid(row=3, column=3, sticky="ew", padx=(0, 8), pady=2)

        ctk.CTkLabel(f, text="FPS da animação:").grid(row=4, column=0, sticky="w", padx=8, pady=(2, 8))
        self.ent_fps = ctk.CTkEntry(f, width=90)
        self.ent_fps.insert(0, "12")
        self.ent_fps.grid(row=4, column=1, sticky="ew", padx=(4, 8), pady=(2, 8))

        for ent in (self.ent_dt,):
            ent.bind("<KeyRelease>", lambda e: self._atualizar_labels())

    def _on_regime_change(self, valor):
        if valor == "Transiente":
            self.frame_transiente.pack(fill="x", padx=10, pady=(0, 8), before=self._widget_secao6())
        else:
            self.frame_transiente.pack_forget()
        self._atualizar_labels()

    def _widget_secao6(self):
        # localiza o label "6. Método de solução" para inserir o frame transiente antes dele
        for child in self.winfo_children():
            if isinstance(child, ctk.CTkLabel) and child.cget("text").startswith("6."):
                return child
        return None

    def _on_solver_change(self, valor):
        if valor == "SOR":
            self.lbl_omega.grid(row=1, column=0, sticky="w", pady=(4, 0))
            self.ent_omega.grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(4, 0))
        else:
            self.lbl_omega.grid_forget()
            self.ent_omega.grid_forget()

    def _usar_dt_critico(self):
        try:
            dtc = self._calcular_dt_critico()
            if dtc is not None:
                self.ent_dt.delete(0, "end")
                self.ent_dt.insert(0, f"{0.8*dtc:.6g}")
        except Exception:
            pass
        self._atualizar_labels()

    def _calcular_dt_critico(self):
        try:
            Lx = float(self.ent_Lx.get()); Ly = float(self.ent_Ly.get())
            Nx = int(self.ent_Nx.get()); Ny = int(self.ent_Ny.get())
            mesh = Mesh2D(Lx, Ly, Nx, Ny)
            mat = self.frame_material.obter_material()
            return sv.dt_critico(mat.alpha, mesh.dx, mesh.dy)
        except Exception:
            return None

    def _atualizar_labels(self):
        try:
            Lx = float(self.ent_Lx.get()); Ly = float(self.ent_Ly.get())
            Nx = int(self.ent_Nx.get()); Ny = int(self.ent_Ny.get())
            mesh = Mesh2D(Lx, Ly, Nx, Ny)
            self.lbl_dxdy.configure(
                text=f"dx = {mesh.dx:.4g} m   dy = {mesh.dy:.4g} m   "
                     f"({mesh.n_nos} graus de liberdade)")
        except Exception:
            self.lbl_dxdy.configure(text="(preencha Lx, Ly, Nx, Ny)")

        if hasattr(self, "seg_esquema") and self.seg_esquema.get() == "Explícito":
            dtc = self._calcular_dt_critico()
            if dtc is not None:
                try:
                    dt_atual = float(self.ent_dt.get())
                except ValueError:
                    dt_atual = None
                aviso = ""
                if dt_atual is not None and dt_atual > dtc:
                    aviso = "   ⚠ INSTÁVEL: Δt > Δt_crítico!"
                self.lbl_dtcrit.configure(
                    text=f"Δt crítico (estabilidade FTCS) = {dtc:.4g} s{aviso}",
                    text_color="#c0392b" if aviso else "#e07b00")
        elif hasattr(self, "lbl_dtcrit"):
            self.lbl_dtcrit.configure(text="(esquema implícito: incondicionalmente estável)",
                                       text_color="gray60")

    # -----------------------------------------------------------------
    def montar_mesh(self) -> Mesh2D:
        Lx = float(self.ent_Lx.get()); Ly = float(self.ent_Ly.get())
        Nx = int(self.ent_Nx.get()); Ny = int(self.ent_Ny.get())
        return Mesh2D(Lx, Ly, Nx, Ny)

    def montar_contornos(self) -> ContornosRetangulo:
        return ContornosRetangulo(
            esquerda=self.bc_esquerda.obter_bc(),
            direita=self.bc_direita.obter_bc(),
            inferior=self.bc_inferior.obter_bc(),
            superior=self.bc_superior.obter_bc(),
        )

    def coletar_parametros(self) -> dict:
        """Lê e valida todos os campos da GUI, retornando um dict pronto para o worker."""
        mesh = self.montar_mesh()
        contornos = self.montar_contornos()
        material = self.frame_material.obter_material()

        regime = "permanente" if self.seg_regime.get() == "Permanente" else "transiente"
        metodo_map = {"Jacobi": "jacobi", "Gauss-Seidel": "gauss_seidel",
                      "SOR": "sor", "Direto": "direto"}
        solver_metodo = metodo_map[self.opt_solver.get()]

        params = {
            "mesh": mesh,
            "contornos": contornos,
            "material": material,
            "regime": regime,
            "T_inicial": float(self.ent_Tini.get()),
            "solver_metodo": solver_metodo,
            "tolerancia": float(self.ent_tol.get()),
            "max_iter": int(self.ent_maxiter.get()),
        }
        if solver_metodo == "sor":
            params["omega"] = float(self.ent_omega.get())

        if regime == "transiente":
            params["esquema"] = "explicito" if self.seg_esquema.get() == "Explícito" else "implicito"
            params["dt"] = float(self.ent_dt.get())
            params["t_final"] = float(self.ent_tfinal.get())
            params["n_frames_alvo"] = int(self.ent_nframes.get())
            params["fps"] = int(self.ent_fps.get())
            if params["esquema"] == "explicito":
                dtc = sv.dt_critico(material.alpha, mesh.dx, mesh.dy)
                if params["dt"] > dtc:
                    raise ValueError(
                        f"Δt = {params['dt']:.4g} s excede o Δt crítico ({dtc:.4g} s) "
                        f"para o esquema explícito -- a solução será instável. "
                        f"Reduza Δt ou use o esquema implícito.")
        return params
