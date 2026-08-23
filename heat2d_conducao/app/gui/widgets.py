"""
widgets.py
----------
Widgets reutilizáveis da GUI:
    - FrameBC       : configuração da condição de contorno de UMA face
    - FrameMaterial : seleção de material (padrão ou customizado)

Mantidos separados do painel de entrada principal para facilitar reuso e
manutenção (cada face do retângulo usa uma instância independente de FrameBC).
"""

import customtkinter as ctk
from app.core.bc import BC
from app.core.materials import MATERIAIS_PADRAO, material_customizado

TIPO_LABEL = {
    "dirichlet": "Temperatura prescrita (Dirichlet)",
    "neumann": "Fluxo prescrito (Neumann)",
    "conveccao": "Convecção",
    "simetria": "Simetria / Isolado",
}
LABEL_TIPO = {v: k for k, v in TIPO_LABEL.items()}


class FrameBC(ctk.CTkFrame):
    """Painel de configuração da condição de contorno de uma face do retângulo."""

    def __init__(self, master, nome_face: str, cor_indicador: str, **kwargs):
        super().__init__(master, **kwargs)
        self.nome_face = nome_face

        self.grid_columnconfigure(1, weight=1)

        faixa = ctk.CTkFrame(self, width=6, fg_color=cor_indicador, corner_radius=3)
        faixa.grid(row=0, column=0, rowspan=4, sticky="ns", padx=(2, 8), pady=2)

        titulo = ctk.CTkLabel(self, text=f"Face {nome_face}", font=ctk.CTkFont(weight="bold"))
        titulo.grid(row=0, column=1, sticky="w", pady=(2, 4))

        self.opt_tipo = ctk.CTkOptionMenu(self, values=list(TIPO_LABEL.values()),
                                           command=self._on_tipo_change)
        self.opt_tipo.grid(row=1, column=1, sticky="ew", pady=2)
        self.opt_tipo.set(TIPO_LABEL["simetria"])

        # sub-frame com os parâmetros, reconstruído a cada troca de tipo
        self.frame_params = ctk.CTkFrame(self, fg_color="transparent", height=1)
        self.frame_params.grid(row=2, column=1, sticky="ew", pady=(4, 4))
        self.frame_params.grid_columnconfigure(1, weight=1)

        self._entradas = {}
        self._on_tipo_change(self.opt_tipo.get())

    def _limpar_params(self):
        for w in self.frame_params.winfo_children():
            w.destroy()
        self._entradas = {}

    def _add_campo(self, row, rotulo, chave, valor_padrao):
        lbl = ctk.CTkLabel(self.frame_params, text=rotulo)
        lbl.grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
        ent = ctk.CTkEntry(self.frame_params, width=110)
        ent.insert(0, str(valor_padrao))
        ent.grid(row=row, column=1, sticky="ew", pady=2)
        self._entradas[chave] = ent

    def _on_tipo_change(self, label_escolhido):
        self._limpar_params()
        tipo = LABEL_TIPO[label_escolhido]
        if tipo == "dirichlet":
            self._add_campo(0, "T [°C]:", "T", 25.0)
        elif tipo == "neumann":
            self._add_campo(0, "q [W/m²] (saindo>0):", "q", 0.0)
        elif tipo == "conveccao":
            self._add_campo(0, "h [W/(m²·K)]:", "h", 10.0)
            self._add_campo(1, "T∞ [°C]:", "Tinf", 25.0)
        elif tipo == "simetria":
            info = ctk.CTkLabel(self.frame_params, text="(∂T/∂n = 0 -- sem parâmetros)",
                                 text_color="gray60", font=ctk.CTkFont(size=11))
            info.grid(row=0, column=0, columnspan=2, sticky="w")

    def obter_bc(self) -> BC:
        tipo = LABEL_TIPO[self.opt_tipo.get()]
        try:
            if tipo == "dirichlet":
                return BC("dirichlet", T=float(self._entradas["T"].get()))
            elif tipo == "neumann":
                return BC("neumann", q=float(self._entradas["q"].get()))
            elif tipo == "conveccao":
                return BC("conveccao", h=float(self._entradas["h"].get()),
                          Tinf=float(self._entradas["Tinf"].get()))
            else:
                return BC("simetria")
        except ValueError as e:
            raise ValueError(f"Face {self.nome_face}: valor inválido ({e})")

    def exportar_config(self) -> dict:
        return {"tipo_label": self.opt_tipo.get(),
                "params": {k: v.get() for k, v in self._entradas.items()}}

    def importar_config(self, cfg: dict):
        self.opt_tipo.set(cfg["tipo_label"])
        self._on_tipo_change(cfg["tipo_label"])
        for k, v in cfg.get("params", {}).items():
            if k in self._entradas:
                self._entradas[k].delete(0, "end")
                self._entradas[k].insert(0, v)


class FrameMaterial(ctk.CTkFrame):
    """Seleção de material: banco pré-definido ou propriedades customizadas."""

    ROTULO_CUSTOM = "Customizado..."

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(1, weight=1)

        opcoes = list(MATERIAIS_PADRAO.keys()) + [self.ROTULO_CUSTOM]
        ctk.CTkLabel(self, text="Material:").grid(row=0, column=0, sticky="w", pady=2)
        self.opt_material = ctk.CTkOptionMenu(self, values=opcoes, command=self._on_change)
        self.opt_material.set("Aço Carbono")
        self.opt_material.grid(row=0, column=1, sticky="ew", pady=2)

        self.frame_custom = ctk.CTkFrame(self, fg_color="transparent", height=1)
        self.frame_custom.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.frame_custom.grid_columnconfigure(1, weight=1)
        self._entradas_custom = {}

        self.lbl_alpha = ctk.CTkLabel(self, text="", text_color="gray60", font=ctk.CTkFont(size=11))
        self.lbl_alpha.grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 0))

        self._on_change(self.opt_material.get())

    def _on_change(self, valor):
        for w in self.frame_custom.winfo_children():
            w.destroy()
        self._entradas_custom = {}
        if valor == self.ROTULO_CUSTOM:
            campos = [("Nome:", "nome", "Meu material"), ("rho [kg/m³]:", "rho", "2000"),
                      ("cp [J/(kg·K)]:", "cp", "900"), ("k [W/(m·K)]:", "k", "10")]
            for i, (rot, chave, default) in enumerate(campos):
                ctk.CTkLabel(self.frame_custom, text=rot).grid(row=i, column=0, sticky="w", pady=1)
                ent = ctk.CTkEntry(self.frame_custom, width=120)
                ent.insert(0, default)
                ent.grid(row=i, column=1, sticky="ew", pady=1)
                ent.bind("<KeyRelease>", lambda e: self._atualizar_alpha())
                self._entradas_custom[chave] = ent
        self._atualizar_alpha()

    def _atualizar_alpha(self):
        try:
            mat = self.obter_material()
            self.lbl_alpha.configure(
                text=f"alpha = k/(rho·cp) = {mat.alpha:.3e} m²/s")
        except Exception:
            self.lbl_alpha.configure(text="")

    def obter_material(self):
        valor = self.opt_material.get()
        if valor != self.ROTULO_CUSTOM:
            return MATERIAIS_PADRAO[valor]
        try:
            nome = self._entradas_custom["nome"].get() or "Customizado"
            rho = float(self._entradas_custom["rho"].get())
            cp = float(self._entradas_custom["cp"].get())
            k = float(self._entradas_custom["k"].get())
            return material_customizado(nome, rho, cp, k)
        except ValueError as e:
            raise ValueError(f"Material customizado: valor inválido ({e})")

    def exportar_config(self) -> dict:
        return {"material": self.opt_material.get(),
                "custom": {k: v.get() for k, v in self._entradas_custom.items()}}

    def importar_config(self, cfg: dict):
        self.opt_material.set(cfg["material"])
        self._on_change(cfg["material"])
        for k, v in cfg.get("custom", {}).items():
            if k in self._entradas_custom:
                self._entradas_custom[k].delete(0, "end")
                self._entradas_custom[k].insert(0, v)
        self._atualizar_alpha()
