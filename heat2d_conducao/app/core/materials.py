"""
materials.py
------------
Banco de dados de materiais para o simulador didático de condução de calor 2D.

Cada material é definido pelas propriedades termofísicas que aparecem na
equação da difusão de calor:

    rho * cp * dT/dt = k * (d2T/dx2 + d2T/dy2)      (material homogêneo, k=cte)

    rho : massa específica         [kg/m3]
    cp  : calor específico         [J/(kg.K)]
    k   : condutividade térmica    [W/(m.K)]

A difusividade térmica alpha = k / (rho*cp)  [m2/s] é calculada automaticamente
e é o parâmetro que efetivamente governa a física do problema transiente.
"""

from dataclasses import dataclass


@dataclass
class Material:
    nome: str
    rho: float   # kg/m3
    cp: float    # J/(kg.K)
    k: float     # W/(m.K)

    @property
    def alpha(self) -> float:
        """Difusividade térmica alpha = k/(rho*cp)  [m2/s]."""
        return self.k / (self.rho * self.cp)

    def __str__(self):
        return (f"{self.nome}: rho={self.rho:.1f} kg/m3, cp={self.cp:.1f} J/(kg.K), "
                f"k={self.k:.2f} W/(m.K)  =>  alpha={self.alpha:.3e} m2/s")


# Banco de dados de materiais pré-definidos (valores típicos de literatura, ~25 C).
# OBS: valores de referência para fins didáticos; para trabalhos que exijam
# precisão, o usuário deve inserir valores específicos via material_customizado().
MATERIAIS_PADRAO = {
    "Alumínio":    Material("Alumínio",    2700.0, 900.0, 205.0),
    "Cobre":       Material("Cobre",       8960.0, 385.0, 401.0),
    "Ferro":       Material("Ferro",       7870.0, 449.0,  80.0),
    "Aço Carbono": Material("Aço Carbono", 7850.0, 490.0,  50.0),
    "Vidro":       Material("Vidro",       2500.0, 840.0,   1.0),
}


def material_customizado(nome: str, rho: float, cp: float, k: float) -> Material:
    """Cria um material customizado a partir de dados fornecidos pelo usuário."""
    if rho <= 0 or cp <= 0 or k <= 0:
        raise ValueError("rho, cp e k devem ser estritamente positivos.")
    return Material(nome, rho, cp, k)
