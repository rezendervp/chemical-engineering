"""
bc.py
-----
Definição das condições de contorno suportadas nas quatro faces do retângulo.

Tipos:
 - dirichlet : temperatura prescrita              {'tipo':'dirichlet', 'T': valor}
 - neumann   : fluxo de calor prescrito            {'tipo':'neumann',   'q': valor}
 - conveccao : lei do resfriamento de Newton        {'tipo':'conveccao', 'h': valor, 'Tinf': valor}
 - simetria  : gradiente nulo na normal (equivale a neumann com q=0)
               {'tipo':'simetria'}

CONVENÇÃO FÍSICA IMPORTANTE:
    q é definido na direção da NORMAL EXTERNA à face.
    q > 0  =>  calor SAINDO do domínio por aquela face
    q < 0  =>  calor ENTRANDO no domínio por aquela face
    (Lei de Fourier: q = -k * dT/dn, n = normal externa)

A convecção usa a mesma convenção: q_conv = h*(T_fronteira - Tinf), positivo
quando a face está mais quente que o fluido ambiente (calor saindo).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BC:
    tipo: str
    T: Optional[float] = None       # dirichlet [K ou C]
    q: Optional[float] = None       # neumann [W/m2]
    h: Optional[float] = None       # convecção [W/(m2.K)]
    Tinf: Optional[float] = None    # convecção [K ou C]

    def __post_init__(self):
        tipos_validos = ("dirichlet", "neumann", "conveccao", "simetria")
        if self.tipo not in tipos_validos:
            raise ValueError(f"Tipo de condição de contorno inválido: '{self.tipo}'. "
                              f"Use um de {tipos_validos}.")
        if self.tipo == "dirichlet" and self.T is None:
            raise ValueError("BC 'dirichlet' requer o parâmetro T.")
        if self.tipo == "neumann" and self.q is None:
            raise ValueError("BC 'neumann' requer o parâmetro q.")
        if self.tipo == "conveccao" and (self.h is None or self.Tinf is None):
            raise ValueError("BC 'conveccao' requer os parâmetros h e Tinf.")

    def rotulo(self) -> str:
        if self.tipo == "dirichlet":
            return f"T = {self.T:g}"
        if self.tipo == "neumann":
            return f"q = {self.q:g} W/m²"
        if self.tipo == "conveccao":
            return f"h={self.h:g} W/(m²K), T∞={self.Tinf:g}"
        return "simetria (∂T/∂n = 0)"


@dataclass
class ContornosRetangulo:
    """Condições de contorno das quatro faces do domínio retangular."""
    esquerda: BC
    direita: BC
    inferior: BC
    superior: BC

    def como_dict(self):
        return {"esquerda": self.esquerda, "direita": self.direita,
                "inferior": self.inferior, "superior": self.superior}
