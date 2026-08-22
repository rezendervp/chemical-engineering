"""
mesh.py
-------
Geração da malha cartesiana ortogonal (retangular), espaçamento uniforme em
cada direção separadamente: dx pode ser diferente de dy, mas dentro de cada
direção o espaçamento é constante (malha regular por simplicidade didática).

Convenção de índices:
    i = 0 .. Nx-1  -> direção x  (i=0: face esquerda, i=Nx-1: face direita)
    j = 0 .. Ny-1  -> direção y  (j=0: face inferior,  j=Ny-1: face superior)

O índice global (vetorizado, para montagem da matriz esparsa) é:
    p = j * Nx + i
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class Mesh2D:
    Lx: float      # comprimento do domínio em x [m]
    Ly: float      # comprimento do domínio em y [m]
    Nx: int        # número de nós em x (incluindo os dois contornos)
    Ny: int        # número de nós em y (incluindo os dois contornos)

    def __post_init__(self):
        if self.Lx <= 0 or self.Ly <= 0:
            raise ValueError("Lx e Ly devem ser positivos.")
        if self.Nx < 3 or self.Ny < 3:
            raise ValueError(
                "Nx e Ny devem ser >= 3 (é preciso ao menos 1 nó interno em cada direção)."
            )
        self.dx = self.Lx / (self.Nx - 1)
        self.dy = self.Ly / (self.Ny - 1)
        self.x = np.linspace(0.0, self.Lx, self.Nx)
        self.y = np.linspace(0.0, self.Ly, self.Ny)
        # X, Y no formato (Ny, Nx): linha j -> y, coluna i -> x (convenção "imagem")
        self.X, self.Y = np.meshgrid(self.x, self.y)

    @property
    def n_nos(self) -> int:
        return self.Nx * self.Ny

    def idx(self, i: int, j: int) -> int:
        """Índice global (vetorizado) do nó (i,j)."""
        return j * self.Nx + i

    def para_campo(self, vetor: np.ndarray) -> np.ndarray:
        """Converte um vetor global (N,) de volta para o formato de campo (Ny, Nx)."""
        return vetor.reshape(self.Ny, self.Nx)

    def __str__(self):
        return (f"Malha {self.Nx} x {self.Ny} nós ({self.n_nos} graus de liberdade) | "
                f"dx={self.dx:.4g} m, dy={self.dy:.4g} m | Lx={self.Lx} m, Ly={self.Ly} m | "
                f"razão de aspecto da célula dx/dy={self.dx/self.dy:.3f}")
