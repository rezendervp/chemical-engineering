#!/usr/bin/env python3
"""
main.py
-------
Ponto de entrada do simulador de condução de calor 2D.

Uso:
    python main.py

Para gerar o executável (.exe) via PyInstaller, veja README.md e
.github/workflows/build.yml (build automático via GitHub Actions).
"""

from app.gui.main_window import JanelaPrincipal


def main():
    app = JanelaPrincipal()
    app.mainloop()


if __name__ == "__main__":
    main()
