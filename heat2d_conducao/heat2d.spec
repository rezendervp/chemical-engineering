# -*- mode: python ; coding: utf-8 -*-
"""
heat2d.spec
-----------
Especificação do PyInstaller para o simulador de Condução de Calor 2D.

Gera um executável ÚNICO (--onefile) e sem console (--windowed), incluindo:
    - os dados internos do CustomTkinter (temas, fontes) -- necessários,
      do contrário a GUI falha ao iniciar fora do ambiente de desenvolvimento
    - o ícone do aplicativo (assets/icon.ico)

Uso local (Windows, dentro de um venv com as dependências instaladas):
    pyinstaller heat2d.spec

O build automatizado (Windows) roda via GitHub Actions -- veja
.github/workflows/build.yml. Não é necessário rodar isso manualmente para
gerar releases; o workflow faz isso a cada tag "v*".
"""

import customtkinter
from pathlib import Path

block_cipher = None

# dados internos do customtkinter (temas .json, fontes .ttf) precisam ser
# empacotados manualmente -- o PyInstaller não os detecta sozinho.
ctk_path = Path(customtkinter.__file__).parent

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        (str(ctk_path), 'customtkinter'),
        ('assets', 'assets'),
    ],
    hiddenimports=[
        'PIL._tkinter_finder',
        'scipy.sparse.csgraph._validation',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Conducao2D_DEQ-UEM',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico',
)
