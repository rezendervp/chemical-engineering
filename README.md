<!-- BANNER / CAPA -->
<p align="center">
  <img src="https://raw.githubusercontent.com/rezendervp/chemical-engineering/main/cover.svg" alt="Chemical Engineering — Computational Notebooks" width="100%"/>
</p>

<h1 align="center">⚗️ Chemical Engineering — Computational Notebooks</h1>

<p align="center">
  <strong>Códigos de aula, notas de sala e simulações numéricas em Python / Jupyter</strong><br/>
  <em>por Prof. Ricardo V. P. Rezende</em>
</p>

<p align="center">
  <img alt="Jupyter Notebook" src="https://img.shields.io/badge/Jupyter-99.6%25-F37626?style=flat-square&logo=jupyter&logoColor=white"/>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img alt="License" src="https://img.shields.io/github/license/rezendervp/chemical-engineering?style=flat-square&color=4CAF50"/>
  <img alt="Commits" src="https://img.shields.io/github/commit-activity/t/rezendervp/chemical-engineering?style=flat-square&color=blueviolet"/>
  <img alt="Last Commit" src="https://img.shields.io/github/last-commit/rezendervp/chemical-engineering?style=flat-square"/>
</p>

---

## 📌 Sobre o repositório

Este repositório reúne notebooks Jupyter e scripts Python utilizados em disciplinas de **Engenharia Química**, cobrindo temas como transferência de calor, fluidodinâmica computacional, separações, sistemas dinâmicos caóticos e métodos numéricos avançados.

Todo o material é de uso **livre para fins acadêmicos**. Caso utilize ou modifique algum código como base para seu trabalho, por favor **cite o autor e o repositório**.

> Sugestões de melhoria ou abordagens mais elegantes são muito bem-vindas! Deixe sua mensagem na aba [Issues](https://github.com/rezendervp/chemical-engineering/issues).

---

## 📂 Estrutura do Repositório

```
chemical-engineering/
│
├── 🔥 Transferência de Calor
│   ├── Aleta.ipynb
│   ├── Condução_1D_BDF2.ipynb
│   ├── Condução_1D_Transiente.ipynb
│   └── Condução_RK4.ipynb
│
├── 🌊 Fluidodinâmica Computacional (CFD)
│   ├── Advection_wave.ipynb
│   ├── Cavity_Flow.ipynb
│   ├── Lid_Cavity_Flow.ipynb
│   ├── Lattice_Boltzmann_2D_Cilynder.ipynb
│   ├── SPH_2D_simulation.ipynb
│   └── Smoke.ipynb
│
├── 🔬 Separações e Operações Unitárias
│   └── mccabe_thiele/
│
├── 📊 Estatística e Planejamento Experimental
│   └── rsm_kriging/
│
├── 🔄 Sistemas Dinâmicos e EDOs
│   ├── ODE_system.ipynb
│   ├── Equação_Logistica.ipynb
│   └── Rabinovich–Fabrikant.ipynb
│
├── 🧮 Álgebra Linear e Métodos Numéricos
│   ├── Sistema_Linear_Minimização_de_Gradientes.ipynb
│   └── Funcional_da_matriz.ipynb
│
├── 📡 Análise de Dados / DMD
│   ├── DMD_Exemplo_02.ipynb
│   └── Time_Extrapolation_DMD_Exemplo_02.ipynb
│
└── ⚙️ Outros
    ├── Entropia_exemplo.ipynb
    ├── exemplo_de_função.ipynb
    └── app.py
```

---

## 📓 Notebooks — Links Diretos

### 🔥 Transferência de Calor

| Notebook | Descrição |
|---|---|
| [Aleta.ipynb](https://github.com/rezendervp/chemical-engineering/blob/main/Aleta.ipynb) | Simulação numérica de transferência de calor em aletas |
| [Condução\_1D\_BDF2.ipynb](https://github.com/rezendervp/chemical-engineering/blob/main/Condu%C3%A7%C3%A3o_1D_BDF2.ipynb) | Condução 1D com método BDF2 (integração implícita) |
| [Condução\_1D\_Transiente.ipynb](https://github.com/rezendervp/chemical-engineering/blob/main/Condu%C3%A7%C3%A3o_1D_Transiente.ipynb) | Condução transiente 1D |
| [Condução\_RK4.ipynb](https://github.com/rezendervp/chemical-engineering/blob/main/Condu%C3%A7%C3%A3o_RK4.ipynb) | Condução de calor integrada via Runge-Kutta 4ª ordem |

### 🌊 Fluidodinâmica Computacional (CFD)

| Notebook | Descrição |
|---|---|
| [Advection\_wave.ipynb](https://github.com/rezendervp/chemical-engineering/blob/main/Advection_wave.ipynb) | Equação de advecção — ondas de transporte |
| [Cavity\_Flow.ipynb](https://github.com/rezendervp/chemical-engineering/blob/main/Cavity_Flow.ipynb) | Escoamento em cavidade — equações de Navier-Stokes |
| [Lid\_Cavity\_Flow.ipynb](https://github.com/rezendervp/chemical-engineering/blob/main/Lid_Cavity_Flow.ipynb) | Lid-driven cavity flow |
| [Lattice\_Boltzmann\_2D\_Cilynder.ipynb](https://github.com/rezendervp/chemical-engineering/blob/main/Lattice_Boltzmann_2D_Cilynder.ipynb) | Método Lattice Boltzmann — escoamento ao redor de cilindro 2D |
| [SPH\_2D\_simulation.ipynb](https://github.com/rezendervp/chemical-engineering/blob/main/SPH_2D_simulation.ipynb) | Simulação SPH (Smoothed Particle Hydrodynamics) 2D |
| [Smoke.ipynb](https://github.com/rezendervp/chemical-engineering/blob/main/Smoke.ipynb) | Simulação de fumaça / escoamento difusivo |

### 🔬 Separações e Operações Unitárias

| Pasta / Arquivo | Descrição |
|---|---|
| [mccabe\_thiele/](https://github.com/rezendervp/chemical-engineering/tree/main/mccabe_thiele) | Método de McCabe-Thiele para dimensionamento de colunas de destilação |

### 📊 Estatística e Planejamento Experimental

| Pasta / Arquivo | Descrição |
|---|---|
| [rsm\_kriging/](https://github.com/rezendervp/chemical-engineering/tree/main/rsm_kriging) | Superfície de resposta (RSM) e krigagem |

### 🔄 Sistemas Dinâmicos e EDOs

| Notebook | Descrição |
|---|---|
| [ODE\_system.ipynb](https://github.com/rezendervp/chemical-engineering/blob/main/ODE_system.ipynb) | Sistemas de equações diferenciais ordinárias |
| [Equação\_Logistica.ipynb](https://github.com/rezendervp/chemical-engineering/blob/main/Equa%C3%A7%C3%A3o_Logistica.ipynb) | Equação logística — dinâmica populacional e caos |
| [Rabinovich–Fabrikant.ipynb](https://github.com/rezendervp/chemical-engineering/blob/main/Rabinovich%E2%80%93Fabrikant.ipynb) | Atrator caótico de Rabinovich–Fabrikant |

### 🧮 Álgebra Linear e Métodos Numéricos

| Notebook | Descrição |
|---|---|
| [Sistema\_Linear\_Minimização\_de\_Gradientes.ipynb](https://github.com/rezendervp/chemical-engineering/blob/main/Sistema_Linear_Minimiza%C3%A7%C3%A3o_de_Gradientes.ipynb) | Resolução de sistemas lineares via minimização de gradientes |
| [Funcional\_da\_matriz.ipynb](https://github.com/rezendervp/chemical-engineering/blob/main/Funcional_da_matriz.ipynb) | Funções matriciais e álgebra aplicada |

### 📡 Dynamic Mode Decomposition (DMD)

| Notebook | Descrição |
|---|---|
| [DMD\_Exemplo\_02.ipynb](https://github.com/rezendervp/chemical-engineering/blob/main/DMD_Exemplo_02.ipynb) | Decomposição de Modos Dinâmicos — exemplo 2 |
| [Time\_Extrapolation\_DMD\_Exemplo\_02.ipynb](https://github.com/rezendervp/chemical-engineering/blob/main/Time_Extrapolation_DMD_Exemplo_02.ipynb) | Extrapolação temporal via DMD |

### ⚙️ Outros

| Arquivo | Descrição |
|---|---|
| [Entropia\_exemplo.ipynb](https://github.com/rezendervp/chemical-engineering/blob/main/Entropia_exemplo.ipynb) | Cálculo de entropia — exemplo termodinâmico |
| [exemplo\_de\_função.ipynb](https://github.com/rezendervp/chemical-engineering/blob/main/exemplo_de_fun%C3%A7%C3%A3o.ipynb) | Exemplo introdutório de funções Python |
| [app.py](https://github.com/rezendervp/chemical-engineering/blob/main/app.py) | Aplicação Python auxiliar |

---

## 🚀 Como usar

### Pré-requisitos

```bash
pip install -r requirements.txt
```

### Executar localmente

```bash
git clone https://github.com/rezendervp/chemical-engineering.git
cd chemical-engineering
jupyter notebook
```

### Executar no navegador (sem instalação)

Clique no badge abaixo para abrir o repositório diretamente no **GitHub Codespaces** (ambiente já configurado via `.devcontainer`):

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/rezendervp/chemical-engineering)

---

## 🛠️ Tecnologias utilizadas

- **Python 3.x**
- **Jupyter Notebook / JupyterLab**
- **NumPy · SciPy · Matplotlib**
- **Pandas**
- Ver lista completa em [`requirements.txt`](https://github.com/rezendervp/chemical-engineering/blob/main/requirements.txt)

---

## 📜 Licença

Este projeto está licenciado sob a **GNU General Public License v3.0**.  
Veja o arquivo [LICENSE](https://github.com/rezendervp/chemical-engineering/blob/main/LICENSE) para mais detalhes.

---

## ✉️ Contato & Citação

**Prof. Ricardo V. P. Rezende**  
Universidade Estadual de Maringá (UEM) — Maringá, PR, Brasil

Se você utilizar qualquer código deste repositório como base para publicações ou trabalhos acadêmicos, por favor cite:

```
Rezende, R.V.P. (2022). chemical-engineering: Sample codes for Lectures and Classroom notes.
GitHub. https://github.com/rezendervp/chemical-engineering
```

---

<p align="center">
  <sub>© 2022–2026 Prof. Ricardo V. P. Rezende · Feito com ☕ e Python</sub>
</p>
