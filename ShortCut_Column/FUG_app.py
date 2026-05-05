# -*- coding: utf-8 -*-
"""
============================================================
  SIMULADOR DE COLUNA DE ATALHO — MÉTODO FUGK
  Fenske · Underwood · Gilliland · Kirkbride

  Streamlit App  |  Prof. Dr. Ricardo V. P. Rezende
  DEQ/CTC/UEM — Operações Unitárias II
============================================================
  Flash isotérmico (Rachford-Rice) + Antoine + Raoult/NRTL
  Banco de dados: hidrocarbonetos, álcoois, ésteres, éteres
============================================================
"""

import numpy as np
import scipy.optimize as opt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════
#  BANCO DE DADOS DE COMPONENTES
#  Antoine: log10(Psat/mmHg) = A - B/(C + T[°C])  — parâm. NIST/Perry's
# ════════════════════════════════════════════════════════════════════════════

COMPONENT_DB = {
    # ── HIDROCARBONETOS ──────────────────────────────────────────────────
    "Metano":          dict(A=6.82973, B=405.42,  C=267.777, Tc=190.6, Pc=46.1, Mw=16.04,  group="Hidrocarboneto"),
    "Etano":           dict(A=6.90648, B=663.70,  C=256.470, Tc=305.4, Pc=48.8, Mw=30.07,  group="Hidrocarboneto"),
    "Propano":         dict(A=6.82973, B=813.20,  C=248.000, Tc=369.8, Pc=42.5, Mw=44.10,  group="Hidrocarboneto"),
    "n-Butano":        dict(A=6.83029, B=945.90,  C=239.711, Tc=425.1, Pc=38.0, Mw=58.12,  group="Hidrocarboneto"),
    "n-Pentano":       dict(A=6.87601, B=1064.63, C=232.000, Tc=469.7, Pc=33.7, Mw=72.15,  group="Hidrocarboneto"),
    "n-Hexano":        dict(A=6.87601, B=1171.17, C=224.408, Tc=507.6, Pc=30.1, Mw=86.18,  group="Hidrocarboneto"),
    "n-Heptano":       dict(A=6.90253, B=1267.83, C=216.900, Tc=540.3, Pc=27.4, Mw=100.20, group="Hidrocarboneto"),
    "n-Octano":        dict(A=6.91868, B=1351.99, C=209.155, Tc=568.8, Pc=24.9, Mw=114.23, group="Hidrocarboneto"),
    "Benzeno":         dict(A=6.90565, B=1211.03, C=220.790, Tc=562.2, Pc=48.9, Mw=78.11,  group="Hidrocarboneto"),
    "Tolueno":         dict(A=6.95464, B=1344.80, C=219.482, Tc=591.8, Pc=41.1, Mw=92.14,  group="Hidrocarboneto"),
    "Ciclohexano":     dict(A=6.84498, B=1203.53, C=222.863, Tc=553.5, Pc=40.7, Mw=84.16,  group="Hidrocarboneto"),
    "Isobutano":       dict(A=6.82645, B=913.37,  C=242.900, Tc=408.2, Pc=36.5, Mw=58.12,  group="Hidrocarboneto"),
    "Isopentano":      dict(A=6.78967, B=1020.01, C=233.205, Tc=460.4, Pc=33.4, Mw=72.15,  group="Hidrocarboneto"),
    # ── ÁLCOOIS ──────────────────────────────────────────────────────────
    "Metanol":         dict(A=7.87863, B=1473.11, C=230.000, Tc=512.6, Pc=80.9, Mw=32.04,  group="Álcool"),
    "Etanol":          dict(A=8.11220, B=1592.86, C=226.184, Tc=513.9, Pc=61.4, Mw=46.07,  group="Álcool"),
    "1-Propanol":      dict(A=7.74416, B=1437.69, C=198.463, Tc=536.8, Pc=51.7, Mw=60.10,  group="Álcool"),
    "2-Propanol":      dict(A=8.11778, B=1580.92, C=219.610, Tc=508.3, Pc=47.6, Mw=60.10,  group="Álcool"),
    "1-Butanol":       dict(A=7.47680, B=1362.39, C=178.770, Tc=563.1, Pc=44.2, Mw=74.12,  group="Álcool"),
    "2-Butanol":       dict(A=7.62231, B=1417.90, C=190.980, Tc=536.1, Pc=41.8, Mw=74.12,  group="Álcool"),
    # ── ÁGUA ─────────────────────────────────────────────────────────────
    "Água":            dict(A=8.07131, B=1730.63, C=233.426, Tc=647.1, Pc=220.6,Mw=18.02,  group="Água"),
    # ── ÉSTERES ──────────────────────────────────────────────────────────
    "Acetato de etila":dict(A=7.09808, B=1238.71, C=217.000, Tc=523.3, Pc=38.3, Mw=88.11,  group="Éster"),
    "Acetato de metila":dict(A=7.06524,B=1157.63, C=219.726, Tc=506.9, Pc=46.9, Mw=74.08,  group="Éster"),
    "Acetato de butila":dict(A=7.07691,B=1351.99, C=209.000, Tc=575.4, Pc=30.6, Mw=116.16, group="Éster"),
    "Formiato de etila":dict(A=7.11002,B=1159.75, C=218.000, Tc=508.5, Pc=47.4, Mw=74.08,  group="Éster"),
    # ── ÉTERES ───────────────────────────────────────────────────────────
    "Éter dietílico":  dict(A=6.92374, B=1064.07, C=228.800, Tc=466.7, Pc=36.4, Mw=74.12,  group="Éter"),
    "MTBE":            dict(A=6.87776, B=1102.15, C=224.370, Tc=497.1, Pc=34.3, Mw=88.15,  group="Éter"),
    "THF":             dict(A=6.99515, B=1202.29, C=226.254, Tc=540.1, Pc=51.9, Mw=72.11,  group="Éter"),
    # ── CETONAS ──────────────────────────────────────────────────────────
    "Acetona":         dict(A=7.11714, B=1210.59, C=229.664, Tc=508.1, Pc=47.0, Mw=58.08,  group="Cetona"),
    "MEK (2-Butanona)":dict(A=7.06356, B=1261.34, C=221.969, Tc=535.5, Pc=41.5, Mw=72.11,  group="Cetona"),
    # ── ALDEÍDOS / ÁCIDOS ────────────────────────────────────────────────
    "Acetaldeído":     dict(A=7.05534, B=1070.47, C=236.000, Tc=461.0, Pc=55.7, Mw=44.05,  group="Aldeído"),
    "Ác. acético":     dict(A=7.38782, B=1533.31, C=222.309, Tc=592.7, Pc=57.9, Mw=60.05,  group="Ácido"),
    # ── AROMÁTICOS SUBSTITUÍDOS ──────────────────────────────────────────
    "o-Xileno":        dict(A=6.99052, B=1474.68, C=213.686, Tc=630.3, Pc=37.3, Mw=106.17, group="Hidrocarboneto"),
    "m-Xileno":        dict(A=6.99052, B=1462.27, C=215.105, Tc=617.1, Pc=35.4, Mw=106.17, group="Hidrocarboneto"),
    "p-Xileno":        dict(A=6.99052, B=1453.43, C=215.307, Tc=616.2, Pc=35.1, Mw=106.17, group="Hidrocarboneto"),
    "Estireno":        dict(A=6.92409, B=1420.00, C=206.000, Tc=648.0, Pc=38.4, Mw=104.15, group="Hidrocarboneto"),
    # ── GASES LEVES ──────────────────────────────────────────────────────
    "Nitrogênio":      dict(A=6.49457, B=255.68,  C=266.550, Tc=126.2, Pc=33.9, Mw=28.01,  group="Gás leve"),
    "CO2":             dict(A=6.81228, B=1301.679,C=3.494,   Tc=304.2, Pc=73.8, Mw=44.01,  group="Gás leve"),
}

# ════════════════════════════════════════════════════════════════════════════
#  PARÂMETROS NRTL  (tau_ij, tau_ji, alpha_ij)  — pares polares
#  τ_ij = a_ij + b_ij/T[K]  ;  aqui usamos τ médio (b_ij/T_ref)
#  Ref: Gmehling et al. (2012), DECHEMA
# ════════════════════════════════════════════════════════════════════════════

NRTL_DB = {
    # (comp_i, comp_j): (tau_ij, tau_ji, alpha)
    ("Etanol",   "Água"):          ( 1.8290,  0.7219, 0.2994),
    ("Metanol",  "Água"):          ( 1.5070,  0.5260, 0.2945),
    ("1-Propanol","Água"):         ( 2.7760,  0.6036, 0.2720),
    ("2-Propanol","Água"):         ( 2.3560,  0.7270, 0.2790),
    ("1-Butanol","Água"):          ( 3.3430,  0.8640, 0.2480),
    ("Acetona",  "Água"):          ( 2.0890,  1.5280, 0.5651),
    ("Acetona",  "Metanol"):       ( 0.6910,  0.4320, 0.3004),
    ("Acetato de etila","Etanol"): ( 0.7935,  0.3177, 0.2983),
    ("Éter dietílico","Etanol"):   ( 1.0970,  0.3860, 0.3000),
    ("Etanol",   "Benzeno"):       ( 2.7800,  1.2960, 0.4710),
    ("Metanol",  "Benzeno"):       ( 2.9300,  1.4640, 0.4670),
    ("Água",     "Ác. acético"):   ( 0.9740,  1.4130, 0.2991),
}

def get_nrtl(ci, cj):
    """Retorna (tau_ij, tau_ji, alpha) ou None se par não disponível."""
    key1 = (ci, cj)
    key2 = (cj, ci)
    if key1 in NRTL_DB:
        t12, t21, a = NRTL_DB[key1]
        return t12, t21, a
    elif key2 in NRTL_DB:
        t21, t12, a = NRTL_DB[key2]
        return t12, t21, a
    return None

# ════════════════════════════════════════════════════════════════════════════
#  TERMODINÂMICA
# ════════════════════════════════════════════════════════════════════════════

def psat_mmhg(comp, T_C):
    """Pressão de vapor [mmHg] via Antoine. T em °C."""
    d = COMPONENT_DB[comp]
    return 10.0 ** (d["A"] - d["B"] / (d["C"] + T_C))

def psat_atm(comp, T_C):
    return psat_mmhg(comp, T_C) / 760.0

def Ki_raoult(comp, T_C, P_atm):
    """K_i = Psat_i / P  (Raoult / termodinâmica ideal)."""
    return psat_atm(comp, T_C) / P_atm

def gamma_nrtl_binary(x1, tau12, tau21, alpha12):
    """Coeficientes de atividade NRTL para sistema binário."""
    alpha21 = alpha12
    G12 = np.exp(-alpha12 * tau12)
    G21 = np.exp(-alpha21 * tau21)
    x2 = 1.0 - x1
    # ln gamma1
    term1 = x2**2 * (tau21 * (G21/(x1+x2*G21))**2 +
                     tau12*G12/(x2+x1*G12)**2)
    # ln gamma2
    term2 = x1**2 * (tau12 * (G12/(x2+x1*G12))**2 +
                     tau21*G21/(x1+x2*G21)**2)
    g1 = np.exp(np.clip(term1, -20, 20))
    g2 = np.exp(np.clip(term2, -20, 20))
    return g1, g2

def gamma_multicomp_nrtl(x, comps):
    """
    Coeficientes de atividade NRTL multicomponente.
    Usa extensão multicomponente de Renon-Prausnitz.
    Para pares sem parâmetros → gamma = 1 (ideal).
    """
    n = len(comps)
    if n == 0:
        return np.ones(0)
    # Montar matrizes tau e G
    tau = np.zeros((n, n))
    alpha = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                par = get_nrtl(comps[i], comps[j])
                if par is not None:
                    tau[i, j] = par[0]
                    alpha[i, j] = par[2]
    G = np.exp(-alpha * tau)
    # ln gamma multicomponente
    ln_gamma = np.zeros(n)
    for i in range(n):
        num1 = np.sum(x * tau[:, i] * G[:, i])
        den1 = np.sum(x * G[:, i])
        sum2 = 0.0
        for j in range(n):
            num2j = x[j] * G[i, j]
            den2j = np.sum(x * G[:, j])
            tau_avg = np.sum(x * tau[:, j] * G[:, j]) / den2j if den2j > 1e-15 else 0.0
            sum2 += (num2j / den2j) * (tau[i, j] - tau_avg)
        ln_gamma[i] = num1/den1 + sum2 if den1 > 1e-15 else 0.0
    return np.exp(np.clip(ln_gamma, -10, 10))

def Ki_nrtl(comps, x_liq, T_C, P_atm):
    """K_i = gamma_i * Psat_i / P  (Raoult modificado + NRTL)."""
    gamma = gamma_multicomp_nrtl(np.array(x_liq), comps)
    K = np.array([gamma[i] * psat_atm(comps[i], T_C) / P_atm for i in range(len(comps))])
    return K

# ════════════════════════════════════════════════════════════════════════════
#  ALGORITMO FLASH ISOTÉRMICO (Rachford-Rice)
# ════════════════════════════════════════════════════════════════════════════

def rachford_rice(psi, z, K):
    """Função de Rachford-Rice: Σ z_i(K_i-1)/(1+ψ(K_i-1)) = 0."""
    return np.sum(z * (K - 1.0) / (1.0 + psi * (K - 1.0)))

def flash_isotermico(comps, z, T_C, P_atm, thermo="Raoult", max_iter=200, tol=1e-10):
    """
    Flash isotérmico com temperatura e pressão fixas.
    Retorna: psi (fração vaporizada), x (liq), y (vap), K, T_C
    thermo: 'Raoult' ou 'NRTL'
    """
    z = np.array(z, dtype=float)
    n = len(comps)

    # K iniciais via Raoult
    K = np.array([Ki_raoult(c, T_C, P_atm) for c in comps])

    for iteration in range(max_iter):
        K_old = K.copy()

        # Verificar se há fase única
        if np.all(K * z <= 1.0) and rachford_rice(0.0, z, K) < 0:
            # Líquido puro
            return 0.0, z.copy(), z.copy(), K, T_C
        if np.all(K * z >= 1.0) and rachford_rice(1.0, z, K) > 0:
            # Vapor puro
            return 1.0, z.copy(), z.copy(), K, T_C

        # Limites para psi
        psi_min = 1.0 / (1.0 - np.max(K)) + 1e-8
        psi_max = 1.0 / (1.0 - np.min(K)) - 1e-8
        psi_min = max(psi_min, 0.0)
        psi_max = min(psi_max, 1.0)
        if psi_min >= psi_max:
            psi_min, psi_max = 0.0, 1.0

        try:
            psi = opt.brentq(rachford_rice, psi_min + 1e-10, psi_max - 1e-10,
                             args=(z, K), xtol=1e-14, maxiter=500)
        except Exception:
            psi = 0.5

        x = z / (1.0 + psi * (K - 1.0))
        x = np.clip(x, 1e-15, 1.0)
        x /= x.sum()
        y = K * x
        y = np.clip(y, 1e-15, 1.0)
        y /= y.sum()

        # Atualizar K
        if thermo == "NRTL":
            K_new = Ki_nrtl(comps, x, T_C, P_atm)
        else:
            K_new = np.array([Ki_raoult(c, T_C, P_atm) for c in comps])

        err = np.max(np.abs(K_new - K))
        K = K_new
        if err < tol and iteration > 2:
            break

    x = z / (1.0 + psi * (K - 1.0))
    x = np.clip(x, 1e-15, 1.0); x /= x.sum()
    y = K * x
    y = np.clip(y, 1e-15, 1.0); y /= y.sum()
    return psi, x, y, K, T_C

def find_bubble_point(comps, z, P_atm, thermo="Raoult", T_guess=None):
    """Encontra temperatura de bolha para dado P e composição."""
    z = np.array(z, dtype=float)
    if T_guess is None:
        # estimativa: média ponderada dos pontos de ebulição normais
        T_guess = sum(z[i] * (-COMPONENT_DB[c]["C"] +
                     COMPONENT_DB[c]["B"] / (COMPONENT_DB[c]["A"] - np.log10(P_atm * 760.0)))
                     for i, c in enumerate(comps))
    def bubble_eq(T):
        if thermo == "NRTL":
            K = Ki_nrtl(comps, z, T, P_atm)
        else:
            K = np.array([Ki_raoult(c, T, P_atm) for c in comps])
        return np.sum(K * z) - 1.0

    try:
        T_bub = opt.brentq(bubble_eq, -80, 350, xtol=1e-8)
    except Exception:
        T_bub = T_guess
    return T_bub

def find_dew_point(comps, z, P_atm, thermo="Raoult", T_guess=None):
    """Encontra temperatura de orvalho para dado P e composição."""
    z = np.array(z, dtype=float)
    if T_guess is None:
        T_guess = find_bubble_point(comps, z, P_atm, thermo) + 10
    def dew_eq(T):
        if thermo == "NRTL":
            K = Ki_nrtl(comps, z, T, P_atm)
        else:
            K = np.array([Ki_raoult(c, T, P_atm) for c in comps])
        return np.sum(z / K) - 1.0

    try:
        T_dew = opt.brentq(dew_eq, -80, 400, xtol=1e-8)
    except Exception:
        T_dew = T_guess
    return T_dew

def T_from_q(comps, z, P_atm, q, thermo="Raoult"):
    """Calcula temperatura de alimentação a partir de q."""
    T_bub = find_bubble_point(comps, z, P_atm, thermo)
    T_dew = find_dew_point(comps, z, P_atm, thermo)
    if q >= 1.0:
        return T_bub  # líquido saturado → T = T_bolha
    elif q <= 0.0:
        return T_dew  # vapor saturado → T = T_orvalho
    else:
        return T_bub + (1.0 - q) * (T_dew - T_bub)

# ════════════════════════════════════════════════════════════════════════════
#  MÉTODO FUGK
# ════════════════════════════════════════════════════════════════════════════

def fenske(alpha, lk, hk, rec_LK_D, rec_HK_B):
    """Equação de Fenske → N_min."""
    S_LK    = rec_LK_D / (1.0 - rec_LK_D)
    bHK_dHK = rec_HK_B / (1.0 - rec_HK_B)
    alpha_ratio = alpha[lk] / alpha[hk]
    N_min = np.log(S_LK * bHK_dHK) / np.log(alpha_ratio)
    return N_min, S_LK, bHK_dHK

def fenske_distribution(alpha, z, F, lk, N_min, S_LK):
    """Distribuição generalizada de Fenske."""
    S_all = S_LK * (alpha / alpha[lk]) ** N_min
    d_all = F * z * S_all / (1.0 + S_all)
    b_all = F * z - d_all
    D, B  = d_all.sum(), b_all.sum()
    x_D   = d_all / D if D > 0 else d_all
    x_B   = b_all / B if B > 0 else b_all
    return S_all, d_all, b_all, D, B, x_D, x_B

def underwood(alpha, z, q, lk, hk, d_all):
    """Equação de Underwood → theta, V_min, R_min."""
    def uw_eq(theta):
        return np.sum(alpha * z / (alpha - theta)) - (1.0 - q)
    # Raiz entre alpha_HK e alpha_LK
    a_lo, a_hi = alpha[hk] + 1e-9, alpha[lk] - 1e-9
    try:
        theta = opt.brentq(uw_eq, a_lo, a_hi, xtol=1e-14, maxiter=500)
    except Exception:
        theta = (alpha[lk] + alpha[hk]) / 2.0
    D = d_all.sum()
    V_min = np.sum(alpha * d_all / (alpha - theta))
    R_min = V_min / D - 1.0 if D > 0 else 0.0
    return theta, V_min, max(R_min, 0.0)

def molokanov(X):
    return 1.0 - np.exp(
        (1.0 + 54.4 * X) / (11.0 + 117.2 * X) * (X - 1.0) / np.sqrt(X + 1e-15)
    )

def gilliland(N_min, R_min, R_factor=1.5):
    """Correlação de Gilliland (Molokanov 1972) → N_real."""
    R_op   = R_factor * R_min
    X_gill = (R_op - R_min) / (R_op + 1.0)
    Y_gill = molokanov(X_gill)
    N_real = (N_min + Y_gill) / (1.0 - Y_gill)
    return R_op, X_gill, Y_gill, N_real

def kirkbride(z, lk, hk, B, D, x_B, x_D, N_total):
    """Localização do prato de alimentação (Kirkbride)."""
    ratio = (B / D) * (z[hk] / z[lk]) * (x_B[lk] / x_D[hk]) ** 2
    Nr_Ns = 10.0 ** (0.206 * np.log10(ratio))
    Nr    = max(1, int(np.ceil(N_total * Nr_Ns / (1.0 + Nr_Ns))))
    Ns    = max(1, N_total - Nr)
    N_feed = Nr + 1
    return Nr_Ns, Nr, Ns, N_feed

# ════════════════════════════════════════════════════════════════════════════
#  GRÁFICOS
# ════════════════════════════════════════════════════════════════════════════

AZUL   = "#1a3a5c"
VERDE  = "#2ecc71"
VERM   = "#e74c3c"
CINZA  = "#95a5a6"

def plot_gilliland(X_gill, Y_gill, N_min, N_real, R_min, R_op):
    fig, ax = plt.subplots(figsize=(7, 5))
    X_arr = np.linspace(0.002, 0.99, 800)
    Y_arr = molokanov(X_arr)
    ax.plot(X_arr, Y_arr, color=AZUL, lw=2.2, label="Molokanov (1972)")
    ax.plot(X_gill, Y_gill, "o", ms=10, color=VERM, zorder=5,
            label=f"Este caso  X={X_gill:.3f}, Y={Y_gill:.3f}")
    ax.plot([X_gill]*2, [0, Y_gill], "--", color=VERM, lw=1.1, alpha=0.6)
    ax.plot([0, X_gill], [Y_gill]*2, "--", color=VERM, lw=1.1, alpha=0.6)
    ax.set_xlabel(r"$X = (R - R_{min})\;/\;(R + 1)$", fontsize=11)
    ax.set_ylabel(r"$Y = (N - N_{min})\;/\;(N + 1)$",  fontsize=11)
    ax.set_title("Diagrama de Gilliland", fontsize=12, fontweight="bold", color=AZUL)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig

def plot_underwood(alpha, z, theta, lk, hk):
    fig, ax = plt.subplots(figsize=(6, 4))
    th_range = np.linspace(alpha.min() * 0.7, alpha.max() * 1.1, 6000)
    soma_vals = []
    for th in th_range:
        if np.any(np.abs(alpha - th) < 0.04):
            soma_vals.append(np.nan)
        else:
            v = np.sum(alpha * z / (alpha - th))
            soma_vals.append(v if abs(v) < 12 else np.nan)
    ax.plot(th_range, soma_vals, color=AZUL, lw=1.8)
    ax.axhline(0, color="black", lw=1.1)
    ax.axvline(theta, color=VERM, lw=2, ls="--", label=rf"$\theta={theta:.4f}$")
    colors_comp = plt.cm.tab10(np.linspace(0, 0.9, len(alpha)))
    for i, (a, col) in enumerate(zip(alpha, colors_comp)):
        ax.axvline(a, color=col, lw=1.1, ls=":", alpha=0.8)
    ax.set_xlim(alpha.min() * 0.6, alpha.max() * 1.15)
    ax.set_ylim(-12, 12)
    ax.set_xlabel(r"$\theta$", fontsize=12)
    ax.set_ylabel(r"$\sum \alpha_i z_i/(\alpha_i - \theta)$", fontsize=10)
    ax.set_title(r"Raiz de Underwood — $\theta$", fontsize=11, fontweight="bold", color=AZUL)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def plot_composicoes(components, z, x_D, x_B):
    fig, ax = plt.subplots(figsize=(9, 4))
    x_idx = np.arange(len(components))
    w = 0.26
    labels = [c[:12] for c in components]
    for k, (vals, lbl, fc) in enumerate(zip(
        [z, x_D, x_B],
        ["Alimentação (z)", "Destilado (xD)", "Fundo (xB)"],
        ["#3498db", "#2ecc71", "#e74c3c"]
    )):
        bars = ax.bar(x_idx + (k-1)*w, vals, width=w, label=lbl,
                      color=fc, alpha=0.85, edgecolor="white", lw=1.2)
        for bar, v in zip(bars, vals):
            if v > 0.005:
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.013,
                        f"{v:.3f}", ha="center", fontsize=7.5,
                        fontweight="bold", color=AZUL)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax.set_ylabel("Fração molar", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title("Composições — Alimentação, Destilado e Fundo",
                 fontsize=11, fontweight="bold", color=AZUL)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig

def plot_K_volatilidades(components, K, alpha):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(components)))
    labels = [c[:14] for c in components]
    axes[0].bar(labels, K, color=colors, edgecolor="white", lw=1.2, alpha=0.88)
    axes[0].axhline(1.0, color="black", lw=1.2, ls="--", alpha=0.6)
    axes[0].set_ylabel("K-valor", fontsize=11)
    axes[0].set_title("K-valores (equilíbrio L-V)", fontsize=11, fontweight="bold", color=AZUL)
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(labels, alpha, color=colors, edgecolor="white", lw=1.2, alpha=0.88)
    axes[1].axhline(1.0, color=VERM, lw=1.5, ls="--", alpha=0.7, label="α=1 (HK)")
    axes[1].set_ylabel("Volatilidade relativa α", fontsize=11)
    axes[1].set_title("Volatilidades Relativas", fontsize=11, fontweight="bold", color=AZUL)
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig

# ════════════════════════════════════════════════════════════════════════════
#  INTERFACE STREAMLIT
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Simulador FUGK — Destilação Multicomponente",
    page_icon="⚗️",
    layout="wide",
)

st.markdown("""
<style>
    .main-title {font-size:2rem; font-weight:700; color:#1a3a5c; margin-bottom:0;}
    .sub-title  {font-size:1rem; color:#555; margin-top:0;}
    .result-box {background:#f0f4f8; border-left:4px solid #1a3a5c;
                 padding:0.6rem 1rem; border-radius:6px; margin:4px 0;}
    .warn-box   {background:#fff3cd; border-left:4px solid #f39c12;
                 padding:0.5rem 1rem; border-radius:6px;}
    .section-hdr{color:#1a3a5c; font-weight:700; font-size:1.05rem; margin-top:1rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">⚗️ Simulador de Coluna de Atalho — Método FUGK</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Fenske · Underwood · Gilliland · Kirkbride &nbsp;|&nbsp; '
            'Flash Rachford-Rice · Antoine · Raoult/NRTL &nbsp;|&nbsp; '
            'DEQ/CTC/UEM — Prof. Dr. Ricardo V. P. Rezende</p>', unsafe_allow_html=True)
st.markdown("---")

# ── BARRA LATERAL ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configurações")

    # Número de componentes
    n_comp = st.slider("Número de componentes", min_value=2, max_value=8, value=4, step=1)

    # Termodinâmica
    thermo = st.selectbox("Modelo termodinâmico",
                          ["Raoult (ideal)", "NRTL (não-ideal, pares disponíveis)"])
    thermo_key = "NRTL" if "NRTL" in thermo else "Raoult"

    # Condições de operação
    st.markdown("---")
    st.subheader("Condições Globais")
    F_feed = st.number_input("Vazão de alimentação F [kmol/h]", 1.0, 10000.0, 100.0, step=10.0)
    P_atm  = st.number_input("Pressão de operação P [atm]", 0.1, 50.0, 1.0, step=0.5)
    q_feed = st.number_input("Parâmetro de qualidade q", -1.0, 2.0, 1.0, step=0.1,
                             help="q=1: liq. saturado | q=0: vap. saturado | 0<q<1: misto")
    R_factor = st.number_input("Fator de refluxo  R = fator × R_min", 1.1, 5.0, 1.5, step=0.1)

# ── SELEÇÃO DOS COMPONENTES ────────────────────────────────────────────────
all_comps = list(COMPONENT_DB.keys())
groups    = sorted(set(v["group"] for v in COMPONENT_DB.values()))

st.markdown('<p class="section-hdr">1 · Escolha dos Componentes e Composições da Alimentação</p>',
            unsafe_allow_html=True)

comp_cols = st.columns(min(n_comp, 4))
selected  = []
z_vals    = []

for i in range(n_comp):
    col = comp_cols[i % len(comp_cols)]
    with col:
        # Defaults didáticos (exemplo do PDF)
        defaults = ["n-Propano", "n-Butano", "n-Pentano", "n-Hexano",
                    "Benzeno", "Tolueno", "Metanol", "Etanol"]
        default_comp = defaults[i] if i < len(defaults) and defaults[i] in all_comps else all_comps[i]
        comp = st.selectbox(f"Componente {i+1}", all_comps,
                            index=all_comps.index(default_comp), key=f"comp_{i}")
        zi = st.number_input(f"z_{i+1} (fração molar)", 0.0, 1.0,
                             [0.10, 0.30, 0.35, 0.25, 0.10, 0.10, 0.10, 0.10][i] if i < 8 else 0.10,
                             step=0.01, key=f"z_{i}")
        selected.append(comp)
        z_vals.append(zi)

# Verificar componentes duplicados
if len(set(selected)) < len(selected):
    st.error("⚠️ Componentes repetidos! Selecione componentes diferentes.")
    st.stop()

# Normalizar z
z_arr = np.array(z_vals, dtype=float)
z_sum = z_arr.sum()
if abs(z_sum - 1.0) > 1e-6:
    st.warning(f"Σzi = {z_sum:.4f} ≠ 1. As frações serão normalizadas automaticamente.")
z_arr /= z_sum

st.markdown("---")
st.markdown('<p class="section-hdr">2 · Especificação dos Componentes-Chave e Recuperações</p>',
            unsafe_allow_html=True)

col_lk, col_hk, col_rec1, col_rec2 = st.columns(4)
with col_lk:
    lk_name = st.selectbox("Componente-Chave Leve (LK)", selected, index=1 if len(selected) > 1 else 0)
    lk = selected.index(lk_name)
with col_hk:
    hk_options = [c for c in selected if c != lk_name]
    hk_default = hk_options[0] if hk_options else selected[-1]
    hk_name = st.selectbox("Componente-Chave Pesado (HK)", hk_options,
                           index=0)
    hk = selected.index(hk_name)
with col_rec1:
    rec_LK_D = st.slider("Recuperação LK no destilado (%)", 50, 99, 97, step=1) / 100.0
with col_rec2:
    rec_HK_B = st.slider("Recuperação HK no fundo (%)", 50, 99, 98, step=1) / 100.0

if lk == hk:
    st.error("LK e HK não podem ser o mesmo componente!")
    st.stop()

# Verificar ordem de volatilidade: LK deve ter maior Psat que HK
T_ref = 100.0  # °C de referência
ps_lk = psat_mmhg(selected[lk], T_ref)
ps_hk = psat_mmhg(selected[hk], T_ref)
if ps_lk < ps_hk:
    st.error(f"⚠️ {selected[lk]} é MENOS volátil que {selected[hk]} a {T_ref}°C. "
             "Verifique a designação LK/HK.")
    st.stop()

# ── BOTÃO DE CÁLCULO ───────────────────────────────────────────────────────
st.markdown("---")
calc_btn = st.button("🚀 Calcular", type="primary", use_container_width=True)

if calc_btn:
    with st.spinner("Executando Flash + Método FUGK..."):

        # ── 1. Flash para obter T e K-valores ─────────────────────────────
        # Estimar temperatura média da coluna via ponto de bolha da alimentação
        T_feed = T_from_q(selected, z_arr, P_atm, q_feed, thermo_key)

        # Flash isotérmico na alimentação
        psi_F, x_F, y_F, K_feed, T_feed = flash_isotermico(
            selected, z_arr, T_feed, P_atm, thermo=thermo_key)

        # ── 2. Volatilidades relativas (referência = HK) ──────────────────
        alpha_arr = K_feed / K_feed[hk]

        # Verificar se alpha_LK > alpha_HK
        if alpha_arr[lk] <= alpha_arr[hk]:
            st.error("Volatilidade do LK ≤ HK após flash. Revise a especificação.")
            st.stop()

        # ── 3. Fenske ─────────────────────────────────────────────────────
        N_min, S_LK, bHK_dHK = fenske(alpha_arr, lk, hk, rec_LK_D, rec_HK_B)
        S_all, d_all, b_all, D, B, x_D, x_B = fenske_distribution(
            alpha_arr, z_arr, F_feed, lk, N_min, S_LK)

        # ── 4. Underwood ──────────────────────────────────────────────────
        theta, V_min, R_min = underwood(alpha_arr, z_arr, q_feed, lk, hk, d_all)

        # ── 5. Gilliland ──────────────────────────────────────────────────
        R_op, X_gill, Y_gill, N_real = gilliland(N_min, R_min, R_factor)
        N_total = max(2, int(np.ceil(N_real)))

        # ── 6. Kirkbride ──────────────────────────────────────────────────
        try:
            Nr_Ns, Nr, Ns, N_feed = kirkbride(z_arr, lk, hk, B, D, x_B, x_D, N_total)
        except Exception:
            Nr_Ns, Nr, Ns, N_feed = 1.0, N_total//2, N_total//2, N_total//2 + 1

    # ══ EXIBIÇÃO DOS RESULTADOS ════════════════════════════════════════════
    st.success("✅ Cálculo concluído com sucesso!")

    # Linha de resumo rápido
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("N_min (Fenske)",  f"{N_min:.2f}")
    col2.metric("R_min (Underwood)", f"{R_min:.4f}")
    col3.metric("R_op", f"{R_op:.4f}")
    col4.metric("N teórico total", f"{N_total}")
    col5.metric("Prato aliment. N_F", f"{N_feed} (do topo)")

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Resultados FUGK", "🔬 Flash & Termodinâmica",
                                       "📈 Gráficos", "📋 Tabelas Detalhadas"])

    # ── TAB 1: Resultados FUGK ─────────────────────────────────────────────
    # ── TAB 1: Resultados FUGK ─────────────────────────────────────────────
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### ① Fenske — N_min")
            # Substituir o markdown colorido por st.info ou st.success
            st.info(f"""
            **S_LK** = d_LK/b_LK = {S_LK:.4f}  
            **b_HK/d_HK** = {bHK_dHK:.4f}  
            **α_LK/α_HK** = {alpha_arr[lk]/alpha_arr[hk]:.4f}  
            **N_min** = **{N_min:.4f}** estágios
            """)
    
            st.markdown("#### ② Underwood — R_min")
            st.info(f"""
            **θ** = {theta:.6f} (entre α_HK={alpha_arr[hk]:.4f} e α_LK={alpha_arr[lk]:.4f})  
            **V_min** = {V_min:.3f} kmol/h  
            **R_min** = **{R_min:.4f}**
            """)
    
        with c2:
            st.markdown("#### ③ Gilliland — N real")
            st.success(f"""
            **R_op** = {R_factor} × R_min = {R_op:.4f}  
            **X** = {X_gill:.4f} &nbsp; **Y** (Molokanov) = {Y_gill:.4f}  
            **N_real** = {N_real:.2f} → **N_total = {N_total}** estágios
            """)
    
            st.markdown("#### ④ Kirkbride — Prato de alimentação")
            st.success(f"""
            **Nr/Ns** = {Nr_Ns:.4f}  
            **Nr** (retificação) = {Nr} &nbsp;|&nbsp; **Ns** (esgotamento) = {Ns}  
            **N_F** = **{N_feed}** (do topo)
            """)
        # ── TAB 2: Flash & Termodinâmica ───────────────────────────────────────
        with tab2:
            st.markdown(f"#### Flash Isotérmico — T = {T_feed:.2f} °C · P = {P_atm:.2f} atm · Modelo: {thermo_key}")
    
            col_a, col_b = st.columns(2)
            with col_a:
                psi_label = f"ψ (fração vaporizada) = {psi_F:.4f}"
                if psi_F < 0.01:
                    estado = "🟦 Líquido (ou próximo ao ponto de bolha)"
                elif psi_F > 0.99:
                    estado = "🟧 Vapor (ou próximo ao ponto de orvalho)"
                else:
                    estado = f"🟩 Mistura líquido-vapor (ψ = {psi_F:.3f})"
                st.info(f"{psi_label}\n\n{estado}")
    
            with col_b:
                if thermo_key == "NRTL":
                    pairs_found = []
                    for ci, cj in [(selected[i], selected[j])
                                   for i in range(len(selected))
                                   for j in range(i+1, len(selected))]:
                        if get_nrtl(ci, cj) is not None:
                            pairs_found.append(f"{ci} / {cj}")
                    if pairs_found:
                        st.success("Parâmetros NRTL encontrados:\n" + "\n".join(f"• {p}" for p in pairs_found))
                    else:
                        st.warning("Nenhum par com parâmetros NRTL. Usando Raoult para todos.")
    
            df_flash = pd.DataFrame({
                "Componente": selected,
                "K_i": [f"{v:.4f}" for v in K_feed],
                "α_i (rel. HK)": [f"{v:.4f}" for v in alpha_arr],
                "Psat [mmHg]": [f"{psat_mmhg(c, T_feed):.2f}" for c in selected],
                "x_i (líquido)": [f"{v:.4f}" for v in x_F],
                "y_i (vapor)": [f"{v:.4f}" for v in y_F],
            })
            st.dataframe(df_flash, use_container_width=True)
    
            if thermo_key == "NRTL":
                gamma = gamma_multicomp_nrtl(x_F, selected)
                df_gamma = pd.DataFrame({
                    "Componente": selected,
                    "γ_i (NRTL)": [f"{v:.4f}" for v in gamma],
                })
                st.markdown("##### Coeficientes de Atividade NRTL (fase líquida)")
                st.dataframe(df_gamma, use_container_width=True)
    
            # Temperaturas de bolha e orvalho
            st.markdown("---")
            try:
                T_bub = find_bubble_point(selected, z_arr, P_atm, thermo_key)
                T_dew = find_dew_point(selected, z_arr, P_atm, thermo_key)
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("T bolha (alimentação)", f"{T_bub:.2f} °C")
                cc2.metric("T orvalho (alimentação)", f"{T_dew:.2f} °C")
                cc3.metric("T flash utilizada", f"{T_feed:.2f} °C")
            except Exception as e:
                st.warning(f"Não foi possível calcular T bolha/orvalho: {e}")

    # ── TAB 3: Gráficos ────────────────────────────────────────────────────
    with tab3:
        gc1, gc2 = st.columns(2)
        with gc1:
            st.pyplot(plot_gilliland(X_gill, Y_gill, N_min, N_real, R_min, R_op))
        with gc2:
            st.pyplot(plot_underwood(alpha_arr, z_arr, theta, lk, hk))
        st.pyplot(plot_composicoes(selected, z_arr, x_D, x_B))
        st.pyplot(plot_K_volatilidades(selected, K_feed, alpha_arr))

    # ── TAB 4: Tabelas Detalhadas ──────────────────────────────────────────
    with tab4:
        st.markdown("#### Resumo Completo dos Resultados")
        rows = [
            ("N_min (Fenske)",          f"{N_min:.4f}",  "estágios teóricos"),
            ("θ (Underwood)",           f"{theta:.6f}",  "—"),
            ("V_min",                   f"{V_min:.3f}",  "kmol/h"),
            ("R_min",                   f"{R_min:.4f}",  "—"),
            (f"R_op ({R_factor}×R_min)",f"{R_op:.4f}",   "—"),
            ("X (Gilliland)",           f"{X_gill:.4f}", "—"),
            ("Y (Molokanov)",           f"{Y_gill:.4f}", "—"),
            ("N_real",                  f"{N_real:.2f}", "estágios"),
            ("N_total (arredondado)",   f"{N_total}",    "estágios teóricos"),
            ("Nr (retificação)",        f"{Nr}",         "estágios"),
            ("Ns (esgotamento)",        f"{Ns}",         "estágios"),
            ("N_F (prato alim., topo)", f"{N_feed}",     "do topo"),
            ("D (destilado)",           f"{D:.3f}",      "kmol/h"),
            ("B (fundo)",               f"{B:.3f}",      "kmol/h"),
            ("T flash alimentação",     f"{T_feed:.2f}", "°C"),
            ("ψ (fração vaporizada)",   f"{psi_F:.4f}",  "—"),
        ]
        df_full = pd.DataFrame(rows, columns=["Grandeza", "Valor", "Unidade"])
        st.dataframe(df_full, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### Banco de Dados — Parâmetros de Antoine dos Componentes Selecionados")
        db_rows = []
        for c in selected:
            d = COMPONENT_DB[c]
            db_rows.append([c, d["group"], f"{d['A']:.5f}", f"{d['B']:.3f}",
                            f"{d['C']:.3f}", f"{d['Tc']:.1f}", f"{d['Pc']:.1f}", f"{d['Mw']:.2f}"])
        df_db = pd.DataFrame(db_rows, columns=["Componente","Grupo","A","B","C",
                                               "Tc [K]","Pc [bar]","Mw [g/mol]"])
        st.dataframe(df_db, use_container_width=True, hide_index=True)

# ── RODAPÉ ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Simulador FUGK — Destilação Multicomponente Shortcut Method · "
    "Antoine (NIST/Perry's) · Rachford-Rice Flash · Raoult / NRTL (Renon-Prausnitz) · "
    "Fenske (1932) · Underwood (1948) · Gilliland (1940) / Molokanov (1972) · Kirkbride (1944) · "
    "DEQ/CTC/UEM · Prof. Dr. Ricardo V. P. Rezende"
)
