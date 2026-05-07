"""
╔══════════════════════════════════════════════════════════════════════╗
║        MÉTODO DE PONCHON-SAVARIT — App Streamlit Interativo         ║
║        Destilação Binária · Flash Calculations Rigorosas             ║
╚══════════════════════════════════════════════════════════════════════╝

Algoritmo de estágios correto:

  RETIFICAÇÃO (topo → alimentação):
    Partir de (xD, HL(xD)).
    Loop:
      1. Reta de (x_cur, HL_cur) ao polo ΔR → intersecta curva HV em (y_n, HV_n)
         [y_n calculado numericamente: onde a reta corta HV(y) = hV_mix(y, T_dew(y))]
      2. Flash de orvalho em y_n: dew_T(y_n) → x_n, T_n
         → isso define a TIE-LINE termodinâmica do estágio n
         → (x_n, HL_n) é o próximo ponto de partida
      Repetir até x_n <= zF.

  ESGOTAMENTO (fundo → alimentação):
    Partir de (xW, HL(xW)).
    Loop:
      1. Flash de bolha em x_cur: bubble_T(x_cur) → y_m, T_m
         → tie-line do estágio: (x_cur, HL_cur) ↔ (y_m, HV_m)
      2. Reta de (y_m, HV_m) ao polo ΔS → intersecta curva HL em (x_next, HL_next)
         [x_next calculado numericamente]
      3. x_cur ← x_next
      Repetir até x_cur >= zF.

  ISOTERMAS VISUAIS: distribuídas uniformemente em T no campo do diagrama.
  São independentes das tie-lines dos estágios (podem coincidentalmente coincidir).
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ponchon-Savarit",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{ font-family:'IBM Plex Sans',sans-serif; }
.stApp{ background:#0d1b2a; color:#e8f4f8; }
[data-testid="stSidebar"]{ background:#112233; border-right:1px solid #1e3a5f; }
[data-testid="stSidebar"] *{ color:#cce0f0 !important; }
.main-title{ font-family:'IBM Plex Mono',monospace; font-size:2.0rem;
             font-weight:600; color:#4fc3f7; letter-spacing:-1px; margin-bottom:0; }
.main-sub  { font-size:0.88rem; color:#78909c; margin-top:3px; font-family:'IBM Plex Mono',monospace; }
.sec-hdr   { font-family:'IBM Plex Mono',monospace; font-size:0.70rem; font-weight:600;
             letter-spacing:3px; color:#4fc3f7; text-transform:uppercase;
             margin:18px 0 6px 0; border-bottom:1px solid #1e3a5f; padding-bottom:3px; }
.rcard     { background:#1a2f45; border:1px solid #1e3a5f; border-left:4px solid #4fc3f7;
             border-radius:6px; padding:10px 14px; margin:5px 0;
             font-family:'IBM Plex Mono',monospace; font-size:0.82rem; }
.rcard.pr  { border-left-color:#ce93d8; }
.rcard.ps  { border-left-color:#ffb74d; }
.rcard.fd  { border-left-color:#26a69a; }
.rcard.stg { border-left-color:#ef5350; }
.rcard.wrn { border-left-color:#ffa726; background:#2a1f10; }
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
#  BANCO DE DADOS
# ═════════════════════════════════════════════════════════════════════════════
# Antoine: log10(P/mmHg) = A - B/(C + T[°C])
# Hvap_ref [kJ/mol] a 25°C; dHvap_dT [kJ/(mol·°C)]
# CpL, CpV [kJ/(mol·K)]; Tb [°C] a 1 atm

COMPOUNDS = {
    "Benzeno":   dict(A=6.90565, B=1211.033, C=220.790,
                      Hvap_ref=30.72, dHvap_dT=-0.060,
                      CpL=0.1350, CpV=0.0830, Tb=80.1,  col="#42a5f5"),
    "Tolueno":   dict(A=6.95334, B=1343.943, C=219.377,
                      Hvap_ref=33.18, dHvap_dT=-0.055,
                      CpL=0.1572, CpV=0.1030, Tb=110.6, col="#ef5350"),
    "Etanol":    dict(A=8.11220, B=1592.864, C=226.184,
                      Hvap_ref=38.56, dHvap_dT=-0.045,
                      CpL=0.1120, CpV=0.0780, Tb=78.37, col="#26a69a"),
    "Água":      dict(A=8.07131, B=1730.630, C=233.426,
                      Hvap_ref=44.00, dHvap_dT=-0.042,
                      CpL=0.0754, CpV=0.0340, Tb=100.0, col="#29b6f6"),
    "n-Heptano": dict(A=6.89386, B=1264.370, C=216.640,
                      Hvap_ref=31.77, dHvap_dT=-0.063,
                      CpL=0.2243, CpV=0.1620, Tb=98.4,  col="#ab47bc"),
    "n-Hexano":  dict(A=6.87601, B=1171.170, C=224.408,
                      Hvap_ref=28.85, dHvap_dT=-0.065,
                      CpL=0.1952, CpV=0.1430, Tb=68.7,  col="#ff8f00"),
    "Acetona":   dict(A=7.02447, B=1161.000, C=224.000,
                      Hvap_ref=31.27, dHvap_dT=-0.058,
                      CpL=0.1249, CpV=0.0740, Tb=56.1,  col="#66bb6a"),
    "Metanol":   dict(A=7.89750, B=1474.080, C=229.130,
                      Hvap_ref=37.43, dHvap_dT=-0.048,
                      CpL=0.0812, CpV=0.0480, Tb=64.7,  col="#ffa726"),
}

T_REF = 0.0   # referência de entalpia [°C]

# ═════════════════════════════════════════════════════════════════════════════
#  TERMODINÂMICA BÁSICA
# ═════════════════════════════════════════════════════════════════════════════

def psat(comp, T):
    """Pressão de vapor [mmHg], T em °C."""
    d = COMPOUNDS[comp]
    return 10.0 ** (d["A"] - d["B"] / (d["C"] + T))

def hvap(comp, T):
    """Calor latente [kJ/mol], T em °C."""
    d = COMPOUNDS[comp]
    return max(d["Hvap_ref"] + d["dHvap_dT"] * (T - 25.0), 0.5)

def hL_pure(comp, T):
    return COMPOUNDS[comp]["CpL"] * (T - T_REF)

def hV_pure(comp, T):
    d  = COMPOUNDS[comp]
    Tb = d["Tb"]
    return d["CpL"]*(Tb - T_REF) + hvap(comp, Tb) + d["CpV"]*(T - Tb)

def hL_mix(cA, cB, x, T):
    return x*hL_pure(cA, T) + (1-x)*hL_pure(cB, T)

def hV_mix(cA, cB, y, T):
    return y*hV_pure(cA, T) + (1-y)*hV_pure(cB, T)

def P_mmHg(P_bar):
    return P_bar * 750.062

# ═════════════════════════════════════════════════════════════════════════════
#  FLASH CALCULATIONS
# ═════════════════════════════════════════════════════════════════════════════

def _Trange(cA, cB):
    dA, dB = COMPOUNDS[cA], COMPOUNDS[cB]
    return min(dA["Tb"], dB["Tb"]) - 15, max(dA["Tb"], dB["Tb"]) + 40

def bubble_T(cA, cB, x, P_bar):
    """Temperatura de bolha [°C] para líquido de composição x."""
    Pm = P_mmHg(P_bar)
    lo, hi = _Trange(cA, cB)
    def f(T): return x*psat(cA,T) + (1-x)*psat(cB,T) - Pm
    try:
        return brentq(f, lo, hi, xtol=1e-5)
    except Exception:
        return None

def dew_T(cA, cB, y, P_bar):
    """Temperatura de orvalho [°C] para vapor de composição y."""
    Pm = P_mmHg(P_bar)
    lo, hi = _Trange(cA, cB)
    def f(T): return y/psat(cA,T) + (1-y)/psat(cB,T) - 1.0/Pm
    try:
        return brentq(f, lo, hi, xtol=1e-5)
    except Exception:
        return None

def flash_bub(cA, cB, x, P_bar):
    """
    Flash de bolha: dado x → T_bub, y*, HL(x,T), HV(y*,T).
    Retorna dict ou None.
    """
    T = bubble_T(cA, cB, x, P_bar)
    if T is None: return None
    Pm = P_mmHg(P_bar)
    y  = float(np.clip(x * psat(cA, T) / Pm, 0.0, 1.0))
    return dict(T=T, x=x, y=y,
                HL=hL_mix(cA, cB, x, T),
                HV=hV_mix(cA, cB, y, T))

def flash_dew(cA, cB, y, P_bar):
    """
    Flash de orvalho: dado y → T_dew, x*, HL(x*,T), HV(y,T).
    Retorna dict ou None.
    """
    T = dew_T(cA, cB, y, P_bar)
    if T is None: return None
    Pm = P_mmHg(P_bar)
    x  = float(np.clip(y * Pm / psat(cA, T), 0.0, 1.0))
    return dict(T=T, x=x, y=y,
                HL=hL_mix(cA, cB, x, T),
                HV=hV_mix(cA, cB, y, T))

# ═════════════════════════════════════════════════════════════════════════════
#  CURVAS DE EQUILÍBRIO HL(x) e HV(y)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def build_curves(cA, cB, P_bar, n=100):
    xs   = np.linspace(0.0, 1.0, n)
    yeq  = np.full(n, np.nan)
    HL   = np.full(n, np.nan)
    HV   = np.full(n, np.nan)
    Tbub = np.full(n, np.nan)
    for i, x in enumerate(xs):
        r = flash_bub(cA, cB, x, P_bar)
        if r:
            yeq[i]=r["y"]; HL[i]=r["HL"]; HV[i]=r["HV"]; Tbub[i]=r["T"]
    ok = ~np.isnan(HL)
    return xs[ok], yeq[ok], HL[ok], HV[ok], Tbub[ok]

# ═════════════════════════════════════════════════════════════════════════════
#  ISOTERMAS VISUAIS
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def build_isotherms_visual(cA, cB, P_bar, n_curves, Tmin, Tmax):
    """
    n_curves isotermas igualmente espaçadas em T.
    Cada uma: tie-line horizontal de (x_bub, HL_bub) a (y_dew, HV_dew).
    """
    Pm   = P_mmHg(P_bar)
    Ts   = np.linspace(Tmin + 0.3, Tmax - 0.3, n_curves)
    isos = []
    for T in Ts:
        PsA = psat(cA, T); PsB = psat(cB, T)
        if abs(PsA - PsB) < 1e-8: continue
        x_b = (Pm - PsB) / (PsA - PsB)
        if not (0.0 < x_b < 1.0): continue
        y_b = x_b * PsA / Pm                 # y em equilíbrio no ponto de bolha
        y_d = (1/PsB - 1/Pm) / (1/PsB - 1/PsA)   # y ponto de orvalho
        if not (0.0 < y_d < 1.0): continue
        isos.append(dict(
            T=T,
            x_bub=x_b,  HL_bub=hL_mix(cA, cB, x_b, T),
            y_dew=y_d,  HV_dew=hV_mix(cA, cB, y_d, T),
        ))
    return isos

# ═════════════════════════════════════════════════════════════════════════════
#  INTERSEÇÃO RETA-POLO COM CURVA HV  (retificação: polo=ΔR, busca y ∈ (x_cur,1))
#  INTERSEÇÃO RETA-POLO COM CURVA HL  (esgotamento: polo=ΔS, busca x ∈ (xW, y_cur))
# ═════════════════════════════════════════════════════════════════════════════

def _scan_zeros(f, lo, hi, n=80):
    """Retorna lista de intervalos onde f muda de sinal."""
    pts = np.linspace(lo, hi, n)
    vals = []
    for p in pts:
        try:    vals.append((p, f(p)))
        except: vals.append((p, np.nan))
    brackets = []
    for k in range(len(vals)-1):
        a, fa = vals[k]; b, fb = vals[k+1]
        if np.isnan(fa) or np.isnan(fb): continue
        if fa * fb < 0:
            brackets.append((a, b))
    return brackets

def intersect_pole_to_HV(x_cur, H_cur, pole_x, pole_H, cA, cB, P_bar):
    """
    Reta de (x_cur, H_cur) ao polo (pole_x, pole_H).
    Encontra y ∈ (x_cur, 1) onde H_reta(y) = HV_real(y).
    HV_real(y) via flash de orvalho.
    Retorna flash_dew dict ou None.
    """
    if abs(pole_x - x_cur) < 1e-9: return None
    slope = (pole_H - H_cur) / (pole_x - x_cur)

    def resid(y):
        r = flash_dew(cA, cB, y, P_bar)
        if r is None: return 1e9
        return r["HV"] - (H_cur + slope * (y - x_cur))

    lo = x_cur + 1e-4; hi = 1.0 - 1e-4
    if lo >= hi: return None

    brackets = _scan_zeros(resid, lo, hi)
    for a, b in brackets:
        try:
            y_sol = brentq(resid, a, b, xtol=1e-6)
            return flash_dew(cA, cB, y_sol, P_bar)
        except Exception:
            continue
    return None

def intersect_pole_to_HL(y_cur, H_cur, pole_x, pole_H, cA, cB, P_bar):
    """
    Reta de (y_cur, H_cur) ao polo (pole_x, pole_H).
    Encontra x ∈ (xW, y_cur) onde H_reta(x) = HL_real(x).
    HL_real(x) via flash de bolha.
    Retorna flash_bub dict ou None.
    """
    if abs(pole_x - y_cur) < 1e-9: return None
    slope = (pole_H - H_cur) / (pole_x - y_cur)

    def resid(x):
        r = flash_bub(cA, cB, x, P_bar)
        if r is None: return 1e9
        return r["HL"] - (H_cur + slope * (x - y_cur))

    lo = 1e-4; hi = y_cur - 1e-4
    if lo >= hi: return None

    brackets = _scan_zeros(resid, lo, hi)
    for a, b in brackets:
        try:
            x_sol = brentq(resid, a, b, xtol=1e-6)
            return flash_bub(cA, cB, x_sol, P_bar)
        except Exception:
            continue
    return None

# ═════════════════════════════════════════════════════════════════════════════
#  CÁLCULO DOS POLOS
# ═════════════════════════════════════════════════════════════════════════════

def compute_poles(cA, cB, P_bar, xD, xW, zF, R, q):
    f_xD = flash_bub(cA, cB, xD, P_bar)
    f_xW = flash_bub(cA, cB, xW, P_bar)
    f_zF = flash_bub(cA, cB, zF, P_bar)
    if not all([f_xD, f_xW, f_zF]): return None

    HL_xD  = f_xD["HL"]
    HV_top = f_xD["HV"]   # HV do vapor em equilíbrio com xD (condensador total)

    # Polo de retificação
    HD_p = (R + 1)*HV_top - R*HL_xD

    # HF segundo condição q
    HF = q * f_zF["HL"] + (1 - q) * f_zF["HV"]

    # Polo de esgotamento (colinearidade)
    slope = (HD_p - HF) / (xD - zF)
    HW_p  = HF + slope * (xW - zF)

    # Refluxo mínimo: scan das tie-lines extendidas até x=xD
    xs_scan = np.linspace(xW + 0.005, xD - 0.005, 80)
    HD_max = -1e10; Rm = None
    for xs in xs_scan:
        r = flash_bub(cA, cB, xs, P_bar)
        if r is None: continue
        dy = r["y"] - xs
        if abs(dy) < 1e-8: continue
        sl = (r["HV"] - r["HL"]) / dy
        hcand = r["HL"] + sl * (xD - xs)
        if hcand > HD_max:
            HD_max = hcand
            denom  = HV_top - HL_xD
            Rm = (HD_max - HV_top) / denom if abs(denom) > 1e-9 else None

    return dict(HD_p=HD_p, HW_p=HW_p, HF=HF, Rm=Rm,
                f_xD=f_xD, f_xW=f_xW, f_zF=f_zF,
                HL_xD=HL_xD, HV_top=HV_top)

# ═════════════════════════════════════════════════════════════════════════════
#  ALGORITMO DE ESTÁGIOS
# ═════════════════════════════════════════════════════════════════════════════

def calc_stages(cA, cB, P_bar, xD, xW, zF, poles, max_st=30):
    HD_p = poles["HD_p"]; HW_p = poles["HW_p"]
    HL_xD = poles["HL_xD"]

    # ─── RETIFICAÇÃO ────────────────────────────────────────────────────
    # Início: ponto (xD, HL(xD)) na curva de líquido
    # A cada passo:
    #   1. Reta (x_cur, HL_cur) → ΔR  →  intersecta HV  →  flash_dew(y_n)
    #      flash_dew retorna {T, x_eq, y_n, HL_eq, HV_n}
    #      A tie-line termodinâmica É: (x_eq, HL_eq) ↔ (y_n, HV_n)
    #   2. Próximo ponto: (x_eq, HL_eq)
    stages_R = []
    x_cur = xD; H_cur = HL_xD

    for i in range(max_st):
        # Passo 1 — reta operacional → ponto em HV
        r = intersect_pole_to_HV(x_cur, H_cur, xD, HD_p, cA, cB, P_bar)
        if r is None: break

        stages_R.append(dict(
            n        = i + 1,
            # reta operacional
            op_xA=x_cur,  op_HA=H_cur,
            op_xB=r["y"], op_HB=r["HV"],
            # tie-line termodinâmica
            tl_xL=r["x"], tl_HL=r["HL"],
            tl_yV=r["y"], tl_HV=r["HV"],
            T=r["T"],
        ))

        x_cur = r["x"]; H_cur = r["HL"]

        if x_cur <= zF + 1e-4: break

    # ─── ESGOTAMENTO ────────────────────────────────────────────────────
    # Início: ponto (xW, HL(xW)) na curva de líquido
    # A cada passo:
    #   1. Flash de bolha em x_cur  →  sobe tie-line  →  (y_m, HV_m)
    #   2. Reta (y_m, HV_m) → ΔS   →  intersecta HL  →  flash_bub(x_next)
    #      A tie-line termodinâmica É: (x_cur, HL_cur) ↔ (y_m, HV_m)
    #   3. Próximo ponto: (x_next, HL_next)
    stages_S = []
    HL_xW = poles["f_xW"]["HL"]
    x_cur = xW; H_cur = HL_xW

    for i in range(max_st):
        # Passo 1 — tie-line (flash de bolha sobe)
        r_bub = flash_bub(cA, cB, x_cur, P_bar)
        if r_bub is None: break

        y_m  = r_bub["y"]; HV_m = r_bub["HV"]; T_m = r_bub["T"]

        # Passo 2 — reta operacional de esgotamento → ponto em HL
        r_next = intersect_pole_to_HL(y_m, HV_m, xW, HW_p, cA, cB, P_bar)
        if r_next is None: break

        stages_S.append(dict(
            n        = i + 1,
            # tie-line termodinâmica
            tl_xL=x_cur,    tl_HL=H_cur,
            tl_yV=y_m,      tl_HV=HV_m,
            T=T_m,
            # reta operacional
            op_xA=y_m,          op_HA=HV_m,
            op_xB=r_next["x"],  op_HB=r_next["HL"],
        ))

        x_cur = r_next["x"]; H_cur = r_next["HL"]

        if x_cur >= zF - 1e-4: break

    return stages_R, stages_S

# ═════════════════════════════════════════════════════════════════════════════
#  PLOT H-x-y
# ═════════════════════════════════════════════════════════════════════════════

def make_hxy(cA, cB, P_bar,
             xs, yeq, HL, HV, Tbub,
             xD, xW, zF, R, q, poles,
             stages_R, stages_S,
             isos_vis, n_iso_show,
             show_iso, show_annot, show_stages,
             figsize=(12, 9)):

    HD_p=poles["HD_p"]; HW_p=poles["HW_p"]; HF=poles["HF"]
    HL_xD=poles["HL_xD"]; HL_xW=poles["f_xW"]["HL"]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0d1b2a"); ax.set_facecolor("#0d1b2a")

    # Região bifásica
    ax.fill_between(xs, HL, HV, color="#1c3555", alpha=0.50, zorder=1)

    # ── Isotermas visuais ──────────────────────────────────────────────
    if show_iso and isos_vis:
        shown = isos_vis[:n_iso_show]
        cmap  = plt.cm.plasma
        for k, iso in enumerate(shown):
            c = cmap(0.10 + 0.78 * k / max(len(shown)-1, 1))
            ax.plot([iso["x_bub"], iso["y_dew"]],
                    [iso["HL_bub"], iso["HV_dew"]],
                    color=c, lw=0.9, ls='--', alpha=0.60, zorder=2)
            ax.plot(iso["x_bub"], iso["HL_bub"], 'o', color=c, ms=3.5, alpha=0.70, zorder=3)
            ax.plot(iso["y_dew"], iso["HV_dew"], 'o', color=c, ms=3.5, alpha=0.70, zorder=3)
            ax.text(iso["x_bub"]-0.022, iso["HL_bub"],
                    f"{iso['T']:.0f}°", fontsize=6, color=c,
                    ha="right", va="center", fontfamily="monospace", alpha=0.88)

    # ── Curvas HL e HV ────────────────────────────────────────────────
    ax.plot(xs, HV, color="#4fc3f7", lw=2.5, zorder=4, label="$H_V(y)$ vapor sat.")
    ax.plot(xs, HL, color="#ef5350", lw=2.5, zorder=4, label="$H_L(x)$ líquido sat.")
    mid = len(xs)//2
    ax.text(xs[mid]+0.03, HV[mid]+0.22, "$H_V(y)$",
            color="#4fc3f7", fontsize=11, fontweight="bold", fontfamily="monospace")
    ax.text(xs[mid]+0.03, HL[mid]-0.48, "$H_L(x)$",
            color="#ef5350", fontsize=11, fontweight="bold", fontfamily="monospace")

    # ── Estágios ──────────────────────────────────────────────────────
    if show_stages:
        # Retificação
        for s in stages_R:
            # Reta operacional (linha fina de polo a HV)
            ax.plot([s["op_xA"], s["op_xB"]], [s["op_HA"], s["op_HB"]],
                    color="#9c67c8", lw=1.5, alpha=0.80, zorder=5,
                    label="Op. Ret." if s["n"]==1 else "")
            # Tie-line termodinâmica (tracejado mais espesso)
            ax.plot([s["tl_xL"], s["tl_yV"]], [s["tl_HL"], s["tl_HV"]],
                    color="#f48fb1", lw=2.2, ls='--', zorder=6,
                    label="Tie-line Ret." if s["n"]==1 else "")
            ax.plot(s["tl_yV"], s["tl_HV"], 'o', color="#9c67c8", ms=8, zorder=7,
                    markeredgecolor="white", markeredgewidth=0.5)
            ax.plot(s["tl_xL"], s["tl_HL"], 'o', color="#f48fb1", ms=8, zorder=7,
                    markeredgecolor="white", markeredgewidth=0.5)
            if show_annot:
                ax.text(s["tl_xL"]-0.01, s["tl_HL"]-0.5,
                        f"R{s['n']}", fontsize=8, color="#f48fb1",
                        fontweight="bold", ha="center", fontfamily="monospace")

        # Esgotamento
        for s in stages_S:
            ax.plot([s["tl_xL"], s["tl_yV"]], [s["tl_HL"], s["tl_HV"]],
                    color="#ffb74d", lw=2.2, ls='--', zorder=6,
                    label="Tie-line Esgo." if s["n"]==1 else "")
            ax.plot([s["op_xA"], s["op_xB"]], [s["op_HA"], s["op_HB"]],
                    color="#ef6c00", lw=1.5, alpha=0.80, zorder=5,
                    label="Op. Esgo." if s["n"]==1 else "")
            ax.plot(s["tl_yV"], s["tl_HV"], 'o', color="#ffb74d", ms=8, zorder=7,
                    markeredgecolor="white", markeredgewidth=0.5)
            ax.plot(s["op_xB"], s["op_HB"], 'o', color="#ef6c00", ms=8, zorder=7,
                    markeredgecolor="white", markeredgewidth=0.5)
            if show_annot:
                ax.text(s["op_xB"]-0.01, s["op_HB"]-0.5,
                        f"S{s['n']}", fontsize=8, color="#ffb74d",
                        fontweight="bold", ha="center", fontfamily="monospace")

    # ── Reta Δ_R – F – Δ_S ───────────────────────────────────────────
    if abs(xD - zF) > 1e-6:
        sl_rf = (HD_p - HF) / (xD - zF)
        xs_l  = np.array([xW-0.02, xD+0.01])
        ax.plot(xs_l, HF + sl_rf*(xs_l-zF),
                color="#b0b030", lw=1.2, ls='-.', alpha=0.65, zorder=3,
                label="Reta $\\Delta_R$–F–$\\Delta_S$")

    # ── Polos ─────────────────────────────────────────────────────────
    ax.plot(xD, HD_p, '*', color="#ce93d8", ms=22, zorder=9,
            markeredgecolor="white", markeredgewidth=0.7)
    ax.plot(xW, HW_p, '*', color="#ffb74d", ms=22, zorder=9,
            markeredgecolor="white", markeredgewidth=0.7)

    # ── Pontos de produto e alimentação ───────────────────────────────
    ax.plot(xD, HL_xD, 's', color="#4fc3f7", ms=10, zorder=8,
            markeredgecolor="white", markeredgewidth=0.7)
    ax.plot(xW, HL_xW, 's', color="#ef5350", ms=10, zorder=8,
            markeredgecolor="white", markeredgewidth=0.7)
    ax.plot(zF, HF,     'D', color="#26a69a",  ms=11, zorder=8,
            markeredgecolor="white", markeredgewidth=0.7)

    if show_annot:
        # Anotações dos polos
        ax.annotate(f"$\\Delta_R$  ({xD:.2f}, {HD_p:.2f})",
                    xy=(xD, HD_p), xytext=(xD-0.32, HD_p-0.9),
                    fontsize=8.5, color="#ce93d8", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#ce93d8", lw=1.2),
                    fontfamily="monospace", zorder=10,
                    bbox=dict(fc="#1a0a2e", ec="#ce93d8", alpha=0.88, boxstyle="round,pad=0.3"))
        ax.annotate(f"$\\Delta_S$  ({xW:.2f}, {HW_p:.2f})",
                    xy=(xW, HW_p), xytext=(xW+0.08, HW_p+0.8),
                    fontsize=8.5, color="#ffb74d", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#ffb74d", lw=1.2),
                    fontfamily="monospace", zorder=10,
                    bbox=dict(fc="#2a1500", ec="#ffb74d", alpha=0.88, boxstyle="round,pad=0.3"))
        # Anotações dos produtos
        for val, HL_v, lbl, col, dx in [
            (xD, HL_xD, f"$x_D={xD:.2f}$", "#4fc3f7", -0.16),
            (xW, HL_xW, f"$x_W={xW:.2f}$", "#ef5350", +0.07),
        ]:
            ax.annotate(lbl, xy=(val, HL_v), xytext=(val+dx, HL_v+0.9),
                        fontsize=8.5, color=col, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.0),
                        fontfamily="monospace", zorder=9)
        ax.annotate(f"F  $z_F={zF:.2f}$, $q={q:.2f}$\n$H_F={HF:.2f}$ kJ/mol",
                    xy=(zF, HF), xytext=(zF+0.07, HF+0.9),
                    fontsize=8, color="#26a69a", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#26a69a", lw=1.0),
                    fontfamily="monospace", zorder=9)

    # ── Eixos e formatação ────────────────────────────────────────────
    ax.set_xlim(-0.03, 1.05)
    allH = np.concatenate([HL[~np.isnan(HL)], HV[~np.isnan(HV)]])
    Hlo  = min(float(allH.min())-1.0, HW_p-1.5)
    Hhi  = max(float(allH.max())+1.0, HD_p+1.5)
    mg   = (Hhi-Hlo)*0.06
    ax.set_ylim(Hlo-mg, Hhi+mg)

    ax.set_xlabel(f"Fração molar de {cA}  ($x$ ou $y$)",
                  color="#90caf9", fontsize=12, labelpad=8, fontfamily="monospace")
    ax.set_ylabel("Entalpia molar  $H$  (kJ/mol)",
                  color="#90caf9", fontsize=12, labelpad=8, fontfamily="monospace")
    ax.set_title(
        f"H-x-y  ·  {cA}/{cB}  ·  P={P_bar:.3f} bar  ·  R={R:.2f}  ·  q={q:.2f}",
        color="#4fc3f7", fontsize=12.5, fontweight="bold",
        fontfamily="monospace", pad=12)

    ax.tick_params(colors="#607d8b", labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
    ax.grid(True, color="#1e3a5f", lw=0.6, alpha=0.8)
    ax.grid(True, which='minor', color="#141f2e", lw=0.3, alpha=0.5)
    ax.minorticks_on()

    hs, ls = ax.get_legend_handles_labels()
    by_l = dict(zip(ls, hs))
    ax.legend(by_l.values(), by_l.keys(), loc="upper left",
              fontsize=8.5, framealpha=0.85, edgecolor="#1e3a5f",
              facecolor="#0d1b2a", labelcolor="#cce0f0")
    fig.tight_layout(pad=0.8)
    return fig

def make_yx(xs, yeq, cA, cB, xD, xW, zF, stages_R, stages_S, show_stages):
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    fig.patch.set_facecolor("#0d1b2a"); ax.set_facecolor("#112233")

    ax.plot([0,1],[0,1], color="#455a64", lw=1.0, ls='--')
    ax.plot(xs, yeq, color="#4fc3f7", lw=2.2, label="Equilíbrio")

    if show_stages:
        for s in stages_R:
            # Passo horizontal (de x* até y*) e vertical (de y* sobe)
            ax.plot([s["tl_xL"], s["tl_xL"]], [s["tl_xL"], s["tl_yV"]],
                    color="#9c67c8", lw=1.2, alpha=0.80)
            ax.plot([s["tl_xL"], s["tl_yV"]], [s["tl_yV"], s["tl_yV"]],
                    color="#f48fb1", lw=1.2, ls='--', alpha=0.80)
            ax.plot(s["tl_xL"], s["tl_yV"], 'o',
                    color="#f48fb1", ms=6, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.4)
        for s in stages_S:
            ax.plot([s["tl_xL"], s["tl_xL"]], [s["tl_xL"], s["tl_yV"]],
                    color="#ef6c00", lw=1.2, alpha=0.80)
            ax.plot([s["tl_xL"], s["tl_yV"]], [s["tl_yV"], s["tl_yV"]],
                    color="#ffb74d", lw=1.2, ls='--', alpha=0.80)
            ax.plot(s["tl_xL"], s["tl_yV"], 'o',
                    color="#ffb74d", ms=6, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.4)

    for val, col, lbl in [(xD,"#4fc3f7","$x_D$"),
                           (xW,"#ef5350","$x_W$"),
                           (zF,"#26a69a","$z_F$")]:
        ax.axvline(val, color=col, lw=0.7, ls=':', alpha=0.55)
        ax.text(val+0.012, 0.04, lbl, color=col, fontsize=8.5, fontfamily="monospace")

    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel(f"$x$  ({cA})", color="#90caf9", fontsize=10, fontfamily="monospace")
    ax.set_ylabel(f"$y$  ({cA})", color="#90caf9", fontsize=10, fontfamily="monospace")
    ax.set_title("Diagrama $y$-$x$", color="#4fc3f7", fontsize=11, fontfamily="monospace")
    ax.tick_params(colors="#607d8b", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
    ax.grid(True, color="#1e3a5f", lw=0.6, alpha=0.7)
    fig.tight_layout(pad=0.5)
    return fig

# ═════════════════════════════════════════════════════════════════════════════
#  INTERFACE
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-title">⚗ Ponchon–Savarit</div>
<div class="main-sub">Flash Calculations · Diagrama H-x-y · Tie-lines Termodinâmicas</div>
""", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.markdown('<div class="sec-hdr">Sistema</div>', unsafe_allow_html=True)
    clist = list(COMPOUNDS.keys())
    cA = st.selectbox("Componente leve (A)", clist, index=0)
    cB_opts = [c for c in clist if c != cA]
    cB = st.selectbox("Componente pesado (B)", cB_opts, index=1)
    P_bar = st.slider("Pressão (bar)", 0.10, 10.0, 1.013, 0.005, format="%.3f bar")

    st.markdown('<div class="sec-hdr">Coluna</div>', unsafe_allow_html=True)
    xD = st.slider("Destilado  xD",   0.50, 0.999, 0.90, 0.005, format="%.3f")
    xW = st.slider("Resíduo    xW",   0.001, 0.45, 0.05, 0.005, format="%.3f")
    zF = st.slider("Alimentação zF",  0.05,  0.95, 0.45, 0.005, format="%.3f")
    R  = st.slider("Refluxo  R",      0.5,  12.0,  2.5,  0.1,   format="%.1f")

    q_opt = st.selectbox("Condição da alimentação",
                         ["1.0 — Líquido saturado",
                          "0.0 — Vapor saturado",
                          "Personalizado"])
    if   "1.0" in q_opt: q = 1.0
    elif "0.0" in q_opt: q = 0.0
    else:                q = st.slider("Valor de q", -0.5, 1.5, 0.8, 0.05)

    st.markdown('<div class="sec-hdr">Isotermas Visuais</div>', unsafe_allow_html=True)
    show_iso   = st.toggle("Mostrar isotermas (campo T)", value=True)
    n_iso_show = st.slider("Quantidade", 3, 25, 10, 1) if show_iso else 0

    st.markdown('<div class="sec-hdr">Visualização</div>', unsafe_allow_html=True)
    show_stages = st.toggle("Desenhar estágios", value=True)
    show_annot  = st.toggle("Anotações",         value=True)
    n_pts       = st.slider("Pontos nas curvas", 40, 200, 80, 10)

# Validação
errs = []
if xW >= zF:  errs.append("xW deve ser < zF")
if zF >= xD:  errs.append("zF deve ser < xD")
if COMPOUNDS[cA]["Tb"] >= COMPOUNDS[cB]["Tb"]:
    errs.append(f"{cA} (Tb={COMPOUNDS[cA]['Tb']:.1f}°C) deve ter Tb < {cB} (Tb={COMPOUNDS[cB]['Tb']:.1f}°C)")
for e in errs:
    st.markdown(f'<div class="rcard wrn">⚠ {e}</div>', unsafe_allow_html=True)
if errs: st.stop()

# ── Computações ───────────────────────────────────────────────────────
with st.spinner("Calculando curvas de equilíbrio…"):
    xs, yeq, HL, HV, Tbub = build_curves(cA, cB, P_bar, n=n_pts)

if xs is None or len(xs) < 5:
    st.error("Erro nas curvas — verifique os parâmetros."); st.stop()

isos_vis = []
if show_iso:
    with st.spinner("Calculando isotermas visuais…"):
        isos_vis = build_isotherms_visual(
            cA, cB, P_bar, 40, float(Tbub.min()), float(Tbub.max()))

with st.spinner("Calculando polos…"):
    poles = compute_poles(cA, cB, P_bar, xD, xW, zF, R, q)

if poles is None:
    st.error("Não foi possível calcular os polos."); st.stop()

Rm = poles["Rm"]
if Rm is not None and R < Rm:
    st.markdown(
        f'<div class="rcard wrn">⛔  R = {R:.3f} < R_min ≈ {Rm:.4f} — '
        f'Separação impossível!</div>', unsafe_allow_html=True)

stages_R, stages_S = [], []
if show_stages:
    with st.spinner("Calculando estágios (flash rigoroso)…"):
        stages_R, stages_S = calc_stages(
            cA, cB, P_bar, xD, xW, zF, poles)

# ── Layout ────────────────────────────────────────────────────────────
col_plot, col_info = st.columns([2.7, 1.0])

with col_plot:
    tab_hxy, tab_yx = st.tabs(["📊 H-x-y", "📈 y-x"])
    with tab_hxy:
        with st.spinner("Renderizando…"):
            fig = make_hxy(cA, cB, P_bar, xs, yeq, HL, HV, Tbub,
                           xD, xW, zF, R, q, poles,
                           stages_R, stages_S,
                           isos_vis, n_iso_show,
                           show_iso, show_annot, show_stages)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    with tab_yx:
        fig2 = make_yx(xs, yeq, cA, cB, xD, xW, zF,
                       stages_R, stages_S, show_stages)
        st.pyplot(fig2, use_container_width=True); plt.close(fig2)

with col_info:
    st.markdown('<div class="sec-hdr">Resultados</div>', unsafe_allow_html=True)
    ratio = f"{R/Rm:.3f}" if (Rm and Rm > 0) else "—"
    st.markdown(f"""
<div class="rcard pr"><b>Polo Δ<sub>R</sub></b><br>
  x = {xD:.3f} &nbsp;|&nbsp; H'<sub>D</sub> = <b>{poles['HD_p']:.3f} kJ/mol</b>
</div>
<div class="rcard ps"><b>Polo Δ<sub>S</sub></b><br>
  x = {xW:.3f} &nbsp;|&nbsp; H'<sub>W</sub> = <b>{poles['HW_p']:.3f} kJ/mol</b>
</div>
<div class="rcard fd"><b>Alimentação F</b><br>
  z<sub>F</sub>={zF:.3f} · q={q:.2f}<br>
  H<sub>F</sub> = <b>{poles['HF']:.3f} kJ/mol</b>
</div>
<div class="rcard"><b>Refluxo</b><br>
  R<sub>min</sub> ≈ {Rm:.4f if Rm else '—'}<br>
  R/R<sub>min</sub> = <b>{ratio}</b>
</div>
""", unsafe_allow_html=True)

    n_tot = len(stages_R) + len(stages_S)
    if n_tot:
        st.markdown(f"""
<div class="rcard stg"><b>Estágios ideais</b><br>
  Total: <b>{n_tot}</b>  (incl. refervedor)<br>
  Ret.: {len(stages_R)}  |  Esgo.: {len(stages_S)}<br>
  Pratos na coluna: <b>{n_tot-1}</b>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">Compostos</div>', unsafe_allow_html=True)
    dA=COMPOUNDS[cA]; dB=COMPOUNDS[cB]
    TbA=bubble_T(cA,cB,1.0,P_bar); TbB=bubble_T(cA,cB,0.0,P_bar)
    st.markdown(f"""
<div class="rcard" style="border-left-color:{dA['col']}">
  <b>{cA}</b> (leve)<br>
  T<sub>eb</sub> = {TbA:.1f if TbA else '?'}°C @ {P_bar:.3f} bar<br>
  λ(25°C) = {dA['Hvap_ref']:.2f} kJ/mol
</div>
<div class="rcard" style="border-left-color:{dB['col']}">
  <b>{cB}</b> (pesado)<br>
  T<sub>eb</sub> = {TbB:.1f if TbB else '?'}°C @ {P_bar:.3f} bar<br>
  λ(25°C) = {dB['Hvap_ref']:.2f} kJ/mol
</div>""", unsafe_allow_html=True)

    if n_tot:
        st.markdown('<div class="sec-hdr">Tabela de Estágios</div>', unsafe_allow_html=True)
        with st.expander("Ver detalhes"):
            import pandas as pd
            rows = []
            for s in stages_R:
                rows.append({"N":s["n"],"Seção":"Ret.",
                              "x* (liq)":f"{s['tl_xL']:.4f}",
                              "y* (vap)":f"{s['tl_yV']:.4f}",
                              "T (°C)":f"{s['T']:.1f}",
                              "HL (kJ/mol)":f"{s['tl_HL']:.3f}",
                              "HV (kJ/mol)":f"{s['tl_HV']:.3f}"})
            for s in stages_S:
                rows.append({"N":len(stages_R)+s["n"],"Seção":"Esgo.",
                              "x* (liq)":f"{s['tl_xL']:.4f}",
                              "y* (vap)":f"{s['tl_yV']:.4f}",
                              "T (°C)":f"{s['T']:.1f}",
                              "HL (kJ/mol)":f"{s['tl_HL']:.3f}",
                              "HV (kJ/mol)":f"{s['tl_HV']:.3f}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with st.expander("ℹ️ Algoritmo e modelos"):
    st.markdown(f"""
**VLE:** Lei de Raoult + Antoine (log₁₀P = A − B/(C+T), P em mmHg, T em °C)

**Entalpias** (ref.: líquido puro a {T_REF}°C = 0 kJ/mol):
- Líq. puro: CpL·(T−Tref)  |  Vap. puro: CpL·(Tb−Tref) + λ(Tb) + CpV·(T−Tb)
- Mistura: regra de mistura ideal (sem calor de mistura)

**Polos:**  H'_D = (R+1)·HV(y_topo) − R·HL(xD)  |  Δ_S por colinearidade Δ_R–F–Δ_S

**Estágios — Retificação:**
  reta (x_cur,HL) → Δ_R → corta HV (interseção numérica) → flash de orvalho em y_n
  → tie-line termodinâmica: (x*,HL*) ↔ (y_n,HV_n)  →  próximo x_cur = x*

**Estágios — Esgotamento:**
  flash de bolha em x_cur → tie-line: (x_cur,HL) ↔ (y_m,HV)
  → reta (y_m,HV) → Δ_S → corta HL (interseção numérica) → próximo x_cur

**Isotermas visuais:** espaçadas uniformemente em T ∈ [T_bub(xW), T_bub(xD)].
São independentes das tie-lines dos estágios (podem coincidir por acaso).
""")
