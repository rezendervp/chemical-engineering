"""
╔══════════════════════════════════════════════════════════════════════════╗
║  MÉTODO DE PONCHON-SAVARIT  —  App Streamlit                           ║
║  Flash via Antoine → CubicSpline → algoritmo robusto de estágios       ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Abscissa única: x (fração molar do leve no líquido) para TODAS as     ║
║  splines. Isso torna o algoritmo robusto mesmo em sistemas com          ║
║  azeótropo.                                                             ║
║                                                                         ║
║  spHL(x)  = HL de bolha na composição x                                ║
║  spHV(x)  = HV de orvalho do vapor em equilíbrio com x  (≡ HV(y*(x))) ║
║  spYeq(x) = y*(x)  — equilíbrio                                        ║
║  spTbub(x)= T de bolha em x  (para labels das tie-lines)               ║
║                                                                         ║
║  RETIFICAÇÃO:                                                           ║
║    Dado (x_cur, HL_cur), reta ao polo ΔR:                              ║
║      H_line(x) = HL_cur + slope*(spYeq(x) - spYeq(x_cur))             ║
║    Intersecção com spHV(x):  resid(x) = spHV(x) - H_line(x)           ║
║    Brentq em (x_cur+ε, xD-ε) → x_n;  y_n = spYeq(x_n)               ║
║    Tie-line: (x_n, spHL(x_n)) ↔ (y_n, spHV(x_n))                     ║
║    Próximo: x_cur ← x_n                                                ║
║                                                                         ║
║  ESGOTAMENTO:                                                           ║
║    Dado (x_cur, HL_cur), tie-line sobe: y_m = spYeq(x_cur)            ║
║    Reta de (y_m, spHV(x_cur)) ao polo ΔS:                             ║
║      H_line(x) = HV_cur + slope*(spYeq(x) - y_m)                      ║
║    Intersecção com spHL(x): resid(x) = spHL(x) - H_line(x)            ║
║    Brentq em (x_cur+ε, xD) → x_next                                   ║
║    Próximo: x_cur ← x_next                                             ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
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
.main-title{ font-family:'IBM Plex Mono',monospace; font-size:1.9rem;
             font-weight:600; color:#4fc3f7; letter-spacing:-1px; }
.main-sub  { font-size:0.86rem; color:#78909c; font-family:'IBM Plex Mono',monospace; }
.sec-hdr   { font-family:'IBM Plex Mono',monospace; font-size:0.68rem; font-weight:600;
             letter-spacing:3px; color:#4fc3f7; text-transform:uppercase;
             margin:16px 0 5px 0; border-bottom:1px solid #1e3a5f; padding-bottom:3px; }
.rcard     { background:#1a2f45; border:1px solid #1e3a5f; border-left:4px solid #4fc3f7;
             border-radius:6px; padding:10px 14px; margin:5px 0;
             font-family:'IBM Plex Mono',monospace; font-size:0.81rem; line-height:1.6; }
.rcard.pr  { border-left-color:#ce93d8; }
.rcard.ps  { border-left-color:#ffb74d; }
.rcard.fd  { border-left-color:#26a69a; }
.rcard.stg { border-left-color:#ef5350; }
.rcard.wrn { border-left-color:#ffa726; background:#2a1f10; }
.rcard.ok  { border-left-color:#66bb6a; }
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
#  BANCO DE DADOS
#  Antoine: log10(P/mmHg) = A - B/(C + T[°C])
#  Hvap_ref [kJ/mol] a 25 °C;  dHvap_dT [kJ/(mol·°C)]
#  CpL, CpV  [kJ/(mol·K)];    Tb [°C] a 1 atm
# ═════════════════════════════════════════════════════════════════════════════
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
T_REF = 0.0   # °C — entalpia de referência (líquido puro = 0 kJ/mol)

# ═════════════════════════════════════════════════════════════════════════════
#  FUNÇÕES TERMODINÂMICAS PURAS
# ═════════════════════════════════════════════════════════════════════════════

def psat(comp, T):
    d = COMPOUNDS[comp]
    return 10.0 ** (d["A"] - d["B"] / (d["C"] + T))   # mmHg

def P_mmHg(P_bar): return P_bar * 750.062

def hvap(comp, T):
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

def _Trange(cA, cB):
    lo = min(COMPOUNDS[cA]["Tb"], COMPOUNDS[cB]["Tb"]) - 15
    hi = max(COMPOUNDS[cA]["Tb"], COMPOUNDS[cB]["Tb"]) + 40
    return lo, hi

def bubble_T(cA, cB, x, P_bar):
    """Temperatura de bolha para líquido de composição x."""
    Pm = P_mmHg(P_bar)
    lo, hi = _Trange(cA, cB)
    try:
        return brentq(lambda T: x*psat(cA,T) + (1-x)*psat(cB,T) - Pm,
                      lo, hi, xtol=1e-5)
    except Exception:
        return None

# ═════════════════════════════════════════════════════════════════════════════
#  CONSTRUÇÃO DAS CURVAS + SPLINES
#  Tudo parametrizado em x ∈ [0,1].
#  spYeq(x) = y*(x),  spHL(x) = HL(x),  spHV(x) = HV(y*(x)),  spTbub(x)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def build_splines(cA, cB, P_bar, n=160):
    """
    Calcula n pontos via flash de bolha e ajusta splines cúbicas.
    Retorna dict com arrays e 4 splines.
    """
    Pm   = P_mmHg(P_bar)
    xs   = np.linspace(0.0, 1.0, n)
    yeq  = np.zeros(n)
    HL   = np.zeros(n)
    HV   = np.zeros(n)
    Tbub = np.zeros(n)

    for i, x in enumerate(xs):
        T = bubble_T(cA, cB, x, P_bar)
        if T is None:
            T = COMPOUNDS[cA]["Tb"] + (COMPOUNDS[cB]["Tb"] -
                COMPOUNDS[cA]["Tb"]) * x  # fallback linear
        y       = float(np.clip(x * psat(cA, T) / Pm, 0.0, 1.0))
        yeq[i]  = y
        HL[i]   = hL_mix(cA, cB, x, T)
        HV[i]   = hV_mix(cA, cB, y, T)
        Tbub[i] = T

    # volatilidade relativa local e média geométrica
    with np.errstate(divide='ignore', invalid='ignore'):
        mask  = (xs > 0.01) & (xs < 0.99) & (yeq > xs)
        alpha_local = np.where(mask,
            yeq*(1-xs) / np.where(xs*(1-yeq) > 1e-12, xs*(1-yeq), np.nan),
            np.nan)
    alpha_mean = float(np.exp(np.nanmean(np.log(alpha_local[mask]))))

    sp = dict(
        xs=xs, yeq=yeq, HL=HL, HV=HV, Tbub=Tbub,
        alpha_mean=alpha_mean,
        spHL  = CubicSpline(xs, HL),
        spHV  = CubicSpline(xs, HV),    # HV como função de x
        spYeq = CubicSpline(xs, yeq),
        spTbub= CubicSpline(xs, Tbub),
    )
    return sp

# ═════════════════════════════════════════════════════════════════════════════
#  POLOS E ENTALPIA DA ALIMENTAÇÃO
# ═════════════════════════════════════════════════════════════════════════════

def compute_poles(sp, xD, xW, zF, R, q):
    spHL = sp["spHL"]; spHV = sp["spHV"]; spYeq = sp["spYeq"]

    HL_xD  = float(spHL(xD))
    HV_top = float(spHV(xD))   # HV do vapor em equilíbrio com xD

    # Polo de retificação
    HD_p = (R + 1)*HV_top - R*HL_xD

    # Entalpia da alimentação
    HL_zF = float(spHL(zF))
    HV_zF = float(spHV(zF))
    HF    = q*HL_zF + (1 - q)*HV_zF

    # Polo de esgotamento (colinearidade ΔR – F – ΔS)
    slope_rf = (HD_p - HF) / (xD - zF)
    HW_p     = HF + slope_rf*(xW - zF)

    # Rmin: extensão das tie-lines até x = xD
    # Tie-line em x_i: slope_tl = (HV(x_i) - HL(x_i)) / (yeq(x_i) - x_i)
    # H_at_xD = HL(x_i) + slope_tl * (xD - x_i)
    xs_s   = sp["xs"]
    mask   = (xs_s > xW + 0.01) & (xs_s < xD - 0.01)
    xs_m   = xs_s[mask]
    HLm    = spHL(xs_m); HVm = spHV(xs_m); Ym = spYeq(xs_m)
    dy     = Ym - xs_m
    ok     = np.abs(dy) > 1e-6
    sl_tl  = np.where(ok, (HVm - HLm) / dy, 0.0)
    H_at_xD = HLm + sl_tl*(xD - xs_m)
    idx_max = int(np.argmax(H_at_xD))
    HD_min  = float(H_at_xD[idx_max])
    denom   = HV_top - HL_xD
    Rm      = float((HD_min - HV_top) / denom) if abs(denom) > 1e-9 else None

    return dict(HD_p=HD_p, HW_p=HW_p, HF=HF, Rm=Rm,
                HL_xD=HL_xD, HV_top=HV_top,
                HL_xW=float(spHL(xW)), HL_zF=HL_zF, HV_zF=HV_zF)

# ═════════════════════════════════════════════════════════════════════════════
#  INTERSECÇÕES RETA–CURVA  (núcleo do algoritmo)
# ═════════════════════════════════════════════════════════════════════════════

def _first_root(f, lo, hi, n_scan=60):
    """Varre [lo,hi] com n_scan pontos e retorna a primeira raiz via brentq."""
    pts  = np.linspace(lo, hi, n_scan)
    vals = np.array([f(p) for p in pts])
    for k in range(len(vals)-1):
        if np.isnan(vals[k]) or np.isnan(vals[k+1]): continue
        if vals[k]*vals[k+1] < 0.0:
            try:
                return brentq(f, pts[k], pts[k+1], xtol=1e-7)
            except Exception:
                continue
    return None

def intersect_rect(x_cur, HL_cur, xD, HD_p, sp):
    """
    RETIFICAÇÃO: reta de (x_cur, HL_cur) ao polo ΔR = (xD, HD_p).
    A reta é parametrizada em x (abscissa do diagrama):
        H_line(x) = HL_cur + slope * (spYeq(x) - spYeq(x_cur))
    onde slope = (HD_p - HL_cur) / (1 - spYeq(x_cur))  … não,
    vamos usar:
        slope = (HD_p - HL_cur) / (xD - x_cur)
        H_line(x) = HL_cur + slope * (x - x_cur)   ← reta em x
    Intersecção com spHV(x):
        resid(x) = spHV(x) - H_line(x)
    Busca em x ∈ (x_cur + ε, xD - ε).
    Retorna x_n (ponto em HV) ou None.
    """
    if abs(xD - x_cur) < 1e-8: return None
    slope = (HD_p - HL_cur) / (xD - x_cur)
    spHV  = sp["spHV"]

    def resid(x):
        H_line = HL_cur + slope*(x - x_cur)
        return float(spHV(x)) - H_line

    lo = x_cur + 1e-4
    hi = xD    - 1e-4
    if lo >= hi: return None
    return _first_root(resid, lo, hi)

def intersect_strip(y_cur, HV_cur, xW, HW_p, sp):
    """
    ESGOTAMENTO: reta de (y_cur, HV_cur) ao polo ΔS = (xW, HW_p).
    Nota: y_cur é a composição do VAPOR (abscissa em HV).
    A reta em x:
        slope = (HW_p - HV_cur) / (xW - y_cur)
        H_line(x) = HV_cur + slope*(x - y_cur)
    Intersecção com spHL(x):
        resid(x) = spHL(x) - H_line(x)
    Busca em x ∈ (xW + ε, y_cur - ε).
    Retorna x_next (ponto em HL) ou None.
    """
    if abs(xW - y_cur) < 1e-8: return None
    slope = (HW_p - HV_cur) / (xW - y_cur)
    spHL  = sp["spHL"]

    def resid(x):
        H_line = HV_cur + slope*(x - y_cur)
        return float(spHL(x)) - H_line

    lo = xW    + 1e-4
    hi = y_cur - 1e-4
    if lo >= hi: return None
    return _first_root(resid, lo, hi)

# ═════════════════════════════════════════════════════════════════════════════
#  ALGORITMO DE ESTÁGIOS
# ═════════════════════════════════════════════════════════════════════════════

def calc_stages(sp, poles, xD, xW, zF, max_st=40):
    """
    Retorna (stages_R, stages_S) — listas de dicts com geometria de cada estágio.

    RETIFICAÇÃO — de xD descendo até zF:
      Partida: x_cur = xD,  HL_cur = HL(xD)
      Loop:
        1. intersect_rect → x_n  (ponto em HV)
        2. y_n   = spYeq(x_n)          ← tie-line termodinâmica
           HL_n  = spHL(x_n)
           HV_n  = spHV(x_n)
        3. x_cur ← x_n;  HL_cur ← HL_n
        Para quando x_n ≤ zF.

    ESGOTAMENTO — de xW subindo até zF:
      Partida: x_cur = xW,  HL_cur = HL(xW)
      Loop:
        1. y_m   = spYeq(x_cur)        ← tie-line sobe
           HV_m  = spHV(x_cur)
        2. intersect_strip(y_m, HV_m) → x_next  (ponto em HL)
           HL_next = spHL(x_next)
        3. x_cur ← x_next;  HL_cur ← HL_next
        Para quando x_next ≥ zF.
    """
    spHL  = sp["spHL"]; spHV = sp["spHV"]; spYeq = sp["spYeq"]
    HD_p  = poles["HD_p"]; HW_p = poles["HW_p"]

    # ── RETIFICAÇÃO ──────────────────────────────────────────────────────────
    stages_R = []
    x_cur  = xD
    HL_cur = poles["HL_xD"]

    for i in range(max_st):
        x_n = intersect_rect(x_cur, HL_cur, xD, HD_p, sp)
        if x_n is None or x_n <= 1e-5 or x_n >= xD - 1e-5:
            break

        y_n   = float(spYeq(x_n))
        HL_n  = float(spHL(x_n))
        HV_n  = float(spHV(x_n))
        T_n   = float(sp["spTbub"](x_n))

        stages_R.append(dict(
            n=i+1,
            # reta operacional: (x_cur,HL_cur) → polo → ponto HV em x_n
            op_x0=x_cur, op_H0=HL_cur,
            op_x1=x_n,   op_H1=HV_n,
            # tie-line: (x_n,HL_n) ↔ (y_n,HV_n)
            tl_xL=x_n,  tl_HL=HL_n,
            tl_yV=y_n,  tl_HV=HV_n,
            T=T_n,
        ))

        x_cur  = x_n
        HL_cur = HL_n
        if x_cur <= zF + 1e-4:
            break

    # ── ESGOTAMENTO ──────────────────────────────────────────────────────────
    stages_S = []
    x_cur  = xW
    HL_cur = poles["HL_xW"]

    for i in range(max_st):
        y_m  = float(spYeq(x_cur))
        HV_m = float(spHV(x_cur))
        T_m  = float(sp["spTbub"](x_cur))

        x_next = intersect_strip(y_m, HV_m, xW, HW_p, sp)
        if x_next is None or x_next <= xW + 1e-5:
            break

        HL_next = float(spHL(x_next))

        stages_S.append(dict(
            n=i+1,
            # tie-line: (x_cur,HL_cur) ↔ (y_m,HV_m)
            tl_xL=x_cur,  tl_HL=HL_cur,
            tl_yV=y_m,    tl_HV=HV_m,
            T=T_m,
            # reta operacional: (y_m,HV_m) → polo ΔS → (x_next,HL_next)
            op_x0=y_m,    op_H0=HV_m,
            op_x1=x_next, op_H1=HL_next,
        ))

        x_cur  = x_next
        HL_cur = HL_next
        if x_cur >= zF - 1e-4:
            break

    return stages_R, stages_S

# ═════════════════════════════════════════════════════════════════════════════
#  TIE-LINES VISUAIS (isotermas)
#  N pontos de x uniformemente espaçados em [xW, xD].
#  Cada tie-line: (x_i, HL(x_i)) ↔ (yeq(x_i), HV(x_i))
# ═════════════════════════════════════════════════════════════════════════════

def build_tielines_visual(sp, xW, xD, n):
    xs_tl   = np.linspace(xW, xD, n + 2)[1:-1]   # exclui xW e xD
    spHL    = sp["spHL"]; spHV = sp["spHV"]
    spYeq   = sp["spYeq"]; spTbub = sp["spTbub"]
    tls = []
    for x in xs_tl:
        y = float(spYeq(x))
        tls.append(dict(
            xL=x,           HL=float(spHL(x)),
            yV=y,           HV=float(spHV(x)),
            T=float(spTbub(x)),
        ))
    return tls

# ═════════════════════════════════════════════════════════════════════════════
#  PLOT PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

def make_plot(sp, poles, stages_R, stages_S, tls_vis,
              cA, cB, P_bar, xD, xW, zF, R, q,
              show_tls, n_tl_show, show_stages, show_annot,
              figsize=(12, 9)):

    xs   = sp["xs"]; HL = sp["HL"]; HV = sp["HV"]
    HD_p = poles["HD_p"]; HW_p = poles["HW_p"]; HF = poles["HF"]
    HL_xD= poles["HL_xD"]; HL_xW = poles["HL_xW"]
    spYeq= sp["spYeq"]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0d1b2a"); ax.set_facecolor("#0d1b2a")

    # ── Região bifásica ────────────────────────────────────────────────
    ax.fill_between(xs, HL, HV, color="#1c3555", alpha=0.45, zorder=1)

    # ── Tie-lines visuais ──────────────────────────────────────────────
    if show_tls and tls_vis:
        shown = tls_vis[:n_tl_show]
        cmap  = plt.cm.plasma
        for k, tl in enumerate(shown):
            c = cmap(0.10 + 0.78*k / max(len(shown)-1, 1))
            ax.plot([tl["xL"], tl["yV"]], [tl["HL"], tl["HV"]],
                    color=c, lw=1.0, ls='--', alpha=0.65, zorder=2)
            ax.plot(tl["xL"], tl["HL"], 'o', color=c, ms=3.5, alpha=0.75, zorder=3)
            ax.plot(tl["yV"], tl["HV"], 'o', color=c, ms=3.5, alpha=0.75, zorder=3)
            # Label de temperatura na ponta esquerda (curva HL)
            ax.text(tl["xL"] - 0.018, tl["HL"],
                    f"{tl['T']:.0f}°",
                    fontsize=5.8, color=c, ha="right", va="center",
                    fontfamily="monospace", alpha=0.88)

    # ── Curvas HL e HV ────────────────────────────────────────────────
    ax.plot(xs, HV, color="#4fc3f7", lw=2.5, zorder=4, label="$H_V(y^*)$ vapor sat.")
    ax.plot(xs, HL, color="#ef5350", lw=2.5, zorder=4, label="$H_L(x)$ líquido sat.")
    mid = len(xs)//2
    ax.text(xs[mid]+0.02, HV[mid]+0.18, "$H_V$",
            color="#4fc3f7", fontsize=11, fontweight="bold", fontfamily="monospace")
    ax.text(xs[mid]+0.02, HL[mid]-0.45, "$H_L$",
            color="#ef5350", fontsize=11, fontweight="bold", fontfamily="monospace")

    # ── Estágios ──────────────────────────────────────────────────────
    if show_stages:
        # RETIFICAÇÃO
        for s in stages_R:
            lbl_op  = "Op. Retificação"  if s["n"]==1 else ""
            lbl_tl  = "Tie-line (Ret.)"  if s["n"]==1 else ""
            # reta operacional: do ponto atual (HL) ao ponto em HV
            ax.plot([s["op_x0"], s["op_x1"]],
                    [s["op_H0"], s["op_H1"]],
                    color="#9c67c8", lw=1.6, alpha=0.85, zorder=5, label=lbl_op)
            # tie-line termodinâmica
            ax.plot([s["tl_xL"], s["tl_yV"]],
                    [s["tl_HL"], s["tl_HV"]],
                    color="#f48fb1", lw=2.2, ls='--', zorder=6, label=lbl_tl)
            # pontos
            ax.plot(s["op_x1"], s["op_H1"], 'o',
                    color="#9c67c8", ms=7, zorder=7,
                    markeredgecolor="white", markeredgewidth=0.5)
            ax.plot(s["tl_xL"], s["tl_HL"], 'o',
                    color="#f48fb1", ms=7, zorder=7,
                    markeredgecolor="white", markeredgewidth=0.5)
            if show_annot:
                ax.text(s["tl_xL"]-0.005, s["tl_HL"]-0.45,
                        f"R{s['n']}", fontsize=8, color="#f48fb1",
                        fontweight="bold", ha="center", fontfamily="monospace")

        # ESGOTAMENTO
        for s in stages_S:
            lbl_tl = "Tie-line (Esgo.)" if s["n"]==1 else ""
            lbl_op = "Op. Esgotamento"  if s["n"]==1 else ""
            # tie-line
            ax.plot([s["tl_xL"], s["tl_yV"]],
                    [s["tl_HL"], s["tl_HV"]],
                    color="#ffb74d", lw=2.2, ls='--', zorder=6, label=lbl_tl)
            # reta operacional: do ponto em HV ao próximo ponto em HL
            ax.plot([s["op_x0"], s["op_x1"]],
                    [s["op_H0"], s["op_H1"]],
                    color="#ef6c00", lw=1.6, alpha=0.85, zorder=5, label=lbl_op)
            # pontos
            ax.plot(s["tl_yV"], s["tl_HV"], 'o',
                    color="#ffb74d", ms=7, zorder=7,
                    markeredgecolor="white", markeredgewidth=0.5)
            ax.plot(s["op_x1"], s["op_H1"], 'o',
                    color="#ef6c00", ms=7, zorder=7,
                    markeredgecolor="white", markeredgewidth=0.5)
            if show_annot:
                ax.text(s["op_x1"]+0.005, s["op_H1"]-0.45,
                        f"S{s['n']}", fontsize=8, color="#ffb74d",
                        fontweight="bold", ha="center", fontfamily="monospace")

    # ── Reta ΔR – F – ΔS ─────────────────────────────────────────────
    if abs(xD - zF) > 1e-6:
        sl   = (HD_p - HF) / (xD - zF)
        xl   = np.array([xW - 0.03, xD + 0.02])
        ax.plot(xl, HF + sl*(xl - zF),
                color="#b0b030", lw=1.2, ls='-.', alpha=0.65, zorder=3,
                label="Reta $\\Delta_R$–F–$\\Delta_S$")

    # Verticais dos produtos
    ax.axvline(xD, color="#4fc3f7", lw=0.6, ls=':', alpha=0.35, zorder=2)
    ax.axvline(xW, color="#ef5350", lw=0.6, ls=':', alpha=0.35, zorder=2)
    ax.axvline(zF, color="#26a69a", lw=0.6, ls=':', alpha=0.30, zorder=2)

    # ── Polos ─────────────────────────────────────────────────────────
    ax.plot(xD, HD_p, '*', color="#ce93d8", ms=22, zorder=9,
            markeredgecolor="white", markeredgewidth=0.6)
    ax.plot(xW, HW_p, '*', color="#ffb74d", ms=22, zorder=9,
            markeredgecolor="white", markeredgewidth=0.6)

    # ── Pontos de produto e alimentação ───────────────────────────────
    ax.plot(xD, HL_xD, 's', color="#4fc3f7", ms=10, zorder=8,
            markeredgecolor="white", markeredgewidth=0.7)
    ax.plot(xW, HL_xW, 's', color="#ef5350", ms=10, zorder=8,
            markeredgecolor="white", markeredgewidth=0.7)
    ax.plot(zF, HF,    'D', color="#26a69a",  ms=11, zorder=8,
            markeredgecolor="white", markeredgewidth=0.7)

    # ── Anotações ─────────────────────────────────────────────────────
    if show_annot:
        # polos
        ax.annotate(f"$\\Delta_R$ ({xD:.2f}, {HD_p:.2f})",
                    xy=(xD, HD_p), xytext=(xD-0.30, HD_p-0.9),
                    fontsize=8.5, color="#ce93d8", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#ce93d8", lw=1.2),
                    fontfamily="monospace", zorder=10,
                    bbox=dict(fc="#1a0a2e", ec="#ce93d8", alpha=0.88,
                              boxstyle="round,pad=0.3"))
        ax.annotate(f"$\\Delta_S$ ({xW:.2f}, {HW_p:.2f})",
                    xy=(xW, HW_p), xytext=(xW+0.07, HW_p+0.8),
                    fontsize=8.5, color="#ffb74d", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#ffb74d", lw=1.2),
                    fontfamily="monospace", zorder=10,
                    bbox=dict(fc="#2a1500", ec="#ffb74d", alpha=0.88,
                              boxstyle="round,pad=0.3"))
        # produtos
        for val, Hv, lbl, col, ddx in [
            (xD, HL_xD, f"$x_D={xD:.2f}$", "#4fc3f7", -0.14),
            (xW, HL_xW, f"$x_W={xW:.2f}$", "#ef5350", +0.06),
        ]:
            ax.annotate(lbl, xy=(val, Hv), xytext=(val+ddx, Hv+0.85),
                        fontsize=8.5, color=col, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.0),
                        fontfamily="monospace", zorder=9)
        ax.annotate(f"F  $z_F={zF:.2f}$, $q={q:.2f}$\n$H_F={HF:.2f}$ kJ/mol",
                    xy=(zF, HF), xytext=(zF+0.06, HF+0.85),
                    fontsize=8, color="#26a69a", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#26a69a", lw=1.0),
                    fontfamily="monospace", zorder=9)

    # ── Formatação ────────────────────────────────────────────────────
    ax.set_xlim(-0.03, 1.05)
    Hlo = min(float(HL.min())-1, HW_p-1.5)
    Hhi = max(float(HV.max())+1, HD_p+1.5)
    mg  = (Hhi-Hlo)*0.06
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
    for sp_ in ax.spines.values(): sp_.set_edgecolor("#1e3a5f")
    ax.grid(True, color="#1e3a5f", lw=0.6, alpha=0.8)
    ax.grid(True, which='minor', color="#141f2e", lw=0.3, alpha=0.4)
    ax.minorticks_on()

    hs, ls_ = ax.get_legend_handles_labels()
    by_l = dict(zip(ls_, hs))
    ax.legend(by_l.values(), by_l.keys(), loc="upper left",
              fontsize=8.5, framealpha=0.85, edgecolor="#1e3a5f",
              facecolor="#0d1b2a", labelcolor="#cce0f0")
    fig.tight_layout(pad=0.8)
    return fig

def make_yx(sp, xD, xW, zF, stages_R, stages_S, show_stages):
    xs  = sp["xs"]; yeq = sp["yeq"]
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    fig.patch.set_facecolor("#0d1b2a"); ax.set_facecolor("#112233")

    ax.plot([0,1],[0,1], color="#455a64", lw=1.0, ls='--')
    ax.plot(xs, yeq, color="#4fc3f7", lw=2.2, label="Equilíbrio $y^*(x)$")

    if show_stages:
        for s in stages_R:
            # degrau: vertical de (x*, x*) a (x*, y*), depois horizontal a linha op
            ax.plot([s["tl_xL"], s["tl_xL"]],
                    [s["tl_xL"], s["tl_yV"]],
                    color="#9c67c8", lw=1.3, alpha=0.85)
            ax.plot([s["tl_xL"], s["op_x0"]],
                    [s["tl_yV"], s["tl_yV"]],
                    color="#f48fb1", lw=1.3, ls='--', alpha=0.85)
            ax.plot(s["tl_xL"], s["tl_yV"], 'o',
                    color="#f48fb1", ms=6, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.4)
        for s in stages_S:
            ax.plot([s["tl_xL"], s["tl_xL"]],
                    [s["tl_xL"], s["tl_yV"]],
                    color="#ef6c00", lw=1.3, alpha=0.85)
            ax.plot([s["tl_xL"], s["op_x1"]],
                    [s["tl_yV"], s["tl_yV"]],
                    color="#ffb74d", lw=1.3, ls='--', alpha=0.85)
            ax.plot(s["tl_xL"], s["tl_yV"], 'o',
                    color="#ffb74d", ms=6, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.4)

    for val, col, lbl in [(xD,"#4fc3f7","$x_D$"),
                           (xW,"#ef5350","$x_W$"),
                           (zF,"#26a69a","$z_F$")]:
        ax.axvline(val, color=col, lw=0.7, ls=':', alpha=0.55)
        ax.text(val+0.012, 0.04, lbl, color=col, fontsize=8.5,
                fontfamily="monospace")

    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel(f"$x$  ({cA})", color="#90caf9", fontsize=10,
                  fontfamily="monospace")
    ax.set_ylabel(f"$y$  ({cA})", color="#90caf9", fontsize=10,
                  fontfamily="monospace")
    ax.set_title("Diagrama $y$-$x$", color="#4fc3f7", fontsize=11,
                 fontfamily="monospace")
    ax.tick_params(colors="#607d8b", labelsize=8)
    for sp_ in ax.spines.values(): sp_.set_edgecolor("#1e3a5f")
    ax.grid(True, color="#1e3a5f", lw=0.6, alpha=0.7)
    ax.legend(fontsize=8.5, framealpha=0.8, edgecolor="#1e3a5f",
              facecolor="#0d1b2a", labelcolor="#cce0f0")
    fig.tight_layout(pad=0.5)
    return fig

# ═════════════════════════════════════════════════════════════════════════════
#  INTERFACE
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-title">⚗ Ponchon–Savarit</div>
<div class="main-sub">Antoine → CubicSpline → Flash rigoroso · Tie-lines termodinâmicas</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sec-hdr">Sistema</div>', unsafe_allow_html=True)
    clist = list(COMPOUNDS.keys())
    cA = st.selectbox("Componente leve (A)", clist, index=0)
    cB_opts = [c for c in clist if c != cA]
    default_B = 1 if len(cB_opts) > 1 else 0
    cB = st.selectbox("Componente pesado (B)", cB_opts, index=default_B)
    P_bar = st.slider("Pressão (bar)", 0.10, 10.0, 1.013, 0.005,
                      format="%.3f bar")

    st.markdown('<div class="sec-hdr">Coluna</div>', unsafe_allow_html=True)
    xD = st.slider("Destilado  xD",   0.50, 0.999, 0.90, 0.005, format="%.3f")
    xW = st.slider("Resíduo    xW",   0.001, 0.45,  0.05, 0.005, format="%.3f")
    zF = st.slider("Alimentação zF",  0.05,  0.95,  0.45, 0.005, format="%.3f")
    R  = st.slider("Refluxo  R",      0.1,  15.0,   2.5,  0.05,  format="%.2f")
    q  = st.slider("Condição q  (1=liq.sat, 0=vap.sat)", -2.0, 2.0, 1.0, 0.05,
                   format="%.2f")

    st.markdown('<div class="sec-hdr">Tie-lines Visuais</div>', unsafe_allow_html=True)
    show_tls  = st.toggle("Mostrar tie-lines", value=True)
    n_tl_show = st.slider("Quantidade", 3, 30, 12, 1) if show_tls else 0

    st.markdown('<div class="sec-hdr">Visualização</div>', unsafe_allow_html=True)
    show_stages = st.toggle("Desenhar estágios", value=True)
    show_annot  = st.toggle("Anotações",         value=True)
    n_pts       = st.slider("Pontos nas curvas", 60, 300, 160, 20)

# ── Validação ─────────────────────────────────────────────────────────
errs = []
if xW >= zF:  errs.append("xW deve ser < zF")
if zF >= xD:  errs.append("zF deve ser < xD")
if COMPOUNDS[cA]["Tb"] >= COMPOUNDS[cB]["Tb"]:
    errs.append(f"{cA} deve ter Tb < {cB} (troque A e B)")
for e in errs:
    st.markdown(f'<div class="rcard wrn">⚠ {e}</div>', unsafe_allow_html=True)
if errs: st.stop()

# ── Cálculos ──────────────────────────────────────────────────────────
with st.spinner("Calculando flash + splines…"):
    sp = build_splines(cA, cB, P_bar, n=n_pts)

with st.spinner("Calculando polos…"):
    poles = compute_poles(sp, xD, xW, zF, R, q)

Rm = poles["Rm"]

# Tie-lines visuais: N pontos entre xW e xD usando as splines já calculadas
tls_vis = build_tielines_visual(sp, xW, xD, n=40)

# Estágios
stages_R, stages_S = [], []
if show_stages:
    with st.spinner("Calculando estágios (spline + brentq)…"):
        stages_R, stages_S = calc_stages(sp, poles, xD, xW, zF)

# ── Aviso Rmin ────────────────────────────────────────────────────────
if Rm is not None and R < Rm:
    st.markdown(
        f'<div class="rcard wrn">⛔  R = {R:.3f} < R_min ≈ {Rm:.4f} — '
        f'Separação impossível com este refluxo!</div>',
        unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────
col_plot, col_info = st.columns([2.7, 1.0])

with col_plot:
    tab_hxy, tab_yx = st.tabs(["📊 Diagrama H-x-y", "📈 Diagrama y-x"])

    with tab_hxy:
        with st.spinner("Renderizando…"):
            fig = make_plot(
                sp, poles, stages_R, stages_S, tls_vis,
                cA, cB, P_bar, xD, xW, zF, R, q,
                show_tls, n_tl_show, show_stages, show_annot)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab_yx:
        fig2 = make_yx(sp, xD, xW, zF, stages_R, stages_S, show_stages)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

# ── Painel de resultados ──────────────────────────────────────────────
with col_info:
    st.markdown('<div class="sec-hdr">Resultados</div>', unsafe_allow_html=True)

    ratio_s = f"{R/Rm:.3f}" if (Rm and Rm > 0) else "—"
    col_ratio = "#66bb6a" if (Rm and R >= Rm*1.05) else "#ffa726"

    st.markdown(f"""
<div class="rcard pr">
  <b>Polo Δ<sub>R</sub></b><br>
  x = {xD:.3f} &nbsp;|&nbsp; H'<sub>D</sub> = <b>{poles['HD_p']:.3f} kJ/mol</b>
</div>
<div class="rcard ps">
  <b>Polo Δ<sub>S</sub></b><br>
  x = {xW:.3f} &nbsp;|&nbsp; H'<sub>W</sub> = <b>{poles['HW_p']:.3f} kJ/mol</b>
</div>
<div class="rcard fd">
  <b>Alimentação F</b><br>
  z<sub>F</sub> = {zF:.3f} &nbsp;|&nbsp; q = {q:.2f}<br>
  H<sub>F</sub> = <b>{poles['HF']:.3f} kJ/mol</b>
</div>
<div class="rcard" style="border-left-color:{col_ratio}">
  <b>Refluxo</b><br>
  R<sub>min</sub> ≈ {Rm:.4f if Rm else '—'}<br>
  R / R<sub>min</sub> = <b>{ratio_s}</b>
</div>
""", unsafe_allow_html=True)

    n_R = len(stages_R); n_S = len(stages_S); n_tot = n_R + n_S
    if n_tot > 0:
        st.markdown(f"""
<div class="rcard stg">
  <b>Estágios ideais</b><br>
  Total: <b>{n_tot}</b>  (incl. refervedor)<br>
  Retificação: {n_R} &nbsp;|&nbsp; Esgotamento: {n_S}<br>
  Pratos na coluna: <b>{max(n_tot-1,0)}</b>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">Sistema</div>', unsafe_allow_html=True)
    dA = COMPOUNDS[cA]; dB = COMPOUNDS[cB]
    Tb_A = float(sp["spTbub"](1.0)); Tb_B = float(sp["spTbub"](0.0))

    st.markdown(f"""
<div class="rcard ok">
  <b>α médio (geom.)</b> = <b>{sp['alpha_mean']:.4f}</b>
</div>
<div class="rcard" style="border-left-color:{dA['col']}">
  <b>{cA}</b> (leve)<br>
  T<sub>eb</sub> = {Tb_A:.1f}°C @ {P_bar:.3f} bar<br>
  λ(25°C) = {dA['Hvap_ref']:.2f} kJ/mol
</div>
<div class="rcard" style="border-left-color:{dB['col']}">
  <b>{cB}</b> (pesado)<br>
  T<sub>eb</sub> = {Tb_B:.1f}°C @ {P_bar:.3f} bar<br>
  λ(25°C) = {dB['Hvap_ref']:.2f} kJ/mol
</div>""", unsafe_allow_html=True)

    # Tabela de estágios
    if n_tot > 0:
        st.markdown('<div class="sec-hdr">Tabela</div>', unsafe_allow_html=True)
        with st.expander("Ver detalhes dos estágios"):
            import pandas as pd
            rows = []
            for s in stages_R:
                rows.append({"N": s["n"], "Seção": "Ret.",
                             "x*": f"{s['tl_xL']:.4f}",
                             "y*": f"{s['tl_yV']:.4f}",
                             "T (°C)": f"{s['T']:.1f}",
                             "HL": f"{s['tl_HL']:.3f}",
                             "HV": f"{s['tl_HV']:.3f}"})
            for s in stages_S:
                rows.append({"N": n_R + s["n"], "Seção": "Esgo.",
                             "x*": f"{s['tl_xL']:.4f}",
                             "y*": f"{s['tl_yV']:.4f}",
                             "T (°C)": f"{s['T']:.1f}",
                             "HL": f"{s['tl_HL']:.3f}",
                             "HV": f"{s['tl_HV']:.3f}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)

# ── Notas metodológicas ───────────────────────────────────────────────
with st.expander("ℹ️ Método e modelos"):
    st.markdown(f"""
**VLE:** Lei de Raoult + Antoine  
`log₁₀(P/mmHg) = A − B/(C+T[°C])`

**Entalpias** (ref.: líquido puro a {T_REF}°C = 0 kJ/mol):  
- Líquido: `Cp_L·(T − T_ref)`  
- Vapor: `Cp_L·(T_b−T_ref) + λ(T_b) + Cp_V·(T−T_b)`  
- Misturas: regra de mistura ideal

**Splines:** `CubicSpline(xs, f)` com xs ∈ [0,1], n={n_pts} pts.  
Todas as curvas (HL, HV, y*, T_bub) compartilham a mesma abscissa x.

**Intersecções reta∩curva:** varredura de 60 pts + `brentq` no primeiro cruzamento de sinal.

**Tie-lines visuais:** {n_tl_show} pontos igualmente espaçados em x ∈ [xW, xD],  
avaliados diretamente nas splines (não requerem flash adicional).

**α médio geométrico** = exp(mean(ln(α_local))),  onde  
`α_local(x) = y*(1−x) / [x(1−y*)]`

**R_min:** extensão analítica das tie-lines (via spline) até x = xD,  
maior valor define H'_D,min → R_min = (H'_D,min − HV_top)/(HV_top − HL_xD)
""")
