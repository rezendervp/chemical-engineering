"""
╔══════════════════════════════════════════════════════════════════════╗
║        MÉTODO DE PONCHON-SAVARIT — App Streamlit Interativo         ║
║        Destilação Binária com Flash Calculations Rigorosas           ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyArrowPatch
#from scipy.optimize import brentq, fsolve
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURAÇÃO DA PÁGINA
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Ponchon-Savarit",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS customizado
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Fundo escuro azul-petróleo */
.stApp {
    background: #0d1b2a;
    color: #e8f4f8;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #112233;
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * {
    color: #cce0f0 !important;
}

/* Título principal */
.main-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.1rem;
    font-weight: 600;
    color: #4fc3f7;
    letter-spacing: -1px;
    margin-bottom: 0;
    line-height: 1.1;
}
.main-sub {
    font-size: 0.95rem;
    color: #78909c;
    margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace;
}

/* Cartões de resultado */
.result-card {
    background: #1a2f45;
    border: 1px solid #1e3a5f;
    border-left: 4px solid #4fc3f7;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 6px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.88rem;
}
.result-card.polo-r  { border-left-color: #ab47bc; }
.result-card.polo-s  { border-left-color: #ff8f00; }
.result-card.feed    { border-left-color: #26a69a; }
.result-card.stages  { border-left-color: #ef5350; }
.result-card.warning { border-left-color: #ffa726; background: #2a1f10; }

/* Separador de seção */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 3px;
    color: #4fc3f7;
    text-transform: uppercase;
    margin: 20px 0 8px 0;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 4px;
}

/* Métrica */
.metric-box {
    background: #142236;
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
    border: 1px solid #1e3a5f;
}
.metric-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #4fc3f7;
}
.metric-lab {
    font-size: 0.75rem;
    color: #78909c;
    margin-top: 2px;
}

/* Botão */
.stButton > button {
    background: #1565c0;
    color: white;
    border: none;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    padding: 8px 20px;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #1976d2;
}

/* Slider */
.stSlider label {
    font-size: 0.8rem !important;
    color: #90caf9 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Selectbox */
.stSelectbox label {
    font-size: 0.8rem !important;
    color: #90caf9 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #78909c;
}
.stTabs [aria-selected="true"] {
    color: #4fc3f7 !important;
    border-bottom-color: #4fc3f7 !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    color: #90caf9 !important;
}

/* Info box */
.stAlert {
    border-radius: 6px;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  BANCO DE DADOS TERMODINÂMICOS
# ═══════════════════════════════════════════════════════════════════════

# Parâmetros de Antoine: log10(P_sat/mmHg) = A - B/(C + T[°C])
# Entalpias de vaporização: kJ/mol a T_ref (25°C), com dependência de T
# Cp líquido: kJ/(mol·K)  |  Cp vapor: kJ/(mol·K)
# T_eb: ponto de ebulição a 1 atm (°C)  |  Tc: temperatura crítica (°C)

COMPOUNDS = {
    "Benzeno": {
        "A": 6.90565, "B": 1211.033, "C": 220.790,
        "Hvap_ref": 30.72, "dHvap_dT": -0.060,    # kJ/mol
        "CpL": 0.1350, "CpV": 0.0830,              # kJ/(mol·K)
        "Tc": 288.9, "Tb": 80.1, "M": 78.11,
        "color": "#42a5f5",
    },
    "Tolueno": {
        "A": 6.95334, "B": 1343.943, "C": 219.377,
        "Hvap_ref": 33.18, "dHvap_dT": -0.055,
        "CpL": 0.1572, "CpV": 0.1030,
        "Tc": 318.6, "Tb": 110.6, "M": 92.14,
        "color": "#ef5350",
    },
    "Etanol": {
        "A": 8.11220, "B": 1592.864, "C": 226.184,
        "Hvap_ref": 38.56, "dHvap_dT": -0.045,
        "CpL": 0.1120, "CpV": 0.0780,
        "Tc": 243.1, "Tb": 78.37, "M": 46.07,
        "color": "#26a69a",
    },
    "Água": {
        "A": 8.07131, "B": 1730.630, "C": 233.426,
        "Hvap_ref": 44.00, "dHvap_dT": -0.042,
        "CpL": 0.0754, "CpV": 0.0340,
        "Tc": 374.1, "Tb": 100.0, "M": 18.02,
        "color": "#29b6f6",
    },
    "n-Heptano": {
        "A": 6.89386, "B": 1264.370, "C": 216.640,
        "Hvap_ref": 31.77, "dHvap_dT": -0.063,
        "CpL": 0.2243, "CpV": 0.1620,
        "Tc": 267.0, "Tb": 98.4, "M": 100.20,
        "color": "#ab47bc",
    },
    "n-Hexano": {
        "A": 6.87601, "B": 1171.170, "C": 224.408,
        "Hvap_ref": 28.85, "dHvap_dT": -0.065,
        "CpL": 0.1952, "CpV": 0.1430,
        "Tc": 234.7, "Tb": 68.7, "M": 86.18,
        "color": "#ff8f00",
    },
    "Acetona": {
        "A": 7.02447, "B": 1161.000, "C": 224.000,
        "Hvap_ref": 31.27, "dHvap_dT": -0.058,
        "CpL": 0.1249, "CpV": 0.0740,
        "Tc": 235.1, "Tb": 56.1, "M": 58.08,
        "color": "#66bb6a",
    },
    "Metanol": {
        "A": 7.89750, "B": 1474.080, "C": 229.130,
        "Hvap_ref": 37.43, "dHvap_dT": -0.048,
        "CpL": 0.0812, "CpV": 0.0480,
        "Tc": 239.4, "Tb": 64.7, "M": 32.04,
        "color": "#ffa726",
    },
}

# ═══════════════════════════════════════════════════════════════════════
#  FUNÇÕES TERMODINÂMICAS
# ═══════════════════════════════════════════════════════════════════════

def Psat(comp, T_C):
    """Pressão de vapor (mmHg) via Antoine. T em °C."""
    d = COMPOUNDS[comp]
    return 10.0 ** (d["A"] - d["B"] / (d["C"] + T_C))

def Psat_bar(comp, T_C):
    return Psat(comp, T_C) / 750.062  # mmHg → bar

def Hvap(comp, T_C):
    """Calor latente de vaporização (kJ/mol) na temperatura T."""
    d = COMPOUNDS[comp]
    return max(d["Hvap_ref"] + d["dHvap_dT"] * (T_C - 25.0), 1.0)

def bubble_T(comp_A, comp_B, x, P_bar):
    """Temperatura de bolha (°C) para líquido de composição x a pressão P."""
    P_mmHg = P_bar * 750.062
    def obj(T):
        yA = x * Psat(comp_A, T) / P_mmHg
        yB = (1 - x) * Psat(comp_B, T) / P_mmHg
        return yA + yB - 1.0
    # Limites: pontos de ebulição dos puros ± margem
    dA = COMPOUNDS[comp_A]; dB = COMPOUNDS[comp_B]
    Tlo = min(dA["Tb"], dB["Tb"]) - 5
    Thi = max(dA["Tb"], dB["Tb"]) + 5
    try:
        return brentq(obj, Tlo, Thi, xtol=1e-5)
    except Exception:
        return None

def dew_T(comp_A, comp_B, y, P_bar):
    """Temperatura de orvalho (°C) para vapor de composição y a pressão P."""
    P_mmHg = P_bar * 750.062
    def obj(T):
        xA = y * P_mmHg / Psat(comp_A, T)
        xB = (1 - y) * P_mmHg / Psat(comp_B, T)
        return xA + xB - 1.0
    dA = COMPOUNDS[comp_A]; dB = COMPOUNDS[comp_B]
    Tlo = min(dA["Tb"], dB["Tb"]) - 5
    Thi = max(dA["Tb"], dB["Tb"]) + 5
    try:
        return brentq(obj, Tlo, Thi, xtol=1e-5)
    except Exception:
        return None

def y_from_bubble(comp_A, comp_B, x, T_C, P_bar):
    """y* em equilíbrio com líquido x na temperatura de bolha T."""
    P_mmHg = P_bar * 750.062
    return x * Psat(comp_A, T_C) / P_mmHg

# ─── Entalpias (referência: líquido puro a 0°C = 0) ─────────────────────────

T_REF = 0.0  # °C, referência de entalpia

def HL_pure(comp, T_C):
    """Entalpia do líquido puro (kJ/mol)."""
    d = COMPOUNDS[comp]
    return d["CpL"] * (T_C - T_REF)  # kJ/mol

def HV_pure(comp, T_C):
    """Entalpia do vapor puro (kJ/mol)."""
    d = COMPOUNDS[comp]
    Tb = d["Tb"]
    # Aquece líquido até Tb, vaporiza, aquece vapor até T
    hL_at_Tb = d["CpL"] * (Tb - T_REF)
    lam = Hvap(comp, Tb)
    hV_at_Tb = hL_at_Tb + lam
    hV = hV_at_Tb + d["CpV"] * (T_C - Tb)
    return hV

def H_mix_L(comp_A, comp_B, x, T_C):
    """Entalpia da mistura líquida (kJ/mol) — ideal (sem calor de mistura)."""
    return x * HL_pure(comp_A, T_C) + (1 - x) * HL_pure(comp_B, T_C)

def H_mix_V(comp_A, comp_B, y, T_C):
    """Entalpia da mistura vapor (kJ/mol) — ideal."""
    return y * HV_pure(comp_A, T_C) + (1 - y) * HV_pure(comp_B, T_C)

# ─── Curvas de equilíbrio HL(x) e HV(y) ─────────────────────────────────────

@st.cache_data(show_spinner=False)
def build_equilibrium_curves(comp_A, comp_B, P_bar, n_pts=80):
    """
    Retorna (x_arr, y_arr, HL_arr, HV_arr, T_bub_arr, T_dew_arr)
    usando flash calculations na temperatura de bolha e orvalho.
    """
    x_arr = np.linspace(0.0, 1.0, n_pts)
    y_arr    = np.zeros(n_pts)
    HL_arr   = np.zeros(n_pts)
    HV_arr   = np.zeros(n_pts)
    T_bub_arr = np.zeros(n_pts)
    T_dew_arr = np.zeros(n_pts)

    for i, x in enumerate(x_arr):
        Tb = bubble_T(comp_A, comp_B, x, P_bar)
        if Tb is None:
            y_arr[i] = np.nan; HL_arr[i] = np.nan; HV_arr[i] = np.nan
            continue
        y = y_from_bubble(comp_A, comp_B, x, Tb, P_bar)
        y = np.clip(y, 0.0, 1.0)
        y_arr[i]     = y
        HL_arr[i]    = H_mix_L(comp_A, comp_B, x, Tb)    # kJ/mol
        HV_arr[i]    = H_mix_V(comp_A, comp_B, y, Tb)
        T_bub_arr[i] = Tb

        Td = dew_T(comp_A, comp_B, y, P_bar)
        T_dew_arr[i] = Td if Td is not None else Tb

    return x_arr, y_arr, HL_arr, HV_arr, T_bub_arr, T_dew_arr

@st.cache_data(show_spinner=False)
def build_isotherms(comp_A, comp_B, P_bar, T_list):
    """
    Para cada T em T_list, retorna (x_bub, HL_bub, y_dew, HV_dew)
    que define a isoterma no diagrama H-x-y.
    Também calcula pontos bifásicos intermediários.
    """
    isotherms = []
    P_mmHg = P_bar * 750.062

    for T in T_list:
        # Ponto de bolha: pressão de bolha a essa T e composição variável
        # Encontra x tal que Pbub(x,T) = P (se existir)
        # PsatA*x + PsatB*(1-x) = P  → x = (P - PsatB)/(PsatA - PsatB)
        PsA = Psat(comp_A, T)
        PsB = Psat(comp_B, T)
        if abs(PsA - PsB) < 1e-10:
            continue

        x_bub = (P_mmHg - PsB) / (PsA - PsB)
        if not (0 <= x_bub <= 1):
            continue  # isoterma fora do range

        y_bub = x_bub * PsA / P_mmHg  # y na bolha

        # Ponto de orvalho: x = P*y / PsatA,  x*(PsA-PsB) + PsB = P → x já calculado
        # Na realidade, para uma isoterma T fixo:
        #   Bolha: (x_bub, y_bub) — composição do vapor que começa a se formar
        #   Orvalho: outra composição
        # Para a isoterma no diagrama Hxy, traçamos a linha horizontal em T
        # do ponto de bolha (x_bub, HL(x_bub,T)) ao ponto de orvalho (y_orv, HV(y_orv,T))
        # y_orv: 1/sum(y/Psat) = P  → para orvalho: x_A = y*P/PsA, x_B=(1-y)*P/PsB
        # x_A+x_B=1 → y*(1/PsA - 1/PsB) = 1/PsB - 1/P → y = (1/PsB-1/P)/(1/PsB-1/PsA)
        y_dew = (1/PsB - 1/P_mmHg) / (1/PsB - 1/PsA)
        if not (0 <= y_dew <= 1):
            continue

        x_dew = y_dew * P_mmHg / PsA  # x em equilíbrio com y_dew

        HL_bub = H_mix_L(comp_A, comp_B, x_bub, T)
        HV_dew = H_mix_V(comp_A, comp_B, y_dew, T)

        # Pontos intermediários bifásicos (regra da alavanca)
        n_iso = 12
        fracs = np.linspace(0, 1, n_iso)
        iso_x = np.zeros(n_iso)
        iso_H = np.zeros(n_iso)
        for j, frac in enumerate(fracs):
            # frac = fração de vapor (0=líquido, 1=vapor)
            # Composição global z varia de x_bub a y_dew
            z = x_bub + frac * (y_dew - x_bub)
            # Entalpia pela mistura bifásica
            H = (1 - frac) * H_mix_L(comp_A, comp_B, x_bub, T) + \
                frac * H_mix_V(comp_A, comp_B, y_dew, T)
            iso_x[j] = z
            iso_H[j] = H

        isotherms.append({
            "T": T,
            "x_bub": x_bub, "HL_bub": HL_bub,
            "y_dew": y_dew, "HV_dew": HV_dew,
            "iso_x": iso_x, "iso_H": iso_H,
        })

    return isotherms

# ═══════════════════════════════════════════════════════════════════════
#  ALGORITMO PONCHON-SAVARIT
# ═══════════════════════════════════════════════════════════════════════

def interp_HL(x_val, x_arr, HL_arr):
    """Interpola HL na composição x_val."""
    return float(np.interp(x_val, x_arr, HL_arr))

def interp_HV_from_y(y_val, y_arr, HV_arr, x_arr):
    """Interpola HV na composição y_val."""
    # y_arr pode não ser monotônico em raros casos
    idx = np.argsort(y_arr)
    return float(np.interp(y_val, y_arr[idx], HV_arr[idx]))

def find_y_on_HV_line(xL, HLx, xD, HD_p, x_arr, y_arr, HV_arr, side="rectification"):
    """
    Dado o ponto (xL, HLx) e o polo (xD, HD_p),
    encontra o ponto (y1, HV(y1)) onde a reta polo→(xL,HLx)
    corta a curva HV(y).

    Para retificação: polo = Delta_R = (xD, HD_p), xL < xD
    Para esgotamento: polo = Delta_S = (xW, HW_p), xL > xW
    """
    if abs(xD - xL) < 1e-9:
        return None, None

    slope = (HD_p - HLx) / (xD - xL)

    def residual(y):
        # H na reta (parametrizado por y como abscissa)
        H_line = HLx + slope * (y - xL)
        H_curve = interp_HV_from_y(y, y_arr, HV_arr, x_arr)
        return H_curve - H_line

    # Busca raiz no intervalo (xL, 1) para retificação
    # ou (0, xL) para esgotamento
    if side == "rectification":
        lo, hi = xL + 1e-4, 1.0 - 1e-4
    else:
        lo, hi = 1e-4, xL - 1e-4

    try:
        # Verifica se há mudança de sinal
        vals = [residual(v) for v in np.linspace(lo, hi, 40)]
        for k in range(len(vals) - 1):
            if vals[k] * vals[k+1] < 0:
                y1 = brentq(residual,
                            lo + k*(hi-lo)/39,
                            lo + (k+1)*(hi-lo)/39,
                            xtol=1e-6)
                return y1, interp_HV_from_y(y1, y_arr, HV_arr, x_arr)
    except Exception:
        pass
    return None, None

def find_x_tieline(y1, x_arr, y_arr, HL_arr):
    """
    Dado y1 (vapor), encontra x* em equilíbrio (ponto na curva HL).
    Usa interpolação na curva y(x) para inverter.
    """
    idx = np.argsort(x_arr)
    xs = x_arr[idx]; ys = y_arr[idx]

    def res(x):
        return float(np.interp(x, xs, ys)) - y1

    lo, hi = xs[0] + 1e-5, xs[-1] - 1e-5
    try:
        vals = [res(v) for v in np.linspace(lo, hi, 40)]
        for k in range(len(vals) - 1):
            if vals[k] * vals[k+1] < 0:
                x1 = brentq(res, lo+k*(hi-lo)/39, lo+(k+1)*(hi-lo)/39, xtol=1e-6)
                return x1, interp_HL(x1, x_arr, HL_arr)
    except Exception:
        pass
    return None, None

def ponchon_savarit(x_arr, y_arr, HL_arr, HV_arr, xD, xW, zF, HD_p, HW_p, q, max_stages=30):
    """
    Executa o algoritmo de Ponchon-Savarit completo.
    Retorna lista de estágios com coordenadas para plot.
    """
    HL_xD = interp_HL(xD, x_arr, HL_arr)

    stages = []
    in_rect = True
    x_cur   = xD
    HL_cur  = HL_xD

    # Ponto de alimentação no diagrama
    HL_zF = interp_HL(zF, x_arr, HL_arr)
    HV_zF = interp_HV_from_y(zF, y_arr, HV_arr, x_arr)
    HF    = (1 - q) * HV_zF + q * HL_zF   # HF = HL quando q=1, HV quando q=0

    for n in range(1, max_stages + 1):
        polo_x = xD if in_rect else xW
        polo_H = HD_p if in_rect else HW_p
        side   = "rectification" if in_rect else "stripping"

        # Passo 1: reta de polo por (x_cur, HL_cur) → interseção com HV
        y1, HV1 = find_y_on_HV_line(x_cur, HL_cur, polo_x, polo_H,
                                     x_arr, y_arr, HV_arr, side=side)
        if y1 is None:
            break

        # Passo 2: tie-line de (y1, HV1) → (x1, HL1)
        x1, HL1 = find_x_tieline(y1, x_arr, y_arr, HL_arr)
        if x1 is None:
            break

        stages.append({
            "n": n,
            "section": "R" if in_rect else "S",
            "x_cur": x_cur, "HL_cur": HL_cur,   # ponto de partida
            "y1": y1,  "HV1": HV1,               # ponto no vapor (reta de polo)
            "x1": x1,  "HL1": HL1,               # ponto no líquido (tie-line)
            "polo_x": polo_x, "polo_H": polo_H,
        })

        # Critério de troca de seção
        if in_rect and x1 <= zF:
            in_rect = False

        # Critério de parada
        if x1 <= xW + 1e-3:
            break

        x_cur  = x1
        HL_cur = HL1

    return stages, HF

def compute_poles(comp_A, comp_B, P_bar, xD, xW, zF, R, q,
                  x_arr, y_arr, HL_arr, HV_arr):
    """Calcula os polos Delta_R e Delta_S."""
    # HL e HV nas condições do topo
    HL_xD = interp_HL(xD, x_arr, HL_arr)

    # Vapor topo: composição y em equilíbrio com xD
    idx_xD = np.argmin(np.abs(x_arr - xD))
    y_top  = y_arr[idx_xD]
    HV_top = interp_HV_from_y(y_top, y_arr, HV_arr, x_arr)

    # Polo retificação
    HD_p = (R + 1) * HV_top - R * HL_xD

    # Alimentação
    HL_zF = interp_HL(zF, x_arr, HL_arr)
    HV_zF = interp_HV_from_y(zF, y_arr, HV_arr, x_arr)
    HF    = (1 - q) * HV_zF + q * HL_zF

    # Polo esgotamento: colinear com Delta_R e F
    if abs(xD - zF) < 1e-9:
        HW_p = HF
    else:
        slope = (HD_p - HF) / (xD - zF)
        HW_p  = HF + slope * (xW - zF)

    # Refluxo mínimo: tie-line que passa por xD e dá maior HD_p
    # Encontra interseção das tie-lines estendidas com x=xD
    Rm = None
    HD_p_min = None
    for i in range(len(x_arr) - 1):
        xL_i = x_arr[i]
        y_i  = y_arr[i]
        HL_i = HL_arr[i]
        HV_i = HV_arr[i]
        if abs(y_i - xL_i) < 1e-6:
            continue
        # Equação da tie-line: H = HL_i + (HV_i - HL_i)/(y_i - xL_i) * (comp - xL_i)
        if abs(y_i - xL_i) < 1e-9:
            continue
        slope_tl = (HV_i - HL_i) / (y_i - xL_i)
        H_at_xD  = HL_i + slope_tl * (xD - xL_i)
        if H_at_xD > (HD_p_min or -1e10) and xL_i < xD:
            HD_p_min = H_at_xD
            Rm_cand  = (HD_p_min - HV_top) / (HV_top - HL_xD)
            Rm       = max(Rm_cand, 0.0)

    return HD_p, HW_p, HF, Rm, HL_xD, HV_top, y_top

# ═══════════════════════════════════════════════════════════════════════
#  FUNÇÃO DE PLOT
# ═══════════════════════════════════════════════════════════════════════

def make_hxy_plot(
    comp_A, comp_B, P_bar,
    x_arr, y_arr, HL_arr, HV_arr, T_bub_arr,
    xD, xW, zF, R, q,
    HD_p, HW_p, HF, HL_xD, HV_top,
    stages,
    show_isotherms, isotherms, n_iso,
    show_equil_pts, show_poles_line,
    show_stages, show_annotations,
    figsize=(11, 8.5),
):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#0d1b2a")

    # Cores do sistema
    CA = COMPOUNDS[comp_A]["color"]
    CB = COMPOUNDS[comp_B]["color"]

    # ── Região bifásica preenchida ─────────────────────────────────────
    ax.fill_between(x_arr, HL_arr, HV_arr,
                    color="#1e3a5f", alpha=0.55, zorder=1,
                    label="Região bifásica")

    # ── Isotermas ──────────────────────────────────────────────────────
    if show_isotherms and isotherms:
        sel = isotherms[:n_iso]
        cmap = plt.cm.plasma
        for k, iso in enumerate(sel):
            frac = k / max(len(sel) - 1, 1)
            col  = cmap(0.15 + 0.7 * frac)
            # Linha horizontal da isoterma (bifásica)
            ax.plot(iso["iso_x"], iso["iso_H"],
                    color=col, lw=1.2, ls='--', alpha=0.75, zorder=2)
            # Pontos de bolha e orvalho
            ax.plot(iso["x_bub"], iso["HL_bub"], 'o',
                    color=col, ms=5, zorder=3, alpha=0.85)
            ax.plot(iso["y_dew"], iso["HV_dew"], 'o',
                    color=col, ms=5, zorder=3, alpha=0.85)
            # Rótulo de temperatura
            ax.text(iso["x_bub"] - 0.035,
                    iso["HL_bub"],
                    f"{iso['T']:.0f}°C",
                    fontsize=6.5, color=col, alpha=0.85,
                    ha="right", va="center",
                    fontfamily="monospace")

    # ── Curvas HL e HV ────────────────────────────────────────────────
    ax.plot(x_arr, HV_arr, color="#4fc3f7", lw=2.5, zorder=4,
            label=f"$H_V(y)$ — vapor sat.")
    ax.plot(x_arr, HL_arr, color="#ef5350", lw=2.5, zorder=4,
            label=f"$H_L(x)$ — líquido sat.")

    # Rótulos das curvas
    mid = len(x_arr) // 2
    ax.text(x_arr[mid] + 0.04, HV_arr[mid] + 0.3, "$H_V(y)$",
            color="#4fc3f7", fontsize=11, fontweight="bold",
            fontfamily="monospace", zorder=5)
    ax.text(x_arr[mid] + 0.04, HL_arr[mid] - 0.6, "$H_L(x)$",
            color="#ef5350", fontsize=11, fontweight="bold",
            fontfamily="monospace", zorder=5)

    # ── Pontos de produto na curva HL ─────────────────────────────────
    if show_equil_pts:
        HL_xW = interp_HL(xW, x_arr, HL_arr)
        for val, lbl, col in [
            (xD, f"$x_D={xD:.3f}$", "#4fc3f7"),
            (xW, f"$x_W={xW:.3f}$", "#ef5350"),
        ]:
            ax.axvline(val, color=col, lw=0.6, ls=':', alpha=0.35, zorder=2)

        ax.plot(xD, HL_xD, 's', color="#4fc3f7", ms=9, zorder=6,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.plot(xW, HL_xW, 's', color="#ef5350", ms=9, zorder=6,
                markeredgecolor="white", markeredgewidth=0.8)

        if show_annotations:
            ax.annotate(f"$x_D={xD:.2f}$",
                        xy=(xD, HL_xD), xytext=(xD - 0.18, HL_xD + 0.9),
                        fontsize=9, color="#4fc3f7", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="#4fc3f7",
                                        lw=1.2, connectionstyle="arc3,rad=0.1"),
                        fontfamily="monospace", zorder=7)
            ax.annotate(f"$x_W={xW:.2f}$",
                        xy=(xW, HL_xW), xytext=(xW + 0.08, HL_xW + 0.9),
                        fontsize=9, color="#ef5350", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="#ef5350",
                                        lw=1.2, connectionstyle="arc3,rad=-0.1"),
                        fontfamily="monospace", zorder=7)

    # ── Polo Delta_R ──────────────────────────────────────────────────
    ax.plot(xD, HD_p, '*', color="#ab47bc", ms=20, zorder=8,
            markeredgecolor="white", markeredgewidth=0.6)
    if show_annotations:
        ax.annotate(f"$\\Delta_R$\n({xD:.2f}, {HD_p:.2f})",
                    xy=(xD, HD_p), xytext=(xD - 0.30, HD_p - 1.2),
                    fontsize=8.5, color="#ab47bc", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#ab47bc", lw=1.3),
                    fontfamily="monospace", zorder=9,
                    bbox=dict(fc="#1a0a2e", ec="#ab47bc", alpha=0.85,
                              boxstyle="round,pad=0.3"))

    # ── Polo Delta_S ──────────────────────────────────────────────────
    ax.plot(xW, HW_p, '*', color="#ff8f00", ms=20, zorder=8,
            markeredgecolor="white", markeredgewidth=0.6)
    if show_annotations:
        ax.annotate(f"$\\Delta_S$\n({xW:.2f}, {HW_p:.2f})",
                    xy=(xW, HW_p), xytext=(xW + 0.09, HW_p + 1.2),
                    fontsize=8.5, color="#ff8f00", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#ff8f00", lw=1.3),
                    fontfamily="monospace", zorder=9,
                    bbox=dict(fc="#2a1500", ec="#ff8f00", alpha=0.85,
                              boxstyle="round,pad=0.3"))

    # ── Ponto de alimentação F ────────────────────────────────────────
    ax.plot(zF, HF, 'D', color="#26a69a", ms=11, zorder=8,
            markeredgecolor="white", markeredgewidth=0.8)
    if show_annotations:
        ax.annotate(f"$F$ ({zF:.2f}, {HF:.2f})\n$q={q:.2f}$",
                    xy=(zF, HF), xytext=(zF + 0.08, HF + 0.9),
                    fontsize=8.5, color="#26a69a", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#26a69a", lw=1.2),
                    fontfamily="monospace", zorder=9,
                    bbox=dict(fc="#0a2020", ec="#26a69a", alpha=0.85,
                              boxstyle="round,pad=0.3"))

    # ── Reta Delta_R – F – Delta_S ────────────────────────────────────
    if show_poles_line:
        xs_line = np.array([xW - 0.02, xD + 0.01])
        if abs(xD - zF) > 1e-6:
            slope = (HD_p - HF) / (xD - zF)
            ys_line = HF + slope * (xs_line - zF)
            ax.plot(xs_line, ys_line, color="#b0b030", lw=1.3,
                    ls='-.', alpha=0.65, zorder=3,
                    label="Reta $\\Delta_R$–$F$–$\\Delta_S$")

    # ── Estágios ──────────────────────────────────────────────────────
    if show_stages and stages:
        for stg in stages:
            col_op  = "#7e57c2" if stg["section"] == "R" else "#ffa726"
            col_tie = "#ef5350" if stg["section"] == "R" else "#ffa726"

            # Reta operacional: polo → ponto na HV
            ax.plot([stg["polo_x"], stg["y1"]],
                    [stg["polo_H"], stg["HV1"]],
                    color=col_op, lw=1.5, alpha=0.75, zorder=5)

            # Tie-line: ponto HV → ponto HL
            ax.plot([stg["y1"], stg["x1"]],
                    [stg["HV1"], stg["HL1"]],
                    color=col_tie, lw=2.0, ls='--', zorder=6)

            # Pontos
            ax.plot(stg["y1"], stg["HV1"], 'o',
                    color=col_op, ms=7, zorder=7,
                    markeredgecolor="white", markeredgewidth=0.5)
            ax.plot(stg["x1"], stg["HL1"], 'o',
                    color=col_tie, ms=7, zorder=7,
                    markeredgecolor="white", markeredgewidth=0.5)

            # Número do estágio
            if show_annotations:
                ax.text(stg["x1"] - 0.01, stg["HL1"] - 0.55,
                        str(stg["n"]),
                        fontsize=8.5, color=col_tie,
                        fontweight="bold", ha="center",
                        fontfamily="monospace", zorder=8)

    # ── Decoração ─────────────────────────────────────────────────────
    ax.set_xlim(-0.03, 1.05)

    # Range dinâmico de H
    all_H = np.concatenate([HL_arr[~np.isnan(HL_arr)],
                            HV_arr[~np.isnan(HV_arr)]])
    H_lo = min(all_H.min() - 1.0, HW_p - 1.0)
    H_hi = max(all_H.max() + 1.0, HD_p + 1.0)
    margin = (H_hi - H_lo) * 0.06
    ax.set_ylim(H_lo - margin, H_hi + margin)

    ax.set_xlabel(f"Fração molar de {comp_A}  ($x$ ou $y$)",
                  color="#90caf9", fontsize=12, labelpad=8,
                  fontfamily="monospace")
    ax.set_ylabel("Entalpia molar $H$  (kJ/mol)",
                  color="#90caf9", fontsize=12, labelpad=8,
                  fontfamily="monospace")
    ax.set_title(f"Diagrama $H$-$x$-$y$  ·  {comp_A} / {comp_B}  ·  $P$ = {P_bar:.3f} bar",
                 color="#4fc3f7", fontsize=13, fontweight="bold",
                 fontfamily="monospace", pad=12)

    ax.tick_params(colors="#607d8b", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a5f")

    ax.grid(True, color="#1e3a5f", lw=0.7, alpha=0.8, zorder=0)
    ax.grid(True, which='minor', color="#162030", lw=0.4, alpha=0.5)
    ax.minorticks_on()

    legend = ax.legend(loc="upper left", fontsize=8.5,
                       framealpha=0.85, fancybox=True,
                       edgecolor="#1e3a5f",
                       facecolor="#0d1b2a",
                       labelcolor="#cce0f0")

    fig.tight_layout(pad=0.8)
    return fig

def make_yx_plot(x_arr, y_arr, comp_A, comp_B, P_bar, xD, xW, zF):
    """Diagrama y-x auxiliar."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#112233")

    ax.plot([0, 1], [0, 1], color="#455a64", lw=1, ls='--')
    ax.plot(x_arr, y_arr, color="#4fc3f7", lw=2.2, label="Equilíbrio")

    for val, col, lbl in [(xD, "#4fc3f7", "$x_D$"),
                           (xW, "#ef5350", "$x_W$"),
                           (zF, "#26a69a", "$z_F$")]:
        ax.axvline(val, color=col, lw=0.8, ls=':', alpha=0.6)
        ax.text(val + 0.01, 0.03, lbl, color=col, fontsize=8,
                fontfamily="monospace")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(f"$x$ ({comp_A})", color="#90caf9", fontsize=10,
                  fontfamily="monospace")
    ax.set_ylabel(f"$y$ ({comp_A})", color="#90caf9", fontsize=10,
                  fontfamily="monospace")
    ax.set_title("Diagrama $y$-$x$", color="#4fc3f7", fontsize=11,
                 fontfamily="monospace")
    ax.tick_params(colors="#607d8b", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
    ax.grid(True, color="#1e3a5f", lw=0.6, alpha=0.7)
    fig.tight_layout(pad=0.5)
    return fig

# ═══════════════════════════════════════════════════════════════════════
#  INTERFACE STREAMLIT
# ═══════════════════════════════════════════════════════════════════════

# ── Cabeçalho ─────────────────────────────────────────────────────────
st.markdown("""
<div class="main-title">⚗ Ponchon–Savarit</div>
<div class="main-sub">Destilação Binária · Flash Calculations · Diagrama H-x-y Interativo</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">Sistema</div>', unsafe_allow_html=True)

    compound_list = list(COMPOUNDS.keys())
    comp_A = st.selectbox("Componente leve (A)", compound_list, index=0)
    comp_B_opts = [c for c in compound_list if c != comp_A]
    comp_B = st.selectbox("Componente pesado (B)", comp_B_opts,
                          index=1 if len(comp_B_opts) > 1 else 0)

    P_bar = st.slider("Pressão (bar)", 0.10, 10.0, 1.013, 0.01,
                      format="%.3f bar")

    st.markdown('<div class="section-header">Especificações da Coluna</div>',
                unsafe_allow_html=True)

    xD = st.slider("Destilado  xD", 0.50, 0.999, 0.90, 0.005, format="%.3f")
    xW = st.slider("Resíduo  xW", 0.001, 0.40, 0.05, 0.005, format="%.3f")
    zF = st.slider("Alimentação  zF", 0.05, 0.95, 0.45, 0.01, format="%.3f")
    R  = st.slider("Refluxo  R = L/D", 0.5, 10.0, 2.5, 0.1, format="%.1f")

    q_label = st.selectbox("Condição da alimentação (q)",
                            ["1.0 — Líquido saturado",
                             "0.0 — Vapor saturado",
                             "Personalizado"])
    if q_label.startswith("1"):
        q = 1.0
    elif q_label.startswith("0"):
        q = 0.0
    else:
        q = st.slider("Valor de q", -0.5, 1.5, 0.8, 0.05, format="%.2f")

    st.markdown('<div class="section-header">Isotermas</div>',
                unsafe_allow_html=True)

    show_isotherms = st.toggle("Mostrar isotermas", value=True)
    if show_isotherms:
        n_iso = st.slider("Quantidade de isotermas", 3, 20, 8, 1)
    else:
        n_iso = 0

    st.markdown('<div class="section-header">Opções Visuais</div>',
                unsafe_allow_html=True)

    show_equil_pts  = st.toggle("Pontos de produto (xD, xW)", value=True)
    show_poles_line = st.toggle("Reta ΔR – F – ΔS", value=True)
    show_stages     = st.toggle("Estágios de destilação", value=True)
    show_annotations = st.toggle("Anotações e rótulos", value=True)

    st.markdown('<div class="section-header">Resolução</div>',
                unsafe_allow_html=True)
    n_pts = st.slider("Pontos nas curvas", 40, 200, 100, 10)

# ── Validação ─────────────────────────────────────────────────────────
errors = []
if xW >= zF:
    errors.append("⚠ xW deve ser menor que zF")
if zF >= xD:
    errors.append("⚠ zF deve ser menor que xD")
if COMPOUNDS[comp_A]["Tb"] >= COMPOUNDS[comp_B]["Tb"]:
    errors.append(f"⚠ {comp_A} (Tb={COMPOUNDS[comp_A]['Tb']:.1f}°C) deve ter Tb < {comp_B} (Tb={COMPOUNDS[comp_B]['Tb']:.1f}°C)")

if errors:
    for e in errors:
        st.markdown(f'<div class="result-card warning">{e}</div>',
                    unsafe_allow_html=True)
    if "xW" in errors[0] or "zF" in errors[0]:
        st.stop()

# ── Cálculos principais ───────────────────────────────────────────────
with st.spinner("Calculando curvas de equilíbrio…"):
    x_arr, y_arr, HL_arr, HV_arr, T_bub_arr, T_dew_arr = \
        build_equilibrium_curves(comp_A, comp_B, P_bar, n_pts=n_pts)

# Remover NaN
mask = ~(np.isnan(HL_arr) | np.isnan(HV_arr) | np.isnan(y_arr))
x_arr  = x_arr[mask];  y_arr  = y_arr[mask]
HL_arr = HL_arr[mask]; HV_arr = HV_arr[mask]
T_bub_arr = T_bub_arr[mask]

# Isotermas
if show_isotherms:
    T_min = T_bub_arr.min()
    T_max = T_bub_arr.max()
    T_list = np.linspace(T_min + 0.5, T_max - 0.5, 30)
    with st.spinner("Calculando isotermas…"):
        isotherms_data = build_isotherms(comp_A, comp_B, P_bar, T_list)
else:
    isotherms_data = []

# Polos
HD_p, HW_p, HF, Rm, HL_xD, HV_top, y_top = compute_poles(
    comp_A, comp_B, P_bar, xD, xW, zF, R, q,
    x_arr, y_arr, HL_arr, HV_arr
)

# Verificação Rmin
if Rm is not None and R < Rm:
    st.markdown(
        f'<div class="result-card warning">⛔ R = {R:.2f} < R_min ≈ {Rm:.3f} '
        f'— Separação impossível com este refluxo!</div>',
        unsafe_allow_html=True
    )

# Estágios
stages = []
if show_stages:
    with st.spinner("Calculando estágios…"):
        stages, HF_calc = ponchon_savarit(
            x_arr, y_arr, HL_arr, HV_arr,
            xD, xW, zF, HD_p, HW_p, q
        )

# ── Layout principal ──────────────────────────────────────────────────
col_plot, col_info = st.columns([2.6, 1.0])

with col_plot:
    tab_hxy, tab_yx = st.tabs(["📊 Diagrama H-x-y", "📈 Diagrama y-x"])

    with tab_hxy:
        with st.spinner("Renderizando…"):
            fig = make_hxy_plot(
                comp_A, comp_B, P_bar,
                x_arr, y_arr, HL_arr, HV_arr, T_bub_arr,
                xD, xW, zF, R, q,
                HD_p, HW_p, HF, HL_xD, HV_top,
                stages,
                show_isotherms, isotherms_data, n_iso,
                show_equil_pts, show_poles_line,
                show_stages, show_annotations,
                figsize=(11, 8.5),
            )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab_yx:
        fig2 = make_yx_plot(x_arr, y_arr, comp_A, comp_B, P_bar, xD, xW, zF)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

with col_info:
    st.markdown('<div class="section-header">Resultados</div>',
                unsafe_allow_html=True)

    # Polos
    st.markdown(f"""
<div class="result-card polo-r">
  <b>Polo Δ<sub>R</sub></b><br>
  x = {xD:.3f}<br>
  H'<sub>D</sub> = <b>{HD_p:.3f} kJ/mol</b>
</div>
<div class="result-card polo-s">
  <b>Polo Δ<sub>S</sub></b><br>
  x = {xW:.3f}<br>
  H'<sub>W</sub> = <b>{HW_p:.3f} kJ/mol</b>
</div>
<div class="result-card feed">
  <b>Alimentação F</b><br>
  z<sub>F</sub> = {zF:.3f},  q = {q:.2f}<br>
  H<sub>F</sub> = <b>{HF:.3f} kJ/mol</b>
</div>
""", unsafe_allow_html=True)

    # Refluxo mínimo
    if Rm is not None:
        ratio = R / Rm if Rm > 0 else float('inf')
        rcolor = "#26a69a" if ratio >= 1.1 else "#ef5350"
        st.markdown(f"""
<div class="result-card" style="border-left-color:{rcolor}">
  <b>Refluxo</b><br>
  R<sub>min</sub> ≈ {Rm:.4f}<br>
  R / R<sub>min</sub> = <b>{ratio:.3f}</b>
</div>
""", unsafe_allow_html=True)

    # Estágios
    if stages:
        n_total = len(stages)
        n_ret   = sum(1 for s in stages if s["section"] == "R")
        n_esgo  = n_total - n_ret
        feed_st = next((s["n"] for s in stages if s["section"] == "S"), n_ret + 1)

        st.markdown(f"""
<div class="result-card stages">
  <b>Estágios ideais</b><br>
  Total: <b>{n_total}</b>  (incl. refervedor)<br>
  Pratos na coluna: <b>{n_total - 1}</b><br>
  Retificação: {n_ret}  |  Esgotamento: {n_esgo}<br>
  Prato de alim.: <b>{feed_st}</b> (do topo)
</div>
""", unsafe_allow_html=True)

    # Propriedades dos compostos
    st.markdown('<div class="section-header">Compostos</div>',
                unsafe_allow_html=True)

    dA = COMPOUNDS[comp_A]; dB = COMPOUNDS[comp_B]
    Tb_A = bubble_T(comp_A, comp_B, 1.0, P_bar)
    Tb_B = bubble_T(comp_A, comp_B, 0.0, P_bar)

    st.markdown(f"""
<div class="result-card" style="border-left-color:{dA['color']}">
  <b>{comp_A}</b> (leve)<br>
  T<sub>eb</sub> @ {P_bar:.3f} bar = {Tb_A:.1f if Tb_A else '?'}°C<br>
  λ (25°C) = {dA['Hvap_ref']:.2f} kJ/mol
</div>
<div class="result-card" style="border-left-color:{dB['color']}">
  <b>{comp_B}</b> (pesado)<br>
  T<sub>eb</sub> @ {P_bar:.3f} bar = {Tb_B:.1f if Tb_B else '?'}°C<br>
  λ (25°C) = {dB['Hvap_ref']:.2f} kJ/mol
</div>
""", unsafe_allow_html=True)

    # Tabela de estágios
    if stages:
        st.markdown('<div class="section-header">Tabela de Estágios</div>',
                    unsafe_allow_html=True)
        with st.expander("Ver detalhes", expanded=False):
            rows = []
            for s in stages:
                rows.append({
                    "N": s["n"],
                    "Seção": "Ret." if s["section"] == "R" else "Esgo.",
                    "x (líq.)": f"{s['x1']:.4f}",
                    "y (vap.)": f"{s['y1']:.4f}",
                    "HL (kJ/mol)": f"{s['HL1']:.3f}",
                    "HV (kJ/mol)": f"{s['HV1']:.3f}",
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)

# ── Rodapé info ───────────────────────────────────────────────────────
with st.expander("ℹ️ Sobre o método e modelos utilizados", expanded=False):
    st.markdown(f"""
**Equilíbrio vapor-líquido:** Lei de Raoult (sistema ideal)
- Pressões de vapor via equação de **Antoine** (log₁₀ P = A - B/(C+T))
- Temperatura de bolha e orvalho por método iterativo (Brentq)

**Entalpias:**
- Referência: líquido puro a {T_REF}°C → H = 0
- Líquido: H_L = Cp_L · (T - T_ref)
- Vapor: H_V = Cp_L · (T_eb - T_ref) + λ(T_eb) + Cp_V · (T - T_eb)
- Misturas: regra de mistura ideal (sem calor de mistura)

**Parâmetros Antoine:** fonte NIST/Perry's (T em °C, P em mmHg)

**Polo de retificação:** H'_D = (R+1)·H_V(y_topo) - R·H_L(x_D)

**Polo de esgotamento:** colinearidade ΔR – F – ΔS (balanço global)

**Algoritmo de estágios:** reta polo → HV (interseção numérica) → tie-line (inversão da curva y(x))
    """)

    st.markdown(f"**Compostos disponíveis:** {', '.join(COMPOUNDS.keys())}")
