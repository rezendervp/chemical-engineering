import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

# ─────────────────────────────────────────────
# Banco de componentes (Antoine NIST: log10 P/bar, T/K)
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# Banco de componentes (Antoine NIST/DDB: log10 P/bar, T/K)
# Fonte: Dortmund Data Bank / NIST Chemistry WebBook
# ─────────────────────────────────────────────
COMPONENT_GROUPS = {
    "── Aromáticos ──": None,
    "Benzeno":              {"A": 4.72583,  "B": 1660.652, "C": -1.461,    "Tb_C": 80.1,   "MW": 78.11},
    "Tolueno":              {"A": 4.14157,  "B": 1377.578, "C": -50.507,   "Tb_C": 110.6,  "MW": 92.14},
    "o-Xileno":             {"A": 4.21930,  "B": 1486.115, "C": -57.000,   "Tb_C": 144.4,  "MW": 106.17},
    "m-Xileno":             {"A": 4.20360,  "B": 1469.388, "C": -58.000,   "Tb_C": 139.1,  "MW": 106.17},
    "p-Xileno":             {"A": 4.16390,  "B": 1444.928, "C": -58.000,   "Tb_C": 138.4,  "MW": 106.17},
    "Etilbenzeno":          {"A": 4.16290,  "B": 1449.120, "C": -54.000,   "Tb_C": 136.2,  "MW": 106.17},
    "Estireno":             {"A": 4.20630,  "B": 1528.514, "C": -62.000,   "Tb_C": 145.2,  "MW": 104.15},
    "── Alcanos ──": None,
    "n-Pentano":            {"A": 4.10270,  "B": 1064.631, "C": -41.853,   "Tb_C": 36.1,   "MW": 72.15},
    "n-Hexano":             {"A": 4.00266,  "B": 1171.530, "C": -48.784,   "Tb_C": 68.7,   "MW": 86.18},
    "n-Heptano":            {"A": 4.02832,  "B": 1268.636, "C": -56.199,   "Tb_C": 98.4,   "MW": 100.20},
    "n-Octano":             {"A": 4.04867,  "B": 1355.126, "C": -63.633,   "Tb_C": 125.7,  "MW": 114.23},
    "n-Nonano":             {"A": 4.06420,  "B": 1432.526, "C": -69.700,   "Tb_C": 150.8,  "MW": 128.26},
    "Ciclohexano":          {"A": 4.12830,  "B": 1295.030, "C": -58.100,   "Tb_C": 80.7,   "MW": 84.16},
    "Metilciclohexano":     {"A": 4.03540,  "B": 1345.177, "C": -58.300,   "Tb_C": 100.9,  "MW": 98.19},
    "── Álcoois ──": None,
    "Metanol":              {"A": 5.20277,  "B": 1581.341, "C": -33.500,   "Tb_C": 64.7,   "MW": 32.04},
    "Etanol":               {"A": 5.24671,  "B": 1598.673, "C": -46.424,   "Tb_C": 78.4,   "MW": 46.07},
    "n-Propanol":           {"A": 5.24944,  "B": 1667.465, "C": -44.700,   "Tb_C": 97.2,   "MW": 60.10},
    "i-Propanol":           {"A": 5.12530,  "B": 1480.919, "C": -50.953,   "Tb_C": 82.4,   "MW": 60.10},
    "n-Butanol":            {"A": 5.07758,  "B": 1566.023, "C": -58.355,   "Tb_C": 117.7,  "MW": 74.12},
    "── Cetonas e ésteres ──": None,
    "Acetona":              {"A": 4.42448,  "B": 1312.253, "C": -32.445,   "Tb_C": 56.1,   "MW": 58.08},
    "MEK (butanona)":       {"A": 4.37980,  "B": 1354.073, "C": -43.000,   "Tb_C": 79.6,   "MW": 72.11},
    "Acetato de etila":     {"A": 4.22809,  "B": 1245.702, "C": -55.189,   "Tb_C": 77.1,   "MW": 88.11},
    "Acetato de n-butila":  {"A": 4.27790,  "B": 1375.285, "C": -64.300,   "Tb_C": 126.1,  "MW": 116.16},
    "── Outros ──": None,
    "Água":                 {"A": 5.11564,  "B": 1687.537, "C": -42.980,   "Tb_C": 100.0,  "MW": 18.02},
    "Clorofórmio":          {"A": 4.20775, "B": 1349.291, "C": -53.466, "Tb_C": 61.2, "MW": 119.38},  
    "Diclorometano":        {"A": 4.20889,  "B": 1125.938, "C": -24.064,   "Tb_C": 39.8,   "MW": 84.93},
    "Acetonitrila":         {"A": 4.52500,  "B": 1432.380, "C": -42.613,   "Tb_C": 81.6,   "MW": 41.05},
    "Ácido acético":        {"A": 4.75080,  "B": 1522.540, "C": -45.764,   "Tb_C": 117.9,  "MW": 60.05},
    "Dioxano":              {"A": 4.57096,  "B": 1429.810, "C": -39.630,   "Tb_C": 101.3,  "MW": 88.11},
    "── Personalizado ──": None,
    "Personalizado":        {"A": None,    "B": None,     "C": None,      "Tb_C": None,   "MW": None},
}

COMPONENTS  = {k: v for k, v in COMPONENT_GROUPS.items() if v is not None}
COMP_OPTIONS = list(COMPONENT_GROUPS.keys())

# ─────────────────────────────────────────────
# Banco de parâmetros NRTL (A12, A21 em J/mol; alpha adimensional)
# Fonte: DECHEMA / Smith, Van Ness & Abbott
# Convenção: tau_ij = A_ij / (R * T),  R = 8.314 J/(mol·K)
# ─────────────────────────────────────────────
NRTL_PARAMS = {
    # chave: (comp1, comp2) — sempre em ordem alfabética para busca bidirecional
    ("Etanol",  "Água"):           {"A12": 3458.3, "A21": -53.6,  "alpha": 0.30},
    ("Metanol", "Água"):           {"A12": 2726.4, "A21":  937.3,  "alpha": 0.30},
    ("Acetona", "Água"):           {"A12": 3768.3, "A21": 1116.0,  "alpha": 0.47},
    ("Acetona", "Clorofórmio"):    {"A12":  -2051.00,"A21": -1704., "alpha": 0.30},
    #("Acetona", "Clorofórmio"):     {"A12": 5456, "A21": 2514, "alpha": 0.30},
    #("Acetona", "Clorofórmio"): {"A12": -7128, "A21": -8581, "alpha": 0.30},
   
    ("Acetona", "Metanol"):        {"A12":  980.6, "A21":  481.2,  "alpha": 0.30},
    ("Etanol",  "Tolueno"):        {"A12": 4108.0, "A21": 1021.0,  "alpha": 0.47},
    ("Metanol", "Acetato de etila"):{"A12": 1556.0,"A21": 1733.0,  "alpha": 0.47},
    #("Benzeno", "Etanol"):         {"A12": 5058, "A21": 954, "alpha": 0.3},
    ("Benzeno", "Etanol"):        {"A12": 3040.0, "A21": 1396.0, "alpha": 0.47},  
}

def get_nrtl_params(name1, name2):
    """Busca parâmetros NRTL para o par (name1, name2), em qualquer ordem."""
    key  = (name1, name2)
    keyT = (name2, name1)
    if key in NRTL_PARAMS:
        p = NRTL_PARAMS[key]
        return p["A12"], p["A21"], p["alpha"], False   # False = não inverteu
    elif keyT in NRTL_PARAMS:
        p = NRTL_PARAMS[keyT]
        return p["A21"], p["A12"], p["alpha"], True    # True  = inverteu A12/A21
    return None, None, None, False

# ─────────────────────────────────────────────
# Funções de cálculo — termodinâmica
# ─────────────────────────────────────────────
def pvap(A, B, C, T_K):
    """Pressão de vapor em bar — Antoine NIST (log10 P/bar, T/K)"""
    return 10.0 ** (A - B / (T_K + C))

def nrtl_gamma(x1, T_K, A12, A21, alpha):
    """
    Coeficientes de atividade pelo modelo NRTL (sistema bicomponente).
    A12, A21 : parâmetros de interação [J/mol]
    alpha     : parâmetro de não-aleatoriedade (tipicamente 0.20–0.47)
    Retorna   : (gamma1, gamma2)
    """
    R = 8.314  # J/(mol·K)
    x2 = 1.0 - x1

    # Parâmetros dependentes de T
    tau12 = A12 / (R * T_K)
    tau21 = A21 / (R * T_K)
    G12   = math.exp(-alpha * tau12)
    G21   = math.exp(-alpha * tau21)

    # Denominadores
    denom1 = x1 + x2 * G21   # para ln(gamma1)
    denom2 = x2 + x1 * G12   # para ln(gamma2)

    # Evitar divisão por zero nos extremos puros
    eps = 1e-12
    if abs(denom1) < eps: denom1 = eps
    if abs(denom2) < eps: denom2 = eps

    lng1 = x2**2 * (tau21 * (G21 / denom1)**2 + tau12 * G12 / denom2**2)
    lng2 = x1**2 * (tau12 * (G12 / denom2)**2 + tau21 * G21 / denom1**2)

    return math.exp(lng1), math.exp(lng2)

def bubble_T(x1, A1, B1, C1, A2, B2, C2, P_bar, T_init=None,
             modelo="Raoult", A12=0.0, A21=0.0, alpha_nrtl=0.30):
    """
    Temperatura de bolha para composição x1 via Newton-Raphson.
    Suporta modelos: 'Raoult' (ideal) ou 'NRTL'.
    """
    if T_init is None:
        # Chute ótimo: inverter Antoine analiticamente para cada puro
        Tb1 = B1 / (A1 - math.log10(P_bar)) - C1  # retorna T em K
        Tb2 = B2 / (A2 - math.log10(P_bar)) - C2  # retorna T em K
        T   = x1 * Tb1 + (1.0 - x1) * Tb2  # K
    else:
        T = T_init

    for _ in range(200):
        pb1 = pvap(A1, B1, C1, T)
        pb2 = pvap(A2, B2, C2, T)

        if modelo == "NRTL":
            g1, g2 = nrtl_gamma(x1, T, A12, A21, alpha_nrtl)
        else:
            g1, g2 = 1.0, 1.0

        Pcalc = x1 * g1 * pb1 + (1.0 - x1) * g2 * pb2
        err   = Pcalc - P_bar
        if abs(err) < 1e-7:
            break

        # Derivada numérica (diferença finita progressiva, dT = 0.01 K)
        dT   = 0.01
        pb1d = pvap(A1, B1, C1, T + dT)
        pb2d = pvap(A2, B2, C2, T + dT)
        if modelo == "NRTL":
            g1d, g2d = nrtl_gamma(x1, T + dT, A12, A21, alpha_nrtl)
        else:
            g1d, g2d = 1.0, 1.0

        dPdT = (x1 * g1d * pb1d + (1.0 - x1) * g2d * pb2d - Pcalc) / dT
        if abs(dPdT) < 1e-12:
            break
        T -= err / dPdT

    return T

def calc_vle(A1, B1, C1, A2, B2, C2, P_bar, n_points,
             modelo="Raoult", A12=0.0, A21=0.0, alpha_nrtl=0.30):
    """
    Retorna DataFrame com x1, y1, T_K, T_C, P1sat, P2sat, gamma1, gamma2, alpha_rel.
    Funciona para modelos Raoult e NRTL.
    """
    x_arr  = np.linspace(0, 1, n_points)
    rows   = []
    T_prev = None

    for x1 in x_arr:
        T = bubble_T(x1, A1, B1, C1, A2, B2, C2, P_bar,
                     T_init=T_prev, modelo=modelo,
                     A12=A12, A21=A21, alpha_nrtl=alpha_nrtl)
        T_prev = T

        P1s = pvap(A1, B1, C1, T)
        P2s = pvap(A2, B2, C2, T)

        if modelo == "NRTL":
            g1, g2 = nrtl_gamma(x1, T, A12, A21, alpha_nrtl)
        else:
            g1, g2 = 1.0, 1.0

        y1    = x1 * g1 * P1s / P_bar
        y1    = max(0.0, min(1.0, y1))          # clamp numérico
        alpha = (g1 * P1s) / (g2 * P2s) if P2s > 0 and g2 > 0 else np.nan

        rows.append({
            "x₁":             round(float(x1),  6),
            "y₁":             round(float(y1),  6),
            "T (K)":          round(float(T),   4),
            "T (°C)":         round(float(T - 273.15), 4),
            "P₁ˢᵃᵗ (bar)":   round(float(P1s), 6),
            "P₂ˢᵃᵗ (bar)":   round(float(P2s), 6),
            "γ₁":             round(float(g1),  6),
            "γ₂":             round(float(g2),  6),
            "α₁₂ efetivo":   round(float(alpha),4),
        })

    return pd.DataFrame(rows)

def detectar_azeotropo(df):
    """
    Detecta azeótropo: ponto onde (y1 - x1) muda de sinal.
    Retorna (x_az, y_az, T_az) ou None.
    """
    diff = df["y₁"].values - df["x₁"].values
    for i in range(len(diff) - 1):
        if diff[i] * diff[i+1] < 0:
            # interpolação linear entre os dois pontos
            x0, x1_ = df["x₁"].iloc[i], df["x₁"].iloc[i+1]
            d0, d1  = diff[i], diff[i+1]
            frac    = -d0 / (d1 - d0)
            x_az    = x0 + frac * (x1_ - x0)
            T_az    = df["T (°C)"].iloc[i] + frac * (df["T (°C)"].iloc[i+1] - df["T (°C)"].iloc[i])
            return x_az, x_az, T_az   # no azeótropo y = x
    return None

# ─────────────────────────────────────────────
# Layout Streamlit
# ─────────────────────────────────────────────
st.set_page_config(page_title="Equilíbrio VLE Bicomponente v2", layout="wide")
st.title("⚗️ Equilíbrio Líquido-Vapor — Sistema Bicomponente")
st.caption("Lei de Raoult (ideal) ou NRTL (não ideal) + Equação de Antoine (NIST) — log₁₀(P/bar) = A − B/(T+C), T em K")

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.header("Configuração")

    # ── Modelo termodinâmico ──
    st.subheader("Modelo termodinâmico")
    modelo = st.radio(
        "Fase líquida",
        ["Raoult (ideal)", "NRTL (não ideal)"],
        index=0,
        help="Raoult: γᵢ = 1. NRTL: γᵢ calculado por modelo de energia de Gibbs em excesso."
    )
    usar_nrtl = modelo == "NRTL (não ideal)"

    st.divider()

    # ── Componentes ──
    st.subheader("Light Key (mais volátil)")
    comp1_name = st.selectbox(
        "Light Key", COMP_OPTIONS, index=1,
        format_func=lambda x: x if COMPONENT_GROUPS[x] is not None else x
    )
    if COMPONENT_GROUPS.get(comp1_name) is None:
        comp1_name = "Benzeno"
    d1 = COMPONENTS[comp1_name]
    if comp1_name == "Personalizado":
        c1_label = st.text_input("Nome do componente 1", "Comp A")
        A1 = st.number_input("A₁", value=5.40768, format="%.5f")
        B1 = st.number_input("B₁", value=1322.882, format="%.3f")
        C1 = st.number_input("C₁", value=-53.015, format="%.3f")
    else:
        c1_label = comp1_name
        A1, B1, C1 = d1["A"], d1["B"], d1["C"]
        st.markdown(f"**A** = {A1} | **B** = {B1} | **C** = {C1}")
        st.caption(f"T_eb (1 atm) = {d1['Tb_C']} °C")

    st.subheader("Heavy Key (menos volátil)")
    comp2_name = st.selectbox(
        "Heavy Key", COMP_OPTIONS, index=2,
        format_func=lambda x: x if COMPONENT_GROUPS[x] is not None else x
    )
    if COMPONENT_GROUPS.get(comp2_name) is None:
        comp2_name = "Tolueno"
    d2 = COMPONENTS[comp2_name]
    if comp2_name == "Personalizado":
        c2_label = st.text_input("Nome do componente 2", "Comp B")
        A2 = st.number_input("A₂", value=5.46600, format="%.5f")
        B2 = st.number_input("B₂", value=1576.079, format="%.3f")
        C2 = st.number_input("C₂", value=-47.814, format="%.3f")
    else:
        c2_label = comp2_name
        A2, B2, C2 = d2["A"], d2["B"], d2["C"]
        st.markdown(f"**A** = {A2} | **B** = {B2} | **C** = {C2}")
        st.caption(f"T_eb (1 atm) = {d2['Tb_C']} °C")

    st.divider()

    # ── Parâmetros NRTL ──
    A12_val = A21_val = alpha_val = 0.0
    if usar_nrtl:
        st.subheader("Parâmetros NRTL")
        A12_db, A21_db, alpha_db, _ = get_nrtl_params(c1_label, c2_label)
        par_encontrado = A12_db is not None

        if par_encontrado:
            st.success(f"✅ Parâmetros encontrados no banco interno para {c1_label} / {c2_label}")
        else:
            st.warning("⚠️ Par não encontrado no banco. Insira os parâmetros manualmente.")

        A12_val   = st.number_input("A₁₂ (J/mol)", value=float(A12_db or 1000.0), format="%.1f",
                                    help="Parâmetro de interação do componente 1 com o 2")
        A21_val   = st.number_input("A₂₁ (J/mol)", value=float(A21_db or 1000.0), format="%.1f",
                                    help="Parâmetro de interação do componente 2 com o 1")
        alpha_val = st.number_input("α (não-aleatoriedade)", value=float(alpha_db or 0.30),
                                    min_value=0.10, max_value=0.60, step=0.01, format="%.2f",
                                    help="Tipicamente 0.20–0.47. Valores negativos indicam sistemas associativos.")
        st.caption("Fonte: DECHEMA / Smith, Van Ness & Abbott")

    st.divider()

    # ── Condições ──
    st.subheader("Condições de operação")
    P_atm = st.number_input("Pressão de operação (atm)", value=1.0,
                            min_value=0.1, max_value=20.0, step=0.1)
    P_bar = P_atm* 1.01325 # coeficientes Antoine em atm, T em K
    n_points = st.slider("Número de pontos", min_value=11, max_value=101, value=21, step=5)

    calcular = st.button("🔄 Calcular", type="primary", use_container_width=True)

# ── Aviso sobre limitação do modelo (fixo, sempre visível) ──
with st.expander("ℹ️ Sobre os modelos implementados e suas limitações", expanded=False):
    st.markdown("""
**Fase líquida**
- **Raoult (ideal):** coeficiente de atividade γᵢ = 1 para todos os componentes.
  Adequado para misturas de compostos quimicamente similares (ex.: benzeno/tolueno, alcanos lineares).
- **NRTL:** γᵢ calculado pelo modelo de energia de Gibbs em excesso de Renon & Prausnitz (1968).
  Captura desvios positivos e negativos da idealidade e permite predizer azeótropos.

**Fase vapor — vapor ideal (φ̂ᵢᵛ = 1)**

Ambos os modelos tratam a fase vapor como **gás ideal**, ou seja, o coeficiente de fugacidade
φ̂ᵢᵛ = 1. Esta é uma boa aproximação para pressões abaixo de aproximadamente **5 atm**.
Acima disso, as correções de fugacidade no vapor tornam-se relevantes e seria necessário
um modelo de equação de estado (ex.: Peng-Robinson, SRK) para a fase vapor.
A não-idealidade mais importante em sistemas a baixa pressão está na **fase líquida**,
que é justamente o que o modelo NRTL captura.

**Parâmetros NRTL**
Os parâmetros A₁₂, A₂₁ e α são ajustados a dados experimentais e dependem do par de
componentes. Utilize preferencialmente valores do banco DECHEMA ou da literatura.
    """)



# ── Cálculo ───────────────────────────────────
if calcular or True:
    try:
        # Verificar componente 1 mais volátil
        Tb1 = bubble_T(1.0, A1, B1, C1, A2, B2, C2, P_bar)
        Tb2 = bubble_T(0.0, A1, B1, C1, A2, B2, C2, P_bar)
        if Tb1 > Tb2:
            st.warning(
                f"⚠️ {c1_label} parece menos volátil que {c2_label} nessa pressão. "
                "Reordenação automática aplicada para manter Light Key como mais volátil."
            )
            (A1, B1, C1, A2, B2, C2) = (A2, B2, C2, A1, B1, C1)
            (c1_label, c2_label)     = (c2_label, c1_label)
            (A12_val, A21_val)       = (A21_val, A12_val)   # inverte NRTL também
            st.info("Componentes reordenados automaticamente.")
            Tb1 = bubble_T(1.0, A1, B1, C1, A2, B2, C2, P_bar)
            Tb2 = bubble_T(0.0, A1, B1, C1, A2, B2, C2, P_bar)

        # Calcular VLE com modelo selecionado
        df = calc_vle(A1, B1, C1, A2, B2, C2, P_bar, n_points,
                      modelo="NRTL" if usar_nrtl else "Raoult",
                      A12=A12_val, A21=A21_val, alpha_nrtl=alpha_val)

        # Calcular também Raoult puro (para comparação no gráfico y-x quando NRTL ativo)
        df_raoult = None
        if usar_nrtl:
            df_raoult = calc_vle(A1, B1, C1, A2, B2, C2, P_bar, n_points,
                                 modelo="Raoult")

        # Detectar azeótropo
        az = detectar_azeotropo(df)

        # ── Métricas ──────────────────────────
        alpha_col = "α₁₂ efetivo"
        alpha_mean = df[alpha_col].mean()
        T_bolha_puro1 = df["T (°C)"].iloc[-1]
        T_bolha_puro2 = df["T (°C)"].iloc[0]

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("P operação",        f"{P_atm:.2f} atm")
        col2.metric(f"T_eb {c1_label}",  f"{T_bolha_puro1:.1f} °C")
        col3.metric(f"T_eb {c2_label}",  f"{T_bolha_puro2:.1f} °C")
        col4.metric("α₁₂ efetivo médio", f"{alpha_mean:.3f}")
        col5.metric("Azeótropo",
                    f"x = {az[0]:.3f} | T = {az[2]:.1f} °C" if az else "Não detectado")

        if az:
            st.error(
                f"⚠️ **Azeótropo detectado** em x₁ ≈ y₁ ≈ **{az[0]:.3f}** "
                f"e T ≈ **{az[2]:.1f} °C** — a destilação convencional não ultrapassa esta composição."
            )

        # ── Gráficos ──────────────────────────
        fig = plt.figure(figsize=(14, 10))
        fig.patch.set_facecolor("#0E1117")
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

        AXBG   = "#1A1D23"
        GRID   = "#2A2D35"
        TXT    = "#E0DDD6"
        BLUE   = "#4A9EDF"
        ORANGE = "#E87040"
        GREEN  = "#5BB87A"
        GRAY   = "#888880"
        PURPLE = "#A07EDB"

        def style_ax(ax, title, xlabel, ylabel):
            ax.set_facecolor(AXBG)
            ax.tick_params(colors=TXT, labelsize=9)
            ax.xaxis.label.set_color(TXT)
            ax.yaxis.label.set_color(TXT)
            ax.title.set_color(TXT)
            ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(color=GRID, linewidth=0.6, linestyle="--")
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID)

        x  = df["x₁"].values
        y  = df["y₁"].values
        T  = df["T (°C)"].values
        P1 = df["P₁ˢᵃᵗ (bar)"].values
        P2 = df["P₂ˢᵃᵗ (bar)"].values
        al = df[alpha_col].values
        g1 = df["γ₁"].values
        g2 = df["γ₂"].values

        label_modelo = "NRTL" if usar_nrtl else "Raoult"

        # 1) Diagrama y–x
        ax1 = fig.add_subplot(gs[0, 0])
        style_ax(ax1, f"Diagrama y–x  ({c1_label} / {c2_label})",
                 f"x₁  [{c1_label}]", f"y₁  [{c1_label}]")
        # Curva Raoult de referência (tracejada) quando NRTL ativo
        if usar_nrtl and df_raoult is not None:
            ax1.plot(df_raoult["x₁"].values, df_raoult["y₁"].values,
                     color=BLUE, lw=1.5, ls="--", label="Raoult (ref.)", alpha=0.6)
        ax1.plot(x, y, color=ORANGE if usar_nrtl else BLUE,
                 lw=2, label=label_modelo)
        ax1.plot([0, 1], [0, 1], color=GRAY, lw=1, ls="--", label="y = x")
        ax1.scatter(x[1:-1], y[1:-1],
                    color=ORANGE if usar_nrtl else BLUE, s=18, zorder=5)
        if az:
            ax1.scatter([az[0]], [az[1]], color="red", s=80, zorder=10,
                        marker="*", label=f"Azeótropo x≈{az[0]:.3f}")
        ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
        ax1.legend(fontsize=8, facecolor=AXBG, labelcolor=TXT, edgecolor=GRID)

        # 2) Diagrama T–x–y
        ax2 = fig.add_subplot(gs[0, 1])
        style_ax(ax2, f"Diagrama T–x–y  ({P_atm:.2f} atm)",
                 f"Fração molar [{c1_label}]", "T (°C)")
        ax2.plot(x, T, color=ORANGE, lw=2, label="Curva bolha (T–x)")
        ax2.plot(y, T, color=BLUE,   lw=2, ls="--", label="Curva orvalho (T–y)")
        ax2.fill_betweenx(T, x, y, alpha=0.08, color=GREEN)
        if az:
            ax2.axhline(az[2], color="red", lw=0.8, ls=":", alpha=0.7)
            ax2.axvline(az[0], color="red", lw=0.8, ls=":", alpha=0.7)
        ax2.legend(fontsize=8, facecolor=AXBG, labelcolor=TXT, edgecolor=GRID)

        # 3) Coeficientes de atividade (NRTL) OU Pressões de vapor (Raoult)
        ax3 = fig.add_subplot(gs[1, 0])
        if usar_nrtl:
            style_ax(ax3, "Coeficientes de atividade γ vs x₁",
                     f"x₁  [{c1_label}]", "γ")
            ax3.plot(x, g1, color=ORANGE, lw=2, label=f"γ₁  {c1_label}")
            ax3.plot(x, g2, color=BLUE,   lw=2, label=f"γ₂  {c2_label}")
            ax3.axhline(1.0, color=GRAY, lw=1, ls=":", label="γ = 1 (ideal)")
            ax3.legend(fontsize=8, facecolor=AXBG, labelcolor=TXT, edgecolor=GRID)
        else:
            style_ax(ax3, "Pressões de vapor vs Temperatura",
                     "T (°C)", "Pˢᵃᵗ (bar)")
            ax3.plot(T, P1, color=ORANGE, lw=2, label=f"P₁ˢᵃᵗ  {c1_label}")
            ax3.plot(T, P2, color=BLUE,   lw=2, label=f"P₂ˢᵃᵗ  {c2_label}")
            ax3.axhline(P_bar, color=GRAY, lw=1, ls=":",
                        label=f"P op = {P_bar:.3f} bar")
            ax3.legend(fontsize=8, facecolor=AXBG, labelcolor=TXT, edgecolor=GRID)

        # 4) Volatilidade relativa efetiva vs x
        ax4 = fig.add_subplot(gs[1, 1])
        style_ax(ax4,
                 "Volatilidade relativa efetiva α₁₂ vs x₁"
                 + (" (γᵢ·Pᵢˢᵃᵗ)" if usar_nrtl else " (Pᵢˢᵃᵗ)"),
                 f"x₁  [{c1_label}]", "α₁₂ efetivo")
        ax4.plot(x, al, color=GREEN, lw=2)
        ax4.axhline(np.nanmean(al), color=GRAY, lw=1, ls="--",
                    label=f"α médio = {np.nanmean(al):.3f}")
        ax4.axhline(1.0, color=ORANGE, lw=0.8, ls=":",
                    label="α = 1 (sem separação / azeótropo)")
        ax4.scatter(x[1:-1], al[1:-1], color=GREEN, s=18, zorder=5)
        if az:
            ax4.axvline(az[0], color="red", lw=0.8, ls=":", alpha=0.7,
                        label=f"Azeótropo x≈{az[0]:.3f}")
        ax4.legend(fontsize=8, facecolor=AXBG, labelcolor=TXT, edgecolor=GRID)

        # Título geral
        fig.suptitle(
            f"Equilíbrio VLE — {c1_label} / {c2_label}   |   "
            f"P = {P_atm:.2f} atm   |   {label_modelo} + Antoine (NIST)",
            color=TXT, fontsize=12, fontweight="bold", y=0.98
        )

        st.pyplot(fig)
        plt.close(fig)

        # ── Tabela de dados ───────────────────
        st.subheader("📋 Tabela de dados de equilíbrio")
        fmt = {
            "x₁": "{:.4f}", "y₁": "{:.4f}",
            "T (K)": "{:.2f}", "T (°C)": "{:.2f}",
            "P₁ˢᵃᵗ (bar)": "{:.5f}", "P₂ˢᵃᵗ (bar)": "{:.5f}",
            "γ₁": "{:.4f}", "γ₂": "{:.4f}",
            "α₁₂ efetivo": "{:.4f}",
        }
        st.dataframe(df.style.format(fmt), use_container_width=True, height=300)

        # ── Downloads ─────────────────────────
        st.subheader("📥 Exportar dados")
        col_a, col_b = st.columns(2)

        csv_full = df.to_csv(index=False, header=False, sep=";", decimal=",")
        col_a.download_button(
            "⬇️ CSV completo (separador ;)",
            data=csv_full.encode("utf-8"),
            file_name=f"VLE_{label_modelo}_{c1_label}_{c2_label}_{P_atm}atm.csv",
            mime="text/csv",
            use_container_width=True,
        )

        df_xy  = df[["x₁", "y₁"]].rename(columns={"x₁": "x", "y₁": "y"})
        csv_xy = df_xy.to_csv(index=False, header=False, sep=",", decimal=".")
        col_b.download_button(
            "⬇️ x,y para McCabe-Thiele (CSV)",
            data=csv_xy.encode("utf-8"),
            file_name=f"xy_{label_modelo}_{c1_label}_{c2_label}_{P_atm}atm.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # ── Info parâmetros ───────────────────
        with st.expander("ℹ️ Parâmetros utilizados no cálculo"):
            ac = pd.DataFrame({
                "Componente":  [c1_label, c2_label],
                "A (Antoine)": [A1, A2],
                "B (Antoine)": [B1, B2],
                "C (Antoine)": [C1, C2],
                "Referência":  ["NIST WebBook", "NIST WebBook"],
            })
            st.dataframe(ac, use_container_width=True)
            if usar_nrtl:
                st.markdown(
                    f"**NRTL:** A₁₂ = {A12_val:.1f} J/mol | "
                    f"A₂₁ = {A21_val:.1f} J/mol | α = {alpha_val:.2f} | "
                    f"R = 8.314 J/(mol·K) | τᵢⱼ = Aᵢⱼ / (R·T)"
                )

        with st.expander("📌 Como usar o arquivo x,y no app McCabe-Thiele"):
            st.markdown("""
1. Clique em **⬇️ x,y para McCabe-Thiele** acima.
2. No app McCabe-Thiele, use a opção **"Carregar dados de equilíbrio (CSV)"**.
3. O arquivo tem duas colunas: `x` e `y`, separadas por vírgula, ponto decimal.
4. O app interpolará a curva VLE a partir desses pontos para construir o diagrama.
            """)

    except Exception as e:
        st.error(f"Erro no cálculo: {e}")
        st.exception(e)
