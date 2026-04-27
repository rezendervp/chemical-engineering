import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import io

# ─────────────────────────────────────────────
# Banco de componentes (Antoine NIST: log10 P/bar, T/K)
# ─────────────────────────────────────────────

# Grupos para exibição organizada no selectbox
COMPONENT_GROUPS = {
    "── Aromáticos ──": None,
    "Benzeno":              {"A": 5.40768, "B": 1322.882, "C": -53.015,  "Tb_C": 80.1,   "MW": 78.11},
    "Tolueno":              {"A": 5.46600, "B": 1576.079, "C": -47.814,  "Tb_C": 110.6,  "MW": 92.14},
    "o-Xileno":             {"A": 5.51506, "B": 1736.072, "C": -46.900,  "Tb_C": 144.4,  "MW": 106.17},
    "m-Xileno":             {"A": 5.49750, "B": 1698.673, "C": -48.833,  "Tb_C": 139.1,  "MW": 106.17},
    "p-Xileno":             {"A": 5.49732, "B": 1691.879, "C": -49.235,  "Tb_C": 138.4,  "MW": 106.17},
    "Etilbenzeno":          {"A": 5.50675, "B": 1709.679, "C": -47.747,  "Tb_C": 136.2,  "MW": 106.17},
    "Estireno":             {"A": 5.56640, "B": 1861.894, "C": -44.500,  "Tb_C": 145.2,  "MW": 104.15},
    "── Alcanos ──": None,
    "n-Pentano":            {"A": 5.27087, "B": 1064.631, "C": -41.853,  "Tb_C": 36.1,   "MW": 72.15},
    "n-Hexano":             {"A": 5.26368, "B": 1202.948, "C": -52.636,  "Tb_C": 68.7,   "MW": 86.18},
    "n-Heptano":            {"A": 5.27786, "B": 1323.021, "C": -55.316,  "Tb_C": 98.4,   "MW": 100.20},
    "n-Octano":             {"A": 5.32054, "B": 1461.632, "C": -56.986,  "Tb_C": 125.7,  "MW": 114.23},
    "n-Nonano":             {"A": 5.33939, "B": 1575.415, "C": -60.410,  "Tb_C": 150.8,  "MW": 128.26},
    "Ciclohexano":          {"A": 5.26060, "B": 1295.030, "C": -58.100,  "Tb_C": 80.7,   "MW": 84.16},
    "Metilciclohexano":     {"A": 5.30550, "B": 1438.677, "C": -55.790,  "Tb_C": 100.9,  "MW": 98.19},
    "── Álcoois ──": None,
    "Metanol":              {"A": 5.31301, "B": 1676.569, "C": -21.728,  "Tb_C": 64.7,   "MW": 32.04},
    "Etanol":               {"A": 5.33675, "B": 1648.702, "C": -42.232,  "Tb_C": 78.4,   "MW": 46.07},
    "n-Propanol":           {"A": 5.37350, "B": 1788.020, "C": -35.940,  "Tb_C": 97.2,   "MW": 60.10},
    "i-Propanol":           {"A": 5.24268, "B": 1580.919, "C": -50.953,  "Tb_C": 82.4,   "MW": 60.10},
    "n-Butanol":            {"A": 5.36558, "B": 1891.523, "C": -36.055,  "Tb_C": 117.7,  "MW": 74.12},
    "── Cetonas e ésteres ──": None,
    "Acetona":              {"A": 5.31957, "B": 1490.864, "C": -35.930,  "Tb_C": 56.1,   "MW": 58.08},
    "MEK (butanona)":       {"A": 5.31424, "B": 1596.673, "C": -40.476,  "Tb_C": 79.6,   "MW": 72.11},
    "Acetato de etila":     {"A": 5.30680, "B": 1514.714, "C": -34.846,  "Tb_C": 77.1,   "MW": 88.11},
    "Acetato de n-butila":  {"A": 5.35647, "B": 1694.105, "C": -48.317,  "Tb_C": 126.1,  "MW": 116.16},
    "── Outros ──": None,
    "Água":                 {"A": 5.40221, "B": 1838.675, "C": -31.737,  "Tb_C": 100.0,  "MW": 18.02},
    "Clorofórmio":          {"A": 5.23628, "B": 1431.763, "C": -30.617,  "Tb_C": 61.2,   "MW": 119.38},
    "Diclorometano":        {"A": 5.20889, "B": 1325.938, "C": -24.064,  "Tb_C": 39.8,   "MW": 84.93},
    "Acetonitrila":         {"A": 5.28706, "B": 1492.380, "C": -32.613,  "Tb_C": 81.6,   "MW": 41.05},
    "Ácido acético":        {"A": 5.68206, "B": 1642.540, "C": -39.764,  "Tb_C": 117.9,  "MW": 60.05},
    "Dioxano":              {"A": 5.37096, "B": 1629.810, "C": -39.630,  "Tb_C": 101.3,  "MW": 88.11},
    "── Personalizado ──": None,
    "Personalizado":        {"A": None,    "B": None,     "C": None,     "Tb_C": None,   "MW": None},
}

# Dicionário plano (só componentes reais, sem separadores)
COMPONENTS = {k: v for k, v in COMPONENT_GROUPS.items() if v is not None}

# Lista para o selectbox (inclui separadores desabilitados)
COMP_OPTIONS = list(COMPONENT_GROUPS.keys())

# ─────────────────────────────────────────────
# Funções de cálculo
# ─────────────────────────────────────────────
def pvap(A, B, C, T_K):
    """Pressão de vapor em bar — Antoine NIST (log10 P/bar, T/K)"""
    return 10 ** (A - B / (T_K + C))

def bubble_T(x1, A1, B1, C1, A2, B2, C2, P_bar, T_init=None):
    """Temperatura de bolha para composição x1 via iteração."""
    if T_init is None:
        T_init = 360.0
    T = T_init
    for _ in range(200):
        pb1 = pvap(A1, B1, C1, T)
        pb2 = pvap(A2, B2, C2, T)
        Pcalc = x1 * pb1 + (1 - x1) * pb2
        err = Pcalc - P_bar
        if abs(err) < 1e-7:
            break
        # Newton simples (derivada numérica)
        dT = 0.01
        pb1d = pvap(A1, B1, C1, T + dT)
        pb2d = pvap(A2, B2, C2, T + dT)
        dPdT = (x1 * pb1d + (1 - x1) * pb2d - Pcalc) / dT
        if abs(dPdT) < 1e-12:
            break
        T -= err / dPdT
    return T

def calc_vle(A1, B1, C1, A2, B2, C2, P_bar, n_points):
    """Retorna DataFrame com x1, y1, T_K, T_C, P1sat, P2sat, alpha."""
    x_arr = np.linspace(0, 1, n_points)
    rows = []
    T_prev = None
    for x1 in x_arr:
        T = bubble_T(x1, A1, B1, C1, A2, B2, C2, P_bar, T_init=T_prev)
        T_prev = T
        P1s = pvap(A1, B1, C1, T)
        P2s = pvap(A2, B2, C2, T)
        y1 = x1 * P1s / P_bar
        alpha = P1s / P2s if P2s > 0 else np.nan
        rows.append({
            "x₁":    round(float(x1), 6),
            "y₁":    round(float(y1), 6),
            "T (K)": round(float(T),  4),
            "T (°C)":round(float(T - 273.15), 4),
            "P₁ˢᵃᵗ (bar)": round(float(P1s), 6),
            "P₂ˢᵃᵗ (bar)": round(float(P2s), 6),
            "α₁₂":   round(float(alpha), 4),
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────
# Layout Streamlit
# ─────────────────────────────────────────────
st.set_page_config(page_title="Equilíbrio VLE Bicomponente", layout="wide")
st.title("⚗️ Equilíbrio Líquido-Vapor — Sistema Bicomponente Ideal")
st.caption("Lei de Raoult + Equação de Antoine (NIST) — log₁₀(P/bar) = A − B/(T+C), T em K")

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.header("Configuração")

    st.subheader("Light Key (mais volátil)")
    comp1_name = st.selectbox(
        "Componente 1", COMP_OPTIONS, index=1,
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

    st.subheader("Heaky Key (menos volátil)")
    comp2_name = st.selectbox(
        "Componente 2", COMP_OPTIONS, index=2,
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

    st.subheader("Condições")
    P_atm = st.number_input("Pressão de operação (atm)", value=1.0, min_value=0.1, max_value=20.0, step=0.1)
    P_bar = P_atm * 1.01325

    n_points = st.slider("Número de pontos", min_value=11, max_value=101, value=21, step=5)

    calcular = st.button("🔄 Calcular", type="primary", use_container_width=True)

# ── Cálculo ───────────────────────────────────
if calcular or True:  # calcula sempre ao iniciar
    try:
        # Verificar componente 1 mais volátil
        Tb1 = bubble_T(1.0, A1, B1, C1, A2, B2, C2, P_bar)
        Tb2 = bubble_T(0.0, A1, B1, C1, A2, B2, C2, P_bar)
        if Tb1 > Tb2:
            st.warning(
                f"⚠️ {c1_label} parece menos volátil que {c2_label} nessa pressão. "
                "Considere trocar a ordem dos componentes para convenção x₁ = componente mais volátil."
            )

        df = calc_vle(A1, B1, C1, A2, B2, C2, P_bar, n_points)

        # ── Métricas ──────────────────────────
        alpha_mean = df["α₁₂"].mean()
        alpha_min  = df["α₁₂"].min()
        alpha_max  = df["α₁₂"].max()
        T_bolha_puro1 = df["T (°C)"].iloc[-1]
        T_bolha_puro2 = df["T (°C)"].iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("P operação", f"{P_atm:.2f} atm")
        col2.metric(f"T_eb {c1_label}", f"{T_bolha_puro1:.1f} °C")
        col3.metric(f"T_eb {c2_label}", f"{T_bolha_puro2:.1f} °C")
        col4.metric("α₁₂ médio", f"{alpha_mean:.3f}")

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
        al = df["α₁₂"].values

        # 1) Diagrama y–x
        ax1 = fig.add_subplot(gs[0, 0])
        style_ax(ax1, f"Diagrama y–x  ({c1_label} / {c2_label})", f"x₁  [{c1_label}]", f"y₁  [{c1_label}]")
        ax1.plot(x, y, color=BLUE, lw=2, label="Curva VLE")
        ax1.plot([0, 1], [0, 1], color=GRAY, lw=1, ls="--", label="y = x")
        ax1.scatter(x[1:-1], y[1:-1], color=BLUE, s=18, zorder=5)
        ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
        ax1.legend(fontsize=8, facecolor=AXBG, labelcolor=TXT, edgecolor=GRID)

        # 2) Diagrama T–x–y
        ax2 = fig.add_subplot(gs[0, 1])
        style_ax(ax2, f"Diagrama T–x–y  ({P_atm:.2f} atm)", f"Fração molar [{c1_label}]", "T (°C)")
        ax2.plot(x, T, color=ORANGE, lw=2, label="Curva bolha (T–x)")
        ax2.plot(y, T, color=BLUE,   lw=2, ls="--", label="Curva orvalho (T–y)")
        ax2.fill_betweenx(T, x, y, alpha=0.08, color=GREEN)
        ax2.legend(fontsize=8, facecolor=AXBG, labelcolor=TXT, edgecolor=GRID)

        # 3) Pressões de vapor vs T
        ax3 = fig.add_subplot(gs[1, 0])
        style_ax(ax3, "Pressões de vapor vs Temperatura", "T (°C)", "Pˢᵃᵗ (bar)")
        ax3.plot(T, P1, color=ORANGE, lw=2, label=f"P₁ˢᵃᵗ  {c1_label}")
        ax3.plot(T, P2, color=BLUE,   lw=2, label=f"P₂ˢᵃᵗ  {c2_label}")
        ax3.axhline(P_bar, color=GRAY, lw=1, ls=":", label=f"P op = {P_bar:.3f} bar")
        ax3.legend(fontsize=8, facecolor=AXBG, labelcolor=TXT, edgecolor=GRID)

        # 4) Volatilidade relativa vs x
        ax4 = fig.add_subplot(gs[1, 1])
        style_ax(ax4, "Volatilidade relativa α₁₂ vs x₁", f"x₁  [{c1_label}]", "α₁₂ = P₁ˢᵃᵗ / P₂ˢᵃᵗ")
        ax4.plot(x, al, color=GREEN, lw=2)
        ax4.axhline(alpha_mean, color=GRAY, lw=1, ls="--", label=f"α médio = {alpha_mean:.3f}")
        ax4.axhline(1.0, color=ORANGE, lw=0.8, ls=":", label="α = 1 (sem separação)")
        ax4.scatter(x[1:-1], al[1:-1], color=GREEN, s=18, zorder=5)
        ax4.legend(fontsize=8, facecolor=AXBG, labelcolor=TXT, edgecolor=GRID)

        # Título geral
        fig.suptitle(
            f"Equilíbrio VLE — {c1_label} / {c2_label}   |   P = {P_atm:.2f} atm   |   "
            f"Raoult + Antoine (NIST)",
            color=TXT, fontsize=12, fontweight="bold", y=0.98
        )

        st.pyplot(fig)
        plt.close(fig)

        # ── Tabela de dados ───────────────────
        st.subheader("📋 Tabela de dados de equilíbrio")
        st.dataframe(
            df.style.format({
                "x₁": "{:.4f}", "y₁": "{:.4f}",
                "T (K)": "{:.2f}", "T (°C)": "{:.2f}",
                "P₁ˢᵃᵗ (bar)": "{:.5f}", "P₂ˢᵃᵗ (bar)": "{:.5f}",
                "α₁₂": "{:.4f}",
            }),
            use_container_width=True, height=300
        )

        # ── Downloads ─────────────────────────
        st.subheader("📥 Exportar dados")
        col_a, col_b = st.columns(2)

        # CSV completo
        csv_full = df.to_csv(index=False,header=False, sep=";", decimal=",")
        col_a.download_button(
            "⬇️ CSV completo (separador ;)",
            data=csv_full.encode("utf-8"),
            file_name=f"VLE_{c1_label}_{c2_label}_{P_atm}atm.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # CSV x,y apenas (para app McCabe-Thiele)
        df_xy = df[["x₁", "y₁"]].rename(columns={"x₁": "x", "y₁": "y"})
        csv_xy = df_xy.to_csv(index=False,header=False, sep=",", decimal=".")
        col_b.download_button(
            "⬇️ x,y para McCabe-Thiele (CSV)",
            data=csv_xy.encode("utf-8"),
            file_name=f"xy_{c1_label}_{c2_label}_{P_atm}atm.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # ── Info Antoine ──────────────────────
        with st.expander("ℹ️ Constantes de Antoine utilizadas"):
            ac = pd.DataFrame({
                "Componente": [c1_label, c2_label],
                "A": [A1, A2], "B": [B1, B2], "C": [C1, C2],
                "Referência": ["NIST WebBook", "NIST WebBook"],
                "Equação": ["log₁₀(P/bar) = A − B/(T+C), T em K"] * 2,
            })
            st.dataframe(ac, use_container_width=True)

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
