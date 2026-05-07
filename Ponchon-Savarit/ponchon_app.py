"""
PONCHON-SAVARIT - VERSÃO SIMPLIFICADA E CORRIGIDA
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import pandas as pd

st.set_page_config(layout="wide")
st.title("⚗️ Método de Ponchon-Savarit - Destilação Binária")

# Dados dos compostos (apenas Benzeno e Tolueno para teste, mas pode expandir)
COMPOUNDS = {
    "Benzeno": {"A":6.90565,"B":1211.033,"C":220.790,"Hvap_ref":30.72,"dHvap_dT":-0.060,"CpL":0.1350,"CpV":0.0830,"Tb":80.1,"color":"#42a5f5"},
    "Tolueno": {"A":6.95334,"B":1343.943,"C":219.377,"Hvap_ref":33.18,"dHvap_dT":-0.055,"CpL":0.1572,"CpV":0.1030,"Tb":110.6,"color":"#ef5350"},
}

def Psat(comp, T):
    d = COMPOUNDS[comp]
    return 10**(d["A"] - d["B"]/(d["C"]+T))

def T_pure(comp, P):
    P_mmHg = P*750.062
    def f(T):
        return Psat(comp, T) - P_mmHg
    return brentq(f, 0, 300)

def bubble_T(x, compA, compB, P):
    P_mmHg = P*750.062
    def f(T):
        yA = x*Psat(compA,T)/P_mmHg
        yB = (1-x)*Psat(compB,T)/P_mmHg
        return yA + yB - 1
    T_A = T_pure(compA,P)
    T_B = T_pure(compB,P)
    return brentq(f, min(T_A,T_B)-10, max(T_A,T_B)+10)

def y_eq(x, T, compA, compB, P):
    return x*Psat(compA,T)/(P*750.062)

def HL_pure(comp, T):
    d = COMPOUNDS[comp]
    return d["CpL"]*(T - 0)
def HV_pure(comp, T):
    d = COMPOUNDS[comp]
    Tb = d["Tb"]
    lam = d["Hvap_ref"] + d["dHvap_dT"]*(Tb-25)
    return d["CpL"]*(Tb-0) + lam + d["CpV"]*(T-Tb)
def HL_mix(x, T, compA, compB):
    return x*HL_pure(compA,T) + (1-x)*HL_pure(compB,T)
def HV_mix(y, T, compA, compB):
    return y*HV_pure(compA,T) + (1-y)*HV_pure(compB,T)

@st.cache_data
def get_curves(compA, compB, P, N=100):
    x_vals = np.linspace(0,1,N)
    y_vals = np.zeros(N)
    HL = np.zeros(N)
    HV = np.zeros(N)
    Tb = np.zeros(N)
    for i,x in enumerate(x_vals):
        try:
            T = bubble_T(x, compA, compB, P)
            y = y_eq(x, T, compA, compB, P)
            y_vals[i] = y
            HL[i] = HL_mix(x, T, compA, compB)
            HV[i] = HV_mix(y, T, compA, compB)
            Tb[i] = T
        except:
            y_vals[i] = np.nan
            HL[i] = np.nan
            HV[i] = np.nan
    return x_vals, y_vals, HL, HV, Tb

def compute_poles(xD, xW, zF, R, q, HL, HV, x_arr, y_arr, compA, compB, P):
    # Topo
    HL_D = np.interp(xD, x_arr, HL)
    yD = np.interp(xD, x_arr, y_arr)
    HV_D = np.interp(yD, y_arr, HV)
    HD = (R+1)*HV_D - R*HL_D
    # Alimentação
    HL_F = np.interp(zF, x_arr, HL)
    yF = np.interp(zF, x_arr, y_arr)
    HV_F = np.interp(yF, y_arr, HV)
    HF = (1-q)*HV_F + q*HL_F
    # Polo de esgotamento
    if abs(xD - zF) < 1e-6:
        HW = HF
    else:
        slope = (HD - HF)/(xD - zF)
        HW = HF + slope*(xW - zF)
    # Rmin
    Rmin = None
    for i in range(len(x_arr)):
        if x_arr[i] >= xD: continue
        if abs(y_arr[i] - x_arr[i]) < 1e-6: continue
        slope_tl = (HV[i] - HL[i])/(y_arr[i] - x_arr[i])
        H_at_xD = HL[i] + slope_tl*(xD - x_arr[i])
        if H_at_xD > HV_D:
            R_calc = (H_at_xD - HV_D)/(HV_D - HL_D)
            if R_calc > 0:
                if Rmin is None or R_calc < Rmin:
                    Rmin = R_calc
    return HD, HW, HF, Rmin, HL_D, HV_D, yD

def calculate_stages(xD, xW, zF, HD, HW, HL, HV, x_arr, y_arr, compA, compB, P, max_stages=50):
    stages = []
    x_cur = xD
    HL_cur = np.interp(xD, x_arr, HL)
    in_rect = True
    for n in range(1, max_stages+1):
        polo_x = xD if in_rect else xW
        polo_H = HD if in_rect else HW
        # Encontrar y1 na curva HV
        def res(y):
            if y < 0 or y > 1: return 1e6
            H_line = HL_cur + (polo_H - HL_cur)/(polo_x - x_cur)*(y - x_cur)
            H_curve = np.interp(y, y_arr, HV)
            return H_curve - H_line
        try:
            y1 = brentq(res, max(0.01, x_cur+1e-4), 0.99)
            HV1 = np.interp(y1, y_arr, HV)
        except:
            break
        # Tie-line para x1
        x1 = np.interp(y1, y_arr, x_arr)
        HL1 = np.interp(x1, x_arr, HL)
        stages.append({
            "n": n, "section": "R" if in_rect else "S",
            "xL": x1, "yV": y1,
            "HL": HL1, "HV": HV1,
            "pole_x": polo_x, "pole_H": polo_H,
        })
        if in_rect and x1 <= zF + 1e-4:
            in_rect = False
        if x1 <= xW + 1e-3:
            break
        x_cur = x1
        HL_cur = HL1
    return stages

def plot_hxy(x_arr, y_arr, HL, HV, xD, xW, zF, HD, HW, HF, stages, compA, compB, P, show_stages):
    fig, ax = plt.subplots(figsize=(10,7))
    ax.fill_between(x_arr, HL, HV, color="#c0e0ff", alpha=0.3)
    ax.plot(x_arr, HV, 'b-', lw=2, label='$H_V(y)$')
    ax.plot(x_arr, HL, 'r-', lw=2, label='$H_L(x)$')
    # Pontos de produto
    HL_D = np.interp(xD, x_arr, HL)
    HL_W = np.interp(xW, x_arr, HL)
    ax.plot(xD, HL_D, 's', color='blue', ms=10)
    ax.plot(xW, HL_W, 's', color='red', ms=10)
    ax.plot(zF, HF, 'D', color='green', ms=12)
    # Polos
    ax.plot(xD, HD, '*', color='purple', ms=15, label='$\Delta_R$')
    ax.plot(xW, HW, '*', color='orange', ms=15, label='$\Delta_S$')
    # Reta polos-alimentação
    if abs(xD - zF) > 1e-6:
        xs = np.array([xW, xD])
        ys = HF + (HD-HF)/(xD-zF)*(xs - zF)
        ax.plot(xs, ys, 'k--', alpha=0.5)
    # Estágios
    if show_stages and stages:
        for s in stages:
            # Reta operacional (polo -> ponto vapor)
            ax.plot([s["pole_x"], s["yV"]], [s["pole_H"], s["HV"]], 'gray', lw=1, alpha=0.7)
            # Tie-line
            ax.plot([s["yV"], s["xL"]], [s["HV"], s["HL"]], 'k--', lw=1.5)
            ax.text(s["xL"]-0.02, s["HL"]-0.5, str(s["n"]), fontsize=8)
    ax.set_xlim(0,1)
    ax.set_xlabel(f'Fração molar de {compA}')
    ax.set_ylabel('Entalpia (kJ/mol)')
    ax.set_title(f'Diagrama H-x-y - {compA}/{compB} - P={P:.3f} bar')
    ax.legend()
    ax.grid(True)
    return fig

# Interface
col1, col2 = st.columns([1,3])
with col1:
    compA = st.selectbox("Componente leve", list(COMPOUNDS.keys()))
    compB = st.selectbox("Componente pesado", [c for c in COMPOUNDS if c != compA])
    P = st.number_input("Pressão (bar)", 0.5, 5.0, 1.013, 0.01)
    xD = st.slider("Destilado xD", 0.5, 0.999, 0.90)
    xW = st.slider("Resíduo xW", 0.001, 0.4, 0.05)
    zF = st.slider("Alimentação zF", 0.05, 0.95, 0.45)
    R = st.slider("Refluxo R", 0.5, 10.0, 2.5)
    q = st.selectbox("Condição q", [1.0, 0.0, 0.5, 0.8, 1.2])
    if q not in [1.0, 0.0]:
        q = st.number_input("Valor de q", -0.5, 1.5, 0.8)
    show_stages = st.checkbox("Mostrar estágios", True)
    calc = st.button("Calcular")

if calc:
    with st.spinner("Calculando curvas..."):
        x_arr, y_arr, HL, HV, Tb = get_curves(compA, compB, P, 200)
        HD, HW, HF, Rmin, HL_D, HV_D, yD = compute_poles(xD, xW, zF, R, q, HL, HV, x_arr, y_arr, compA, compB, P)
        if Rmin is None:
            st.error("Não foi possível calcular Rmin. Tente outros valores.")
        elif R < Rmin:
            st.warning(f"Refluxo insuficiente: R={R} < Rmin={Rmin:.3f}")
            stages = []
        else:
            stages = calculate_stages(xD, xW, zF, HD, HW, HL, HV, x_arr, y_arr, compA, compB, P) if show_stages else []
        fig = plot_hxy(x_arr, y_arr, HL, HV, xD, xW, zF, HD, HW, HF, stages, compA, compB, P, show_stages)
        st.pyplot(fig)
        # Resultados
        st.subheader("Resultados")
        cola, colb = st.columns(2)
        cola.metric("ΔR (H'D)", f"{HD:.2f} kJ/mol")
        cola.metric("ΔS (H'W)", f"{HW:.2f} kJ/mol")
        colb.metric("Alimentação HF", f"{HF:.2f} kJ/mol")
        colb.metric("Rmin", f"{Rmin:.4f}" if Rmin else "N/A")
        if stages:
            st.success(f"Estágios calculados: {len(stages)} (incluindo refervedor)")
            st.dataframe(pd.DataFrame(stages))
