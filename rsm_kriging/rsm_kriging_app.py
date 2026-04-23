# -*- coding: utf-8 -*-
"""
RSM–KRIGING ANALYSIS SUITE — STREAMLIT APP
============================================
Universidade Estadual de Maringá (UEM)
Departamento de Engenharia Química / 3DCP Lab

Prof. Dr. Ricardo V. P. Rezende
Doutoranda Allana Ribeiro Mendes

Versão Streamlit com gráficos 3D interativos (Plotly)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel, Matern, WhiteKernel, RBF, RationalQuadratic
)

# ──────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="RSM–Kriging Suite | UEM",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# CSS PERSONALIZADO
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Fundo e tipografia */
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #1565c0 100%);
        padding: 1.6rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 4px 20px rgba(26,35,126,0.3);
    }
    .main-header h1 { margin: 0; font-size: 1.9rem; font-weight: 700; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.9rem; }

    /* Cards de métricas */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 5px solid #1565c0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 0.6rem;
    }
    .metric-card.green  { border-left-color: #2e7d32; }
    .metric-card.orange { border-left-color: #e65100; }
    .metric-card.red    { border-left-color: #c62828; }

    /* Verdict box */
    .verdict-box {
        border-radius: 12px;
        padding: 1.4rem;
        font-size: 1rem;
        font-weight: 500;
        margin-top: 0.8rem;
    }
    .verdict-kriging { background:#e8f5e9; border:2px solid #2e7d32; color:#1b5e20; }
    .verdict-rsm     { background:#e3f2fd; border:2px solid #1565c0; color:#0d47a1; }

    /* Abas */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #f0f4f8;
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #1565c0 !important;
        color: white !important;
    }

    /* Tabelas */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* Sidebar */
    .css-1d391kg { background-color: #f8fafc; }
    .sidebar-section {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    }
    .sidebar-section h4 { color: #1565c0; margin: 0 0 0.7rem; font-size: 0.9rem; }

    /* Botão executar */
    .stButton > button {
        background: linear-gradient(90deg, #1565c0, #283593);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

    /* Info box */
    .info-block {
        background: #e3f2fd;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        border-left: 4px solid #1565c0;
        font-size: 0.88rem;
        color: #0d47a1;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# FUNÇÕES ANALÍTICAS
# ──────────────────────────────────────────────

def montar_kernel(nome):
    base = {"Matern 3/2": Matern(nu=1.5), "Matern 5/2": Matern(nu=2.5),
            "RBF": RBF(), "Rational Quadratic": RationalQuadratic()}[nome]
    return ConstantKernel(1.0) * base + WhiteKernel(noise_level=1e-3)


def calcular_metricas(y, y_pred, nome="Modelo", p=None):
    y, y_pred = np.asarray(y, float), np.asarray(y_pred, float)
    resid = y - y_pred
    n = len(y)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae  = mean_absolute_error(y, y_pred)
    bias = np.mean(resid)
    max_abs = np.max(np.abs(resid))
    mape = np.mean(np.abs(resid) / np.maximum(np.abs(y), 1e-12)) * 100
    sse = np.sum(resid**2)
    sst = np.sum((y - np.mean(y))**2)
    r2  = 1 - sse/sst if sst > 0 else np.nan
    press = sse
    q2    = 1 - press/sst if sst > 0 else np.nan
    if p and n > p+1 and sse > 0:
        aic = n*np.log(sse/n) + 2*p
        bic = n*np.log(sse/n) + p*np.log(n)
    else:
        aic = bic = np.nan
    return {"Modelo": nome, "R²": r2, "RMSE": rmse, "MAE": mae,
            "MAPE (%)": mape, "Bias": bias, "Erro abs máx": max_abs,
            "PRESS": press, "Q²": q2, "AIC": aic, "BIC": bic}


def gradiente_rsm(beta, hn, r1):
    return np.array([beta[1] + beta[3]*r1 + 2*beta[4]*hn,
                     beta[2] + beta[3]*hn + 2*beta[5]*r1])


def hessiana_rsm(beta):
    return np.array([[2*beta[4], beta[3]], [beta[3], 2*beta[5]]])


def classificar_regiao_local(diff_abs, diff_rel, sigma, faixa_y):
    sigma = max(sigma, 1e-12)
    razao_sigma = diff_abs / sigma
    frac_faixa  = diff_abs / max(faixa_y, 1e-12)
    if   razao_sigma < 1 and frac_faixa < 0.03: return "✅ Excelente concordância local"
    elif razao_sigma < 2 and frac_faixa < 0.07: return "🟢 Boa concordância local"
    elif razao_sigma < 3 or  frac_faixa < 0.12: return "🟡 Concordância moderada / atenção"
    else: return "🔴 Conflito relevante entre os modelos"


def analisar_ponto_local(hn_novo, r1_novo, beta, gpr, scaler_X, X_exp, y, HN, R):
    y_rsm = (beta[0] + beta[1]*hn_novo + beta[2]*r1_novo +
             beta[3]*hn_novo*r1_novo + beta[4]*hn_novo**2 + beta[5]*r1_novo**2)
    grad    = gradiente_rsm(beta, hn_novo, r1_novo)
    H       = hessiana_rsm(beta)
    eigvals = np.linalg.eigvals(H)
    X_novo_scaled = scaler_X.transform([[hn_novo, r1_novo]])
    y_krig, sigma = gpr.predict(X_novo_scaled, return_std=True)
    y_krig = y_krig[0]; sigma = sigma[0]
    distancias = np.sqrt((HN - hn_novo)**2 + (R - r1_novo)**2)
    dmin    = np.min(distancias)
    idx_min = np.argmin(distancias)
    diff_abs = abs(y_krig - y_rsm)
    diff_rel = 100*diff_abs / max(abs(y_krig), 1e-12)
    faixa_y  = np.max(y) - np.min(y)
    z_rel    = sigma / max(faixa_y, 1e-12)
    return {
        "HN": hn_novo, "r1": r1_novo,
        "RSM_pred": y_rsm, "Krig_pred": y_krig, "Krig_sigma": sigma,
        "IC_inf_95": y_krig - 1.96*sigma, "IC_sup_95": y_krig + 1.96*sigma,
        "Diff_abs": diff_abs, "Diff_rel_%": diff_rel,
        "Razao_diff_sigma": diff_abs/max(sigma,1e-12),
        "Sigma_rel_faixa_%": 100*z_rel,
        "Dist_ponto_mais_prox": dmin, "Caso_mais_prox": idx_min,
        "Grad_dy_dHN": grad[0], "Grad_dy_dr1": grad[1],
        "Norma_grad": np.linalg.norm(grad),
        "Curvatura_autovalor_1": eigvals[0], "Curvatura_autovalor_2": eigvals[1],
        "Interpretação": classificar_regiao_local(diff_abs, diff_rel, sigma, faixa_y)
    }


@st.cache_data(show_spinner=False)
def rodar_analise(df_bytes, resposta, kernel_nome, hn_novo, r1_novo):
    df = pd.read_excel(io.BytesIO(df_bytes))
    df.columns = df.columns.str.strip()
    dados = df[["caso", "hn", "r1", resposta]].copy()
    HN = dados["hn"].values.astype(float)
    R  = dados["r1"].values.astype(float)
    y  = dados[resposta].values.astype(float)

    # RSM
    X_rsm = np.column_stack([np.ones(len(HN)), HN, R, HN*R, HN**2, R**2])
    model_rsm = LinearRegression(fit_intercept=False)
    model_rsm.fit(X_rsm, y)
    beta      = model_rsm.coef_
    y_pred_rsm = model_rsm.predict(X_rsm)

    # LOOCV RSM
    loo = LeaveOneOut()
    y_pred_rsm_cv = np.zeros_like(y, float)
    for tr, te in loo.split(X_rsm):
        m = LinearRegression(fit_intercept=False).fit(X_rsm[tr], y[tr])
        y_pred_rsm_cv[te[0]] = m.predict(X_rsm[te])[0]

    # Kriging
    X_krig = dados[["hn","r1"]].values
    scaler_X = StandardScaler()
    X_krig_scaled = scaler_X.fit_transform(X_krig)
    kernel = montar_kernel(kernel_nome)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20,
                                   normalize_y=True, random_state=42)
    gpr.fit(X_krig_scaled, y)
    y_pred_krig, y_std_krig = gpr.predict(X_krig_scaled, return_std=True)

    # LOOCV Kriging
    y_pred_krig_cv = np.zeros_like(y, float)
    for tr, te in loo.split(X_krig_scaled):
        g = GaussianProcessRegressor(kernel=montar_kernel(kernel_nome),
                                     n_restarts_optimizer=10,
                                     normalize_y=True, random_state=42)
        g.fit(X_krig_scaled[tr], y[tr])
        y_pred_krig_cv[te[0]] = g.predict(X_krig_scaled[te])[0]

    # Métricas
    met = pd.DataFrame([
        calcular_metricas(y, y_pred_rsm,    "RSM (Treino)",                 p=6),
        calcular_metricas(y, y_pred_rsm_cv, "RSM (LOOCV)",                  p=6),
        calcular_metricas(y, y_pred_krig,   f"Kriging ({kernel_nome}) Treino"),
        calcular_metricas(y, y_pred_krig_cv,f"Kriging ({kernel_nome}) LOOCV"),
    ])

    # Malha
    margin_hn = 0.10*(HN.max()-HN.min())
    margin_r1 = 0.10*(R.max()-R.min())
    hn_range  = np.linspace(HN.min()-margin_hn, HN.max()+margin_hn, 80)
    r1_range  = np.linspace(R.min()-margin_r1,  R.max()+margin_r1,  80)
    HN_grid, R_grid = np.meshgrid(hn_range, r1_range)

    Z_rsm = (beta[0] + beta[1]*HN_grid + beta[2]*R_grid +
             beta[3]*HN_grid*R_grid + beta[4]*HN_grid**2 + beta[5]*R_grid**2)

    X_grid = np.column_stack([HN_grid.ravel(), R_grid.ravel()])
    Z_krig_pred, Z_krig_std = gpr.predict(scaler_X.transform(X_grid), return_std=True)
    Z_krig  = Z_krig_pred.reshape(HN_grid.shape)
    Z_sigma = Z_krig_std.reshape(HN_grid.shape)
    Z_diff  = Z_krig - Z_rsm

    # Ponto local
    ponto_local = analisar_ponto_local(
        hn_novo, r1_novo, beta, gpr, scaler_X, X_krig, y, HN, R)
    y_rsm_local  = ponto_local["RSM_pred"]
    y_krig_local = ponto_local["Krig_pred"]

    # Equação RSM
    eq_text = (f"y = {beta[0]:.5f} {beta[1]:+.5f}·HN {beta[2]:+.5f}·r1 "
               f"{beta[3]:+.5f}·HN·r1 {beta[4]:+.5f}·HN² {beta[5]:+.5f}·r1²")

    # Ponto crítico
    A_mat = np.array([[2*beta[4], beta[3]], [beta[3], 2*beta[5]]])
    b_vec = np.array([-beta[1], -beta[2]])
    try:
        pc   = np.linalg.solve(A_mat, b_vec)
        HN_s, R_s = pc
        y_s  = (beta[0]+beta[1]*HN_s+beta[2]*R_s+beta[3]*HN_s*R_s+
                beta[4]*HN_s**2+beta[5]*R_s**2)
        Hess = np.array([[2*beta[4],beta[3]],[beta[3],2*beta[5]]])
        ev   = np.linalg.eigvals(Hess)
        detH = np.linalg.det(Hess)
        if np.all(ev>0):   cls = "🟢 Mínimo local"
        elif np.all(ev<0): cls = "🔴 Máximo local"
        else:              cls = "🟡 Ponto de sela"
        ponto_critico_df = pd.DataFrame({
            "Parâmetro": ["HN*","r1*",f"{resposta.capitalize()}(HN*,r1*)",
                          "Det(H)","Autovalor 1","Autovalor 2","Classificação",
                          "HN* dentro da faixa?","r1* dentro da faixa?"],
            "Valor": [HN_s, R_s, y_s, detH, ev[0], ev[1], cls,
                      HN.min()<=HN_s<=HN.max(), R.min()<=R_s<=R.max()]
        })
    except Exception:
        ponto_critico_df = pd.DataFrame({"Parâmetro":["Erro"],"Valor":["Sistema singular"]})

    # Overfitting + veredicto
    rmse_rsm_train  = np.sqrt(mean_squared_error(y,y_pred_rsm))
    rmse_krig_train = np.sqrt(mean_squared_error(y,y_pred_krig))
    rmse_rsm_cv     = np.sqrt(mean_squared_error(y,y_pred_rsm_cv))
    rmse_krig_cv    = np.sqrt(mean_squared_error(y,y_pred_krig_cv))
    r2_rsm_train    = r2_score(y,y_pred_rsm)
    r2_krig_train   = r2_score(y,y_pred_krig)
    r2_rsm_cv       = r2_score(y,y_pred_rsm_cv)
    r2_krig_cv      = r2_score(y,y_pred_krig_cv)
    delta_r2_rsm    = r2_rsm_train  - r2_rsm_cv
    delta_r2_krig   = r2_krig_train - r2_krig_cv
    delta_rmse_rsm  = rmse_rsm_cv  - rmse_rsm_train
    delta_rmse_krig = rmse_krig_cv - rmse_krig_train

    def classificar_risco(d_r2, d_rmse):
        if d_r2<0.05 and d_rmse<1.5: return "🟢 Baixo"
        elif d_r2<0.15 and d_rmse<4: return "🟡 Moderado"
        else: return "🔴 Alto"

    risco_rsm  = classificar_risco(delta_r2_rsm,  delta_rmse_rsm)
    risco_krig = classificar_risco(delta_r2_krig, delta_rmse_krig)

    score_rsm  = (2 if rmse_rsm_cv<rmse_krig_cv else 0) + (1 if r2_rsm_cv>r2_krig_cv else 0)
    score_krig = (2 if rmse_krig_cv<=rmse_rsm_cv else 0) + (1 if r2_krig_cv>=r2_rsm_cv else 0)
    if "Alto" in risco_rsm:  score_rsm  -= 1
    if "Alto" in risco_krig: score_krig -= 1
    vencedor = "RSM" if score_rsm>score_krig else "Kriging"

    df_overfit = pd.DataFrame({
        "Modelo": ["RSM",f"Kriging ({kernel_nome})"],
        "ΔR²":   [round(delta_r2_rsm,5),   round(delta_r2_krig,5)],
        "ΔRMSE": [round(delta_rmse_rsm,5),  round(delta_rmse_krig,5)],
        "Risco": [risco_rsm, risco_krig],
        "Score": [score_rsm, score_krig]
    })

    # Tabela detalhada por caso
    res_rsm_cv  = y - y_pred_rsm_cv
    res_krig_cv = y - y_pred_krig_cv
    resultado   = dados.copy()
    resultado["RSM_train"]    = y_pred_rsm
    resultado["RSM_LOOCV"]   = y_pred_rsm_cv
    resultado["RSM_res_LOOCV"] = res_rsm_cv
    resultado["Krig_train"]  = y_pred_krig
    resultado["Krig_sigma"]  = y_std_krig
    resultado["Krig_LOOCV"]  = y_pred_krig_cv
    resultado["Krig_res_LOOCV"] = res_krig_cv

    resumo_final = pd.DataFrame({
        "Parâmetro": [
            "Kernel Kriging","RSM R² treino","RSM RMSE treino",
            "RSM R² LOOCV","RSM RMSE LOOCV",
            f"Kriging R² treino",f"Kriging RMSE treino",
            f"Kriging R² LOOCV",f"Kriging RMSE LOOCV",
            "Sigma médio (Kriging)","Predição local RSM",
            "Predição local Kriging","Diferença local abs","Vencedor"
        ],
        "Valor": [
            kernel_nome, round(r2_rsm_train,6), round(rmse_rsm_train,6),
            round(r2_rsm_cv,6),  round(rmse_rsm_cv,6),
            round(r2_krig_train,6), round(rmse_krig_train,6),
            round(r2_krig_cv,6),    round(rmse_krig_cv,6),
            round(Z_sigma.mean(),6), round(y_rsm_local,6),
            round(y_krig_local,6),   round(abs(y_rsm_local-y_krig_local),6),
            vencedor
        ]
    })

    analise_local_df = pd.DataFrame({
        "Parâmetro": list(ponto_local.keys()),
        "Valor":     list(ponto_local.values())
    })

    return dict(
        HN=HN, R=R, y=y, dados=dados,
        beta=beta, eq_text=eq_text,
        y_pred_rsm=y_pred_rsm, y_pred_rsm_cv=y_pred_rsm_cv,
        y_pred_krig=y_pred_krig, y_pred_krig_cv=y_pred_krig_cv,
        y_std_krig=y_std_krig, res_rsm_cv=res_rsm_cv, res_krig_cv=res_krig_cv,
        HN_grid=HN_grid, R_grid=R_grid,
        Z_rsm=Z_rsm, Z_krig=Z_krig, Z_sigma=Z_sigma, Z_diff=Z_diff,
        y_rsm_local=y_rsm_local, y_krig_local=y_krig_local,
        ponto_local=ponto_local,
        tabela_metricas=met, resultado=resultado,
        resumo_final=resumo_final, analise_local_df=analise_local_df,
        ponto_critico_df=ponto_critico_df, df_overfit=df_overfit,
        vencedor=vencedor, kernel_nome=kernel_nome,
        r2_rsm_cv=r2_rsm_cv, r2_krig_cv=r2_krig_cv,
        rmse_rsm_cv=rmse_rsm_cv, rmse_krig_cv=rmse_krig_cv,
    )

# ──────────────────────────────────────────────
# FUNÇÕES DE PLOTAGEM PLOTLY
# ──────────────────────────────────────────────

COLORSCALE_SURF = "Turbo"
COLORSCALE_CONT = "Jet"
COLORSCALE_SIGMA = "Viridis"
COLORSCALE_DIFF  = "RdBu"

def plot_3d_surface(HN_grid, R_grid, Z, titulo, resposta,
                    HN_pts, R_pts, y_pts,
                    hn_star=None, r1_star=None, y_star=None,
                    colorscale=COLORSCALE_SURF, sigma_surf=None):
    fig = go.Figure()
    # Superfície principal
    kwargs = dict(opacity=0.82, colorscale=colorscale, showscale=True,
                  lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3, roughness=0.5),
                  colorbar=dict(title=resposta.capitalize(), thickness=14, len=0.7))
    if sigma_surf is not None:
        kwargs["surfacecolor"] = sigma_surf
        kwargs["colorbar"]["title"] = "σ (incerteza)"
    fig.add_trace(go.Surface(x=HN_grid, y=R_grid, z=Z, **kwargs))
    # Pontos experimentais
    fig.add_trace(go.Scatter3d(
        x=HN_pts, y=R_pts, z=y_pts,
        mode="markers",
        marker=dict(size=7, color="white", line=dict(color="black", width=2)),
        name="Exp.", hovertemplate="HN=%{x:.2f}<br>r1=%{y:.3f}<br>y=%{z:.4f}<extra></extra>"
    ))
    # Ponto selecionado
    if hn_star is not None:
        fig.add_trace(go.Scatter3d(
            x=[hn_star], y=[r1_star], z=[y_star],
            mode="markers",
            marker=dict(size=10, color="red", symbol="diamond",
                        line=dict(color="black", width=1.5)),
            name="P* selecionado",
            hovertemplate=f"HN={hn_star:.2f}<br>r1={r1_star:.3f}<br>y={y_star:.4f}<extra>P*</extra>"
        ))
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=15, color="#1565c0")),
        scene=dict(
            xaxis=dict(title="HN (mm)", backgroundcolor="#f8fafc",
                       gridcolor="lightgray", showbackground=True),
            yaxis=dict(title="r1 (V/U)", backgroundcolor="#f8fafc",
                       gridcolor="lightgray", showbackground=True),
            zaxis=dict(title=resposta.capitalize(), backgroundcolor="#f8fafc",
                       gridcolor="lightgray", showbackground=True),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2)),
            aspectmode="auto"
        ),
        height=560, margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(yanchor="top", y=0.97, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="#ccc", borderwidth=1),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_contorno(HN_grid, R_grid, Z, titulo, resposta, HN_pts, R_pts, casos,
                  hn_star=None, r1_star=None, colorscale=COLORSCALE_CONT):
    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=HN_grid[0], y=R_grid[:,0], z=Z,
        colorscale=colorscale, ncontours=18, showscale=True,
        contours=dict(showlabels=True, labelfont=dict(size=10, color="black")),
        colorbar=dict(title=resposta.capitalize(), thickness=14)
    ))
    fig.add_trace(go.Scatter(
        x=HN_pts, y=R_pts, mode="markers+text",
        text=[str(c) for c in casos], textposition="top right",
        marker=dict(size=10, color="white", line=dict(color="black",width=2)),
        name="Exp.", hovertemplate="Caso %{text}<br>HN=%{x:.2f}<br>r1=%{y:.3f}<extra></extra>"
    ))
    if hn_star is not None:
        fig.add_trace(go.Scatter(
            x=[hn_star], y=[r1_star], mode="markers+text",
            text=["P*"], textposition="top right",
            marker=dict(size=14, color="red", symbol="star",
                        line=dict(color="black",width=1.5)),
            name="P*"
        ))
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=14, color="#1565c0")),
        xaxis=dict(title="HN (mm)", gridcolor="#e0e0e0"),
        yaxis=dict(title="r1 (V/U)", gridcolor="#e0e0e0"),
        height=440, margin=dict(l=60,r=40,t=50,b=50),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fafafa",
        legend=dict(bgcolor="rgba(255,255,255,0.85)")
    )
    return fig


def plot_parity(y_obs, y_pred_rsm, y_pred_krig, label_rsm, label_krig,
                r2_rsm, r2_krig, resposta):
    fig = go.Figure()
    lims = [min(y_obs.min(), y_pred_rsm.min(), y_pred_krig.min())*0.97,
            max(y_obs.max(), y_pred_rsm.max(), y_pred_krig.max())*1.03]
    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                             line=dict(color="black", dash="dash", width=2),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=y_obs, y=y_pred_rsm, mode="markers",
        name=f"RSM (R²={r2_rsm:.4f})",
        marker=dict(size=10, color="steelblue", line=dict(color="white",width=1.5)),
        hovertemplate="Obs=%{x:.4f}<br>Pred=%{y:.4f}<extra>RSM</extra>"
    ))
    fig.add_trace(go.Scatter(
        x=y_obs, y=y_pred_krig, mode="markers",
        name=f"Kriging (R²={r2_krig:.4f})",
        marker=dict(size=10, color="tomato", symbol="square",
                    line=dict(color="white",width=1.5)),
        hovertemplate="Obs=%{x:.4f}<br>Pred=%{y:.4f}<extra>Kriging</extra>"
    ))
    fig.update_layout(
        title=dict(text=f"Parity Plot — {label_rsm} vs {label_krig}",
                   font=dict(size=14, color="#1565c0")),
        xaxis=dict(title=f"{resposta.capitalize()} observada", gridcolor="#e0e0e0"),
        yaxis=dict(title=f"{resposta.capitalize()} predita",  gridcolor="#e0e0e0"),
        height=440, margin=dict(l=60,r=40,t=50,b=50),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fafafa",
        legend=dict(bgcolor="rgba(255,255,255,0.85)")
    )
    return fig


def plot_residuos(y_pred_rsm_cv, res_rsm_cv, y_pred_krig_cv, res_krig_cv,
                  kernel_nome, resposta):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Resíduos vs Predito (LOOCV)",
                                        "Distribuição dos Resíduos (LOOCV)"))
    # Scatter resíduos
    fig.add_trace(go.Scatter(x=y_pred_rsm_cv, y=res_rsm_cv, mode="markers",
        name="RSM", marker=dict(size=9, color="steelblue",
                                line=dict(color="white",width=1))), row=1, col=1)
    fig.add_trace(go.Scatter(x=y_pred_krig_cv, y=res_krig_cv, mode="markers",
        name=f"Kriging ({kernel_nome})", marker=dict(size=9, color="tomato", symbol="square",
                                line=dict(color="white",width=1))), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)
    fig.update_xaxes(title_text=f"{resposta.capitalize()} predita", row=1, col=1)
    fig.update_yaxes(title_text="Resíduo", row=1, col=1)
    # Histogramas
    fig.add_trace(go.Histogram(x=res_rsm_cv, name="RSM", nbinsx=8,
        marker_color="steelblue", opacity=0.65), row=1, col=2)
    fig.add_trace(go.Histogram(x=res_krig_cv, name=f"Kriging",  nbinsx=8,
        marker_color="tomato", opacity=0.65), row=1, col=2)
    fig.add_vline(x=0, line_dash="dash", line_color="black", row=1, col=2)
    fig.update_xaxes(title_text="Resíduo", row=1, col=2)
    fig.update_yaxes(title_text="Frequência", row=1, col=2)
    fig.update_layout(height=420, margin=dict(l=50,r=40,t=60,b=50),
                      barmode="overlay",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fafafa",
                      legend=dict(bgcolor="rgba(255,255,255,0.85)"))
    return fig


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:0.5rem 0 1rem;'>
        <div style='font-size:2.5rem;'>🔬</div>
        <div style='font-weight:700; color:#1565c0; font-size:1rem;'>RSM–Kriging Suite</div>
        <div style='font-size:0.75rem; color:#666;'>UEM · Eng. Química · 3DCP Lab</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 Dados de Entrada")
    arquivo = st.file_uploader("Carregar arquivo Excel (.xlsx)", type=["xlsx"],
                               help="Arquivo deve conter colunas: caso, hn, r1 e variáveis resposta.")

    resposta_sel   = None
    kernel_sel     = "Matern 5/2"
    hn_novo        = 16.0
    r1_novo        = 0.95
    colunas_excluir = ["caso", "hn", "r1"]

    if arquivo:
        df_prev = pd.read_excel(arquivo)
        df_prev.columns = df_prev.columns.str.strip()
        variaveis = [c for c in df_prev.columns
                     if c not in colunas_excluir and pd.api.types.is_numeric_dtype(df_prev[c])]
        excluidas = [c for c in df_prev.columns
                     if c not in colunas_excluir and not pd.api.types.is_numeric_dtype(df_prev[c])]

        if excluidas:
            st.info(f"Colunas categóricas removidas: {', '.join(excluidas)}")

        st.markdown("### ⚙️ Configurações do Modelo")
        resposta_sel = st.selectbox("Variável Resposta", variaveis,
                                    help="Coluna numérica a ser modelada")
        kernel_sel   = st.selectbox("Kernel Kriging",
                                    ["Matern 5/2","Matern 3/2","RBF","Rational Quadratic"],
                                    help="Função de correlação espacial do GPR")

        st.markdown("### 📍 Ponto de Análise Local")
        hn_range_data = (float(df_prev["hn"].min()), float(df_prev["hn"].max()))
        r1_range_data = (float(df_prev["r1"].min()), float(df_prev["r1"].max()))
        hn_novo = st.number_input("HN (mm)", value=float(df_prev["hn"].mean()),
                                  min_value=hn_range_data[0]*0.5,
                                  max_value=hn_range_data[1]*1.5, step=0.5, format="%.2f")
        r1_novo = st.number_input("r1 (V/U)", value=float(df_prev["r1"].mean()),
                                  min_value=r1_range_data[0]*0.5,
                                  max_value=r1_range_data[1]*1.5, step=0.01, format="%.3f")

        st.markdown("---")
        executar = st.button("▶  Executar Análise Completa", use_container_width=True)
    else:
        executar = False

# ──────────────────────────────────────────────
# HEADER PRINCIPAL
# ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔬 RSM–Kriging Analysis Suite</h1>
    <p>Plataforma interativa para ajuste, comparação e validação de metamodelos | UEM · 3DCP Lab</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# ESTADO INICIAL
# ──────────────────────────────────────────────
if not arquivo:
    st.markdown("""
    <div class="info-block">
        👈 <b>Carregue um arquivo Excel</b> na barra lateral para iniciar a análise.<br>
        O arquivo deve conter as colunas: <code>caso</code>, <code>hn</code>, <code>r1</code>
        e as variáveis resposta numéricas.
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""**Modelos suportados**
- RSM quadrática completa
- Kriging / GPR (4 kernels)""")
    with col2:
        st.markdown("""**Validação**
- LOOCV (Leave-One-Out)
- 11 métricas estatísticas""")
    with col3:
        st.markdown("""**Gráficos 3D interativos**
- Rotação livre e zoom
- Superfícies RSM e Kriging""")
    st.stop()

# ──────────────────────────────────────────────
# EXECUÇÃO
# ──────────────────────────────────────────────
if executar or "res" in st.session_state:
    if executar:
        arquivo.seek(0)
        df_bytes = arquivo.read()
        with st.spinner("⏳ Ajustando modelos e validação LOOCV… aguarde."):
            res = rodar_analise(df_bytes, resposta_sel, kernel_sel, hn_novo, r1_novo)
        st.session_state["res"]     = res
        st.session_state["resposta"] = resposta_sel
    else:
        res      = st.session_state["res"]
        resposta_sel = st.session_state.get("resposta", "y")

    r = res  # atalho

    # ── KPIs no topo ──────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("RSM — R² LOOCV",  f"{r['r2_rsm_cv']:.4f}")
    with k2:
        st.metric("RSM — RMSE LOOCV", f"{r['rmse_rsm_cv']:.4f}")
    with k3:
        st.metric(f"Kriging — R² LOOCV",   f"{r['r2_krig_cv']:.4f}")
    with k4:
        st.metric(f"Kriging — RMSE LOOCV", f"{r['rmse_krig_cv']:.4f}")

    # ── ABAS PRINCIPAIS ───────────────────────────
    tab_3d, tab_cont, tab_parity, tab_res, tab_metricas, tab_analise, tab_veredicto = st.tabs([
        "🌐 Superfícies 3D",
        "🗺️ Mapas de Contorno",
        "📊 Parity Plots",
        "📉 Resíduos",
        "📋 Métricas",
        "🔎 Análise Local & RSM",
        "🏆 Veredicto Final"
    ])

    # ════════════════════════════════════════════
    # ABA 1 — SUPERFÍCIES 3D
    # ════════════════════════════════════════════
    with tab_3d:
        st.markdown("### 🌐 Superfícies 3D Interativas")
        st.caption("Arraste para rotacionar · Scroll para zoom · Duplo clique para resetar a câmera")

        opcao_3d = st.radio("Exibir superfície:",
                            ["RSM", f"Kriging ({r['kernel_nome']})",
                             "Comparação lado a lado", "Incerteza Kriging (σ)"],
                            horizontal=True)

        if opcao_3d == "RSM":
            fig = plot_3d_surface(
                r["HN_grid"], r["R_grid"], r["Z_rsm"],
                f"Superfície RSM — {resposta_sel.capitalize()}",
                resposta_sel, r["HN"], r["R"], r["y"],
                hn_star=hn_novo, r1_star=r1_novo, y_star=r["y_rsm_local"],
                colorscale="Turbo"
            )
            st.plotly_chart(fig, use_container_width=True, config={"scrollZoom":True})

        elif opcao_3d == f"Kriging ({r['kernel_nome']})":
            fig = plot_3d_surface(
                r["HN_grid"], r["R_grid"], r["Z_krig"],
                f"Superfície Kriging [{r['kernel_nome']}] — {resposta_sel.capitalize()}",
                resposta_sel, r["HN"], r["R"], r["y"],
                hn_star=hn_novo, r1_star=r1_novo, y_star=r["y_krig_local"],
                colorscale="Turbo"
            )
            st.plotly_chart(fig, use_container_width=True, config={"scrollZoom":True})

        elif opcao_3d == "Comparação lado a lado":
            c1, c2 = st.columns(2)
            with c1:
                fig1 = plot_3d_surface(
                    r["HN_grid"], r["R_grid"], r["Z_rsm"],
                    f"RSM — {resposta_sel.capitalize()}",
                    resposta_sel, r["HN"], r["R"], r["y"],
                    hn_star=hn_novo, r1_star=r1_novo, y_star=r["y_rsm_local"],
                    colorscale="Blues"
                )
                st.plotly_chart(fig1, use_container_width=True, config={"scrollZoom":True})
            with c2:
                fig2 = plot_3d_surface(
                    r["HN_grid"], r["R_grid"], r["Z_krig"],
                    f"Kriging [{r['kernel_nome']}] — {resposta_sel.capitalize()}",
                    resposta_sel, r["HN"], r["R"], r["y"],
                    hn_star=hn_novo, r1_star=r1_novo, y_star=r["y_krig_local"],
                    colorscale="Reds"
                )
                st.plotly_chart(fig2, use_container_width=True, config={"scrollZoom":True})

        else:  # Incerteza
            fig = plot_3d_surface(
                r["HN_grid"], r["R_grid"], r["Z_krig"],
                f"Incerteza Kriging (σ) — {resposta_sel.capitalize()}",
                resposta_sel, r["HN"], r["R"], r["y"],
                hn_star=hn_novo, r1_star=r1_novo, y_star=r["y_krig_local"],
                colorscale="Viridis", sigma_surf=r["Z_sigma"]
            )
            st.plotly_chart(fig, use_container_width=True, config={"scrollZoom":True})

    # ════════════════════════════════════════════
    # ABA 2 — MAPAS DE CONTORNO
    # ════════════════════════════════════════════
    with tab_cont:
        st.markdown("### 🗺️ Mapas de Contorno 2D")
        opcao_cont = st.radio("Selecionar mapa:",
                              ["RSM", f"Kriging ({r['kernel_nome']})",
                               "Incerteza σ (Kriging)", "Diferença (Kriging − RSM)"],
                              horizontal=True)
        mapa_dados = {
            "RSM": (r["Z_rsm"], COLORSCALE_CONT),
            f"Kriging ({r['kernel_nome']})": (r["Z_krig"], COLORSCALE_CONT),
            "Incerteza σ (Kriging)": (r["Z_sigma"], COLORSCALE_SIGMA),
            "Diferença (Kriging − RSM)": (r["Z_diff"], "RdBu"),
        }
        Z_sel, cs_sel = mapa_dados[opcao_cont]
        fig_c = plot_contorno(
            r["HN_grid"], r["R_grid"], Z_sel,
            f"{opcao_cont} — {resposta_sel.capitalize()}",
            resposta_sel, r["HN"], r["R"], r["dados"]["caso"].values,
            hn_star=hn_novo, r1_star=r1_novo, colorscale=cs_sel
        )
        st.plotly_chart(fig_c, use_container_width=True)

    # ════════════════════════════════════════════
    # ABA 3 — PARITY PLOTS
    # ════════════════════════════════════════════
    with tab_parity:
        st.markdown("### 📊 Parity Plots")
        modo_parity = st.radio("Visualizar:", ["Treino", "LOOCV", "Ambos"], horizontal=True)

        if modo_parity in ["Treino", "Ambos"]:
            st.markdown("#### Treino")
            fig_pt = plot_parity(
                r["y"], r["y_pred_rsm"], r["y_pred_krig"],
                "RSM Treino", f"Kriging Treino",
                r2_score(r["y"], r["y_pred_rsm"]),
                r2_score(r["y"], r["y_pred_krig"]),
                resposta_sel
            )
            st.plotly_chart(fig_pt, use_container_width=True)

        if modo_parity in ["LOOCV", "Ambos"]:
            st.markdown("#### LOOCV")
            fig_pl = plot_parity(
                r["y"], r["y_pred_rsm_cv"], r["y_pred_krig_cv"],
                "RSM LOOCV", f"Kriging LOOCV",
                r["r2_rsm_cv"], r["r2_krig_cv"],
                resposta_sel
            )
            st.plotly_chart(fig_pl, use_container_width=True)

    # ════════════════════════════════════════════
    # ABA 4 — RESÍDUOS
    # ════════════════════════════════════════════
    with tab_res:
        st.markdown("### 📉 Análise de Resíduos (LOOCV)")
        fig_r = plot_residuos(
            r["y_pred_rsm_cv"], r["res_rsm_cv"],
            r["y_pred_krig_cv"], r["res_krig_cv"],
            r["kernel_nome"], resposta_sel
        )
        st.plotly_chart(fig_r, use_container_width=True)

        with st.expander("📄 Tabela detalhada por caso"):
            st.dataframe(r["resultado"].round(6), use_container_width=True)

    # ════════════════════════════════════════════
    # ABA 5 — MÉTRICAS
    # ════════════════════════════════════════════
    with tab_metricas:
        st.markdown("### 📋 Tabela Comparativa de Métricas")
        met = r["tabela_metricas"].copy()

        def color_r2(val):
            try:
                v = float(val)
                if v >= 0.98: return "background-color:#c8e6c9; color:#1b5e20"
                if v >= 0.90: return "background-color:#fff9c4; color:#f57f17"
                return "background-color:#ffcdd2; color:#b71c1c"
            except: return ""
        def color_rmse(val):
            try:
                v = float(val)
                if v < 0.1: return "background-color:#c8e6c9; color:#1b5e20"
                if v < 0.5: return "background-color:#fff9c4; color:#f57f17"
                return "background-color:#ffcdd2; color:#b71c1c"
            except: return ""

        styled = (met.style
                  .applymap(color_r2,  subset=["R²","Q²"])
                  .applymap(color_rmse, subset=["RMSE"])
                  .format({c: "{:.5f}" for c in met.select_dtypes("number").columns}))
        st.dataframe(styled, use_container_width=True)

        st.markdown("#### Resumo Final")
        st.dataframe(r["resumo_final"], use_container_width=True)

        # Download Excel
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            r["tabela_metricas"].to_excel(writer, sheet_name="Métricas", index=False)
            r["resultado"].to_excel(writer, sheet_name="Por Caso", index=False)
            r["resumo_final"].to_excel(writer, sheet_name="Resumo Final", index=False)
        st.download_button(
            "⬇️ Baixar tabelas em Excel",
            data=buf.getvalue(),
            file_name=f"metricas_{resposta_sel}_{r['kernel_nome'].replace(' ','_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # ════════════════════════════════════════════
    # ABA 6 — ANÁLISE LOCAL & RSM
    # ════════════════════════════════════════════
    with tab_analise:
        st.markdown(f"### 🔎 Análise Local — Ponto P*(HN={hn_novo:.2f}, r1={r1_novo:.3f})")

        col_eq, col_pc = st.columns([3, 2])
        with col_eq:
            st.markdown("#### Equação ajustada da RSM")
            st.code(r["eq_text"], language="python")

        with col_pc:
            st.markdown("#### Ponto Crítico da RSM")
            st.dataframe(r["ponto_critico_df"], use_container_width=True, hide_index=True)

        st.markdown("#### Análise Local Completa no Ponto P*")
        df_loc = r["analise_local_df"].copy()
        st.dataframe(df_loc, use_container_width=True, hide_index=True)

        # Gauge de incerteza local
        sigma_val = r["ponto_local"]["Krig_sigma"]
        ic_inf    = r["ponto_local"]["IC_inf_95"]
        ic_sup    = r["ponto_local"]["IC_sup_95"]
        interp    = r["ponto_local"]["Interpretação"]

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Predição RSM no P*",   f"{r['y_rsm_local']:.5f}")
            st.metric("Predição Kriging no P*", f"{r['y_krig_local']:.5f}")
        with c2:
            st.metric("Incerteza σ (Kriging)", f"{sigma_val:.5f}")
            st.metric("IC 95% Kriging", f"[{ic_inf:.4f} , {ic_sup:.4f}]")

        st.markdown(f"**Avaliação de concordância local:** {interp}")

    # ════════════════════════════════════════════
    # ABA 7 — VEREDICTO FINAL
    # ════════════════════════════════════════════
    with tab_veredicto:
        st.markdown("### 🏆 Veredicto Final e Risco de Overfitting")
        st.dataframe(r["df_overfit"], use_container_width=True, hide_index=True)

        st.markdown("---")
        venc = r["vencedor"]
        if venc == "Kriging":
            cls_v = "verdict-kriging"
            texto_v = f"""
            🏆 <b>Modelo mais confiável: Kriging [{r['kernel_nome']}]</b><br><br>
            O Kriging apresentou superioridade na validação cruzada LOOCV com
            R²={r['r2_krig_cv']:.4f} e RMSE={r['rmse_krig_cv']:.4f}.<br>
            Seu comportamento interpolativo não comprometeu a capacidade de generalização.<br>
            <b>Recomendação: utilize o Kriging como metamodelo primário para esta resposta.</b>
            """
        else:
            cls_v = "verdict-rsm"
            texto_v = f"""
            🏆 <b>Modelo mais confiável: RSM (Superfície de Resposta)</b><br><br>
            A RSM apresentou melhor robustez e menor risco de sobreajuste,
            com R²={r['r2_rsm_cv']:.4f} e RMSE={r['rmse_rsm_cv']:.4f} em LOOCV.<br>
            Sua forma analítica explícita facilita interpretação e otimização.<br>
            <b>Recomendação: utilize a RSM como metamodelo primário para esta resposta.</b>
            """
        st.markdown(f'<div class="verdict-box {cls_v}">{texto_v}</div>',
                    unsafe_allow_html=True)

        # Gráfico de pontuação
        df_sc = r["df_overfit"][["Modelo","Score"]].copy()
        fig_sc = px.bar(df_sc, x="Modelo", y="Score", color="Modelo",
                        title="Pontuação Final dos Modelos",
                        color_discrete_map={
                            df_sc["Modelo"].iloc[0]: "#1565c0",
                            df_sc["Modelo"].iloc[1]: "#c62828"
                        },
                        text="Score")
        fig_sc.update_traces(textposition="outside")
        fig_sc.update_layout(showlegend=False, height=360,
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="#fafafa",
                              yaxis=dict(title="Score", gridcolor="#e0e0e0"))
        st.plotly_chart(fig_sc, use_container_width=True)

# ──────────────────────────────────────────────
# RODAPÉ
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.8rem; padding:0.5rem;'>
    RSM–Kriging Analysis Suite v3.0 · Prof. Dr. Ricardo V. P. Rezende &
    Doutoranda Allana Ribeiro Mendes<br>
    Universidade Estadual de Maringá · Departamento de Engenharia Química · 3DCP Lab
</div>
""", unsafe_allow_html=True)
