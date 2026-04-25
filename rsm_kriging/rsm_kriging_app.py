# -*- coding: utf-8 -*-
"""
RSM-KRIGING ANALYSIS SUITE - STREAMLIT APP v3.1
Universidade Estadual de Maringá (UEM) - Eng. Química / 3DCP Lab
Prof. Dr. Ricardo V. P. Rezende | Doutoranda Allana Ribeiro Mendes

Correções v3.1:
  - .applymap() -> .map()  (pandas >= 2.1)
  - Contornos com aspecto quadrado (scaleanchor)
  - Abas de métricas/análise/veredicto sempre renderizadas
  - Dataset: 26 casos | Fatores: hn, r1
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel, Matern, WhiteKernel, RBF, RationalQuadratic
)

st.set_page_config(page_title="RSM-Kriging Suite | UEM", page_icon="🔬",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
html,body,[class*="css"]{font-family:'Segoe UI',sans-serif;}
.main-header{background:linear-gradient(135deg,#1a237e,#1565c0);padding:1.5rem 2rem;
  border-radius:12px;margin-bottom:1.4rem;color:white;box-shadow:0 4px 20px rgba(26,35,126,.3);}
.main-header h1{margin:0;font-size:1.85rem;font-weight:700;}
.main-header p{margin:.3rem 0 0;opacity:.85;font-size:.88rem;}
.verdict-box{border-radius:12px;padding:1.3rem;font-size:.97rem;font-weight:500;
  margin-top:.8rem;line-height:1.75;}
.verdict-kriging{background:#e8f5e9;border:2px solid #2e7d32;color:#1b5e20;}
.verdict-rsm{background:#e3f2fd;border:2px solid #1565c0;color:#0d47a1;}
.stTabs [data-baseweb="tab-list"]{gap:6px;}
.stTabs [data-baseweb="tab"]{background:#f0f4f8;border-radius:8px 8px 0 0;
  padding:.45rem 1.1rem;font-weight:600;}
.stTabs [aria-selected="true"]{background:#1565c0!important;color:white!important;}
.stButton>button{background:linear-gradient(90deg,#1565c0,#283593);color:white;
  border:none;border-radius:8px;padding:.6rem 1.4rem;font-weight:600;font-size:1rem;width:100%;}
</style>
""", unsafe_allow_html=True)

# ── Constantes ────────────────────────────────────────────────────────────────
EXCLUIR  = ["caso","hn","v","u","r1","r2","regime","estabilidade"]
KERNELS  = ["Matern 5/2","Matern 3/2","RBF","Rational Quadratic"]
UNIDADES = {"largura":"mm","altura":"mm","area":"mm²","ar":"mm²",
            "area_norm":"—","perda_alt":"mm","espalhamento":"mm"}

# ── Funções analíticas ────────────────────────────────────────────────────────

def montar_kernel(nome):
    B = {"Matern 3/2":Matern(nu=1.5),"Matern 5/2":Matern(nu=2.5),
         "RBF":RBF(),"Rational Quadratic":RationalQuadratic()}[nome]
    return ConstantKernel(1.0)*B + WhiteKernel(noise_level=1e-3)


def metricas(y,yp,nome,p=None):
    y,yp = np.asarray(y,float),np.asarray(yp,float)
    res=y-yp; sse=np.sum(res**2); sst=np.sum((y-np.mean(y))**2)
    r2=1-sse/sst if sst>0 else np.nan
    rm=float(np.sqrt(mean_squared_error(y,yp)))
    aic=bic=np.nan
    n=len(y)
    if p and n>p+1 and sse>0:
        aic=n*np.log(sse/n)+2*p; bic=n*np.log(sse/n)+p*np.log(n)
    return {"Modelo":nome,"R²":r2,"RMSE":rm,
            "MAE":mean_absolute_error(y,yp),
            "MAPE (%)":np.mean(np.abs(res)/np.maximum(np.abs(y),1e-12))*100,
            "Bias":float(np.mean(res)),"Erro abs máx":float(np.max(np.abs(res))),
            "PRESS":float(sse),"Q²":r2,"AIC":aic,"BIC":bic}


def ponto_local(hn0,r10,beta,gpr,sc,y,HN,R):
    yr=(beta[0]+beta[1]*hn0+beta[2]*r10+beta[3]*hn0*r10
        +beta[4]*hn0**2+beta[5]*r10**2)
    g=np.array([beta[1]+beta[3]*r10+2*beta[4]*hn0,
                beta[2]+beta[3]*hn0+2*beta[5]*r10])
    H=np.array([[2*beta[4],beta[3]],[beta[3],2*beta[5]]])
    ev=np.linalg.eigvals(H)
    yp,sg=gpr.predict(sc.transform([[hn0,r10]]),return_std=True)
    yk=float(yp[0]); s=float(sg[0])
    dist=np.sqrt((HN-hn0)**2+(R-r10)**2)
    da=abs(yk-yr); dr=100*da/max(abs(yk),1e-12); fy=float(np.max(y)-np.min(y))
    s_=max(s,1e-12); rs=da/s_; ff=da/max(fy,1e-12)
    if   rs<1 and ff<0.03: interp="✅ Excelente concordância local"
    elif rs<2 and ff<0.07: interp="🟢 Boa concordância local"
    elif rs<3 or  ff<0.12: interp="🟡 Concordância moderada"
    else:                  interp="🔴 Conflito entre modelos"
    return {"HN (mm)":round(hn0,4),"r1 (V/U)":round(r10,4),
            "RSM — Predição":round(yr,6),"Kriging — Predição":round(yk,6),
            "Kriging — σ":round(s,6),
            "IC 95% inferior":round(yk-1.96*s,6),"IC 95% superior":round(yk+1.96*s,6),
            "Diferença abs":round(da,6),"Diferença rel (%)":round(dr,4),
            "Razão diff/σ":round(da/s_,4),"σ rel faixa (%)":round(100*s/max(fy,1e-12),4),
            "Dist. mais próximo":round(float(np.min(dist)),4),
            "Caso mais próximo":int(np.argmin(dist)),
            "Grad dY/dHN":round(float(g[0]),6),"Grad dY/dr1":round(float(g[1]),6),
            "Norma grad":round(float(np.linalg.norm(g)),6),
            "Autovalor H1":round(float(ev[0].real),6),"Autovalor H2":round(float(ev[1].real),6),
            "Interpretação":interp}


@st.cache_data(show_spinner=False)
def rodar(df_bytes, resp, kname, hn0, r10):
    df=pd.read_excel(io.BytesIO(df_bytes)); df.columns=df.columns.str.strip()
    HN=df["hn"].values.astype(float); R=df["r1"].values.astype(float)
    y=df[resp].values.astype(float); casos=df["caso"].values

    # RSM
    Xr=np.column_stack([np.ones(len(HN)),HN,R,HN*R,HN**2,R**2])
    mr=LinearRegression(fit_intercept=False).fit(Xr,y); beta=mr.coef_
    ypr=mr.predict(Xr); loo=LeaveOneOut()
    ypr_cv=np.zeros_like(y,float)
    for tr,te in loo.split(Xr):
        ypr_cv[te[0]]=LinearRegression(fit_intercept=False).fit(Xr[tr],y[tr]).predict(Xr[te])[0]

    # Kriging
    sc=StandardScaler(); Xs=sc.fit_transform(df[["hn","r1"]].values.astype(float))
    gpr=GaussianProcessRegressor(kernel=montar_kernel(kname),
        n_restarts_optimizer=20,normalize_y=True,random_state=42).fit(Xs,y)
    ypk,ysk=gpr.predict(Xs,return_std=True)
    ypk_cv=np.zeros_like(y,float)
    for tr,te in loo.split(Xs):
        ypk_cv[te[0]]=GaussianProcessRegressor(kernel=montar_kernel(kname),
            n_restarts_optimizer=10,normalize_y=True,random_state=42).fit(Xs[tr],y[tr]).predict(Xs[te])[0]

    # Métricas
    tab_met=pd.DataFrame([metricas(y,ypr,"RSM (Treino)",p=6),
                          metricas(y,ypr_cv,"RSM (LOOCV)",p=6),
                          metricas(y,ypk,f"Kriging ({kname}) Treino"),
                          metricas(y,ypk_cv,f"Kriging ({kname}) LOOCV")])

    # Malha
    m=0.10
    hg=np.linspace(HN.min()*(1-m),HN.max()*(1+m),80)
    rg=np.linspace(R.min()*(1-m),R.max()*(1+m),80)
    HG,RG=np.meshgrid(hg,rg)
    Zr=beta[0]+beta[1]*HG+beta[2]*RG+beta[3]*HG*RG+beta[4]*HG**2+beta[5]*RG**2
    Xg=np.column_stack([HG.ravel(),RG.ravel()])
    Zk_,Zs_=gpr.predict(sc.transform(Xg),return_std=True)
    Zk=Zk_.reshape(HG.shape); Zs=Zs_.reshape(HG.shape); Zd=Zk-Zr

    # Ponto local
    pl=ponto_local(hn0,r10,beta,gpr,sc,y,HN,R)
    yrl=pl["RSM — Predição"]; ykl=pl["Kriging — Predição"]

    # Equação
    eq=(f"{resp}(HN,r1) = {beta[0]:.5f} {beta[1]:+.5f}·HN"
        f" {beta[2]:+.5f}·r1 {beta[3]:+.5f}·HN·r1"
        f" {beta[4]:+.5f}·HN² {beta[5]:+.5f}·r1²")

    # Ponto crítico
    try:
        A=np.array([[2*beta[4],beta[3]],[beta[3],2*beta[5]]])
        pc=np.linalg.solve(A,np.array([-beta[1],-beta[2]]))
        Hs,Rs=pc
        ys=beta[0]+beta[1]*Hs+beta[2]*Rs+beta[3]*Hs*Rs+beta[4]*Hs**2+beta[5]*Rs**2
        ev=np.linalg.eigvals(A)
        if np.all(ev.real>0): cp="🟢 Mínimo local"
        elif np.all(ev.real<0): cp="🔴 Máximo local"
        else: cp="🟡 Ponto de sela"
        pc_df=pd.DataFrame({"Parâmetro":["HN*","r1*",f"{resp}(HN*,r1*)",
                            "Det(H)","Autovalor 1","Autovalor 2","Classificação",
                            "HN* na faixa?","r1* na faixa?"],
                            "Valor":[round(Hs,4),round(Rs,4),round(float(ys),4),
                            round(float(np.linalg.det(A)),4),round(float(ev[0].real),4),
                            round(float(ev[1].real),4),cp,
                            bool(HN.min()<=Hs<=HN.max()),bool(R.min()<=Rs<=R.max())]})
    except Exception:
        pc_df=pd.DataFrame({"Parâmetro":["Erro"],"Valor":["Sistema singular"]})

    # Overfitting
    def _r(a,b): return float(r2_score(a,b))
    def _m(a,b): return float(np.sqrt(mean_squared_error(a,b)))
    r2rt=_r(y,ypr); r2rc=_r(y,ypr_cv); r2kt=_r(y,ypk); r2kc=_r(y,ypk_cv)
    mrt=_m(y,ypr);  mrc=_m(y,ypr_cv);  mkt=_m(y,ypk);  mkc=_m(y,ypk_cv)
    def risco(dr2,dr):
        if dr2<0.05 and dr<1.5: return "🟢 Baixo"
        elif dr2<0.15 and dr<4: return "🟡 Moderado"
        else: return "🔴 Alto"
    rr=risco(r2rt-r2rc,mrc-mrt); rk=risco(r2kt-r2kc,mkc-mkt)
    sr=(2 if mrc<mkc else 0)+(1 if r2rc>r2kc else 0)-(1 if "Alto" in rr else 0)
    sk=(2 if mkc<=mrc else 0)+(1 if r2kc>=r2rc else 0)-(1 if "Alto" in rk else 0)
    venc="RSM" if sr>sk else "Kriging"

    of=pd.DataFrame({"Modelo":["RSM",f"Kriging ({kname})"],
        "R² Treino":[round(r2rt,5),round(r2kt,5)],"R² LOOCV":[round(r2rc,5),round(r2kc,5)],
        "ΔR²":[round(r2rt-r2rc,5),round(r2kt-r2kc,5)],
        "RMSE Treino":[round(mrt,5),round(mkt,5)],"RMSE LOOCV":[round(mrc,5),round(mkc,5)],
        "ΔRMSE":[round(mrc-mrt,5),round(mkc-mkt,5)],
        "Risco":[rr,rk],"Score":[sr,sk]})

    res_rc=y-ypr_cv; res_kc=y-ypk_cv
    resultado=pd.DataFrame({"caso":casos,"hn":HN,"r1":R,resp:y,
        "RSM_treino":ypr,"RSM_LOOCV":ypr_cv,"RSM_res_LOOCV":res_rc,
        "Krig_treino":ypk,"Krig_σ":ysk,"Krig_LOOCV":ypk_cv,"Krig_res_LOOCV":res_kc})

    resumo=pd.DataFrame({"Parâmetro":["Kernel","RSM R² treino","RSM RMSE treino",
        "RSM R² LOOCV","RSM RMSE LOOCV","Kriging R² treino","Kriging RMSE treino",
        "Kriging R² LOOCV","Kriging RMSE LOOCV","σ médio","RSM em P*","Kriging em P*",
        "Diff abs P*","🏆 Vencedor"],
        "Valor":[kname,round(r2rt,6),round(mrt,6),round(r2rc,6),round(mrc,6),
                 round(r2kt,6),round(mkt,6),round(r2kc,6),round(mkc,6),
                 round(float(Zs.mean()),6),yrl,ykl,round(abs(yrl-ykl),6),venc]})

    al_df=pd.DataFrame({"Parâmetro":list(pl.keys()),"Valor":list(pl.values())})

    return dict(HN=HN,R=R,y=y,casos=casos,beta=beta,eq=eq,
                ypr=ypr,ypr_cv=ypr_cv,ypk=ypk,ypk_cv=ypk_cv,ysk=ysk,
                res_rc=res_rc,res_kc=res_kc,
                HG=HG,RG=RG,Zr=Zr,Zk=Zk,Zs=Zs,Zd=Zd,
                yrl=yrl,ykl=ykl,pl=pl,
                tab_met=tab_met,resultado=resultado,resumo=resumo,
                al_df=al_df,pc_df=pc_df,of=of,venc=venc,kname=kname,
                r2rc=r2rc,r2kc=r2kc,mrc=mrc,mkc=mkc,r2rt=r2rt,r2kt=r2kt)

# ── Plotagem ─────────────────────────────────────────────────────────────────

def fig3d(HG,RG,Z,titulo,resp,unid,HN,R,y,casos,
          hn_s=None,r1_s=None,y_s=None,cs="Blues",sc=None,
          show_grid=True,show_spheres=False,show_stems=False,
          camera_eye=None):
    """
    Superfície 3D com opções de customização:
      show_grid    — grade NA superfície (wireframe leve), caixa sempre visível
      show_spheres — shading esférico sobre cada ponto (ilusão 3D real)
      show_stems   — linha tracejada vermelha de cada ponto até a superfície
      camera_eye   — dict(x,y,z) para câmera sincronizada
    """
    fig = go.Figure()

    # ── Eixos: caixa sempre ligada, grade da superfície opcional ──
    ax_base = dict(showticklabels=True, zeroline=False,
                   backgroundcolor="#f0f3f8", showbackground=True,
                   gridcolor="#c8d0e0")
    ax_kw = {
        "xaxis": dict(title="HN (mm)",   **ax_base),
        "yaxis": dict(title="r1 (V/U)",  **ax_base),
        "zaxis": dict(title=f"{resp} ({unid})", **ax_base),
    }

    # ── Superfície principal ──────────────────────────────────────
    # Iluminação suave: ambient alto, specular baixo → sem regiões estouradas
    surf_kw = dict(
        opacity   = 0.88,
        colorscale= cs,
        showscale = True,
        lighting  = dict(ambient=0.75, diffuse=0.55, specular=0.08,
                         roughness=0.85, fresnel=0.1),
        lightposition = dict(x=100, y=200, z=300),
        colorbar  = dict(title=f"{resp}<br>({unid})", thickness=14, len=0.70),
    )
    if sc is not None:
        surf_kw["surfacecolor"] = sc
        surf_kw["colorbar"]["title"] = "σ"

    # Grade NA superfície: contorno de nível leve em branco semitransparente
    if show_grid:
        surf_kw["contours"] = dict(
            x=dict(show=True, color="rgba(255,255,255,0.25)", width=1),
            y=dict(show=True, color="rgba(255,255,255,0.25)", width=1),
            z=dict(show=False),
        )
    fig.add_trace(go.Surface(x=HG, y=RG, z=Z, **surf_kw))

    # ── Pontos experimentais com shading esférico ─────────────────
    # Shading via colorscale radial: centro claro → borda escura
    # dá ilusão de esfera sem adicionar geometria extra
    sphere_cs = [
        [0.0, "rgba(255,255,255,0.95)"],   # centro: branco brilhante
        [0.4, "rgba(220,230,255,0.80)"],   # meio: azul claro
        [1.0, "rgba(60,100,180,0.40)"],    # borda: azul escuro transparente
    ] if show_spheres else None

    if show_spheres:
        # Um Scatter3d por ponto, com símbolo circle e colorscale simulando esfera
        for xi, ri, zi, ci in zip(HN, R, y, casos):
            fig.add_trace(go.Scatter3d(
                x=[xi], y=[ri], z=[zi], mode="markers",
                marker=dict(
                    size=16,
                    color=[0.3],           # posição no colorscale (parte clara)
                    colorscale=sphere_cs,
                    opacity=0.75,
                    line=dict(color="rgba(40,80,160,0.6)", width=1.5),
                ),
                showlegend=False, hoverinfo="skip",
            ))
        # Pontos principais (brancos com borda) por cima
        fig.add_trace(go.Scatter3d(x=HN, y=R, z=y, mode="markers+text",
            text=[str(c) for c in casos], textposition="top center",
            textfont=dict(size=9, color="black"),
            marker=dict(size=6, color="white", line=dict(color="#1a3a7a", width=2)),
            name="Experimental",
            hovertemplate="<b>%{text}</b><br>HN=%{x:.2f}<br>r1=%{y:.3f}<br>y=%{z:.4f}<extra></extra>"))
    else:
        fig.add_trace(go.Scatter3d(x=HN, y=R, z=y, mode="markers+text",
            text=[str(c) for c in casos], textposition="top center",
            textfont=dict(size=9, color="black"),
            marker=dict(size=7, color="white", line=dict(color="black", width=2)),
            name="Experimental",
            hovertemplate="<b>%{text}</b><br>HN=%{x:.2f}<br>r1=%{y:.3f}<br>y=%{z:.4f}<extra></extra>"))

    # ── Hastes verticais até a superfície ────────────────────────
    if show_stems:
        from scipy.interpolate import RegularGridInterpolator
        interp_fn = RegularGridInterpolator(
            (HG[0], RG[:,0]), Z.T, method="linear",
            bounds_error=False, fill_value=None)
        xs_all, ys_all, zs_all = [], [], []
        for xi, ri, zi in zip(HN, R, y):
            z_surf = float(interp_fn([[xi, ri]])[0])
            xs_all += [xi, xi, None]
            ys_all += [ri, ri, None]
            zs_all += [zi, z_surf, None]
        fig.add_trace(go.Scatter3d(
            x=xs_all, y=ys_all, z=zs_all,
            mode="lines",
            line=dict(color="red", width=2.5, dash="dash"),
            name="Haste → superfície",
            hoverinfo="skip",
        ))

    # ── Ponto P* ─────────────────────────────────────────────────
    if hn_s is not None:
        fig.add_trace(go.Scatter3d(
            x=[hn_s], y=[r1_s], z=[y_s], mode="markers",
            marker=dict(size=10, color="red", symbol="diamond",
                        line=dict(color="black", width=1.5)),
            name="P*",
            hovertemplate=f"P*<br>HN={hn_s:.2f}<br>r1={r1_s:.3f}<br>y={y_s:.4f}<extra></extra>"))

    eye = camera_eye if camera_eye else dict(x=1.6, y=-1.6, z=1.2)
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=15, color="#1565c0")),
        scene=dict(camera=dict(eye=eye), aspectmode="auto", **ax_kw),
        height=610, margin=dict(l=0, r=0, t=58, b=0),
        legend=dict(yanchor="top", y=0.97, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="#ccc", borderwidth=1),
        paper_bgcolor="rgba(0,0,0,0)")
    return fig


def fig_cont(HG,RG,Z,titulo,resp,unid,HN,R,casos,hn_s=None,r1_s=None,cs="Jet"):
    fig=go.Figure()
    fig.add_trace(go.Contour(x=HG[0],y=RG[:,0],z=Z,colorscale=cs,ncontours=18,showscale=True,
        contours=dict(showlabels=True,labelfont=dict(size=9,color="black")),
        colorbar=dict(title=f"{resp} ({unid})",thickness=14)))
    fig.add_trace(go.Scatter(x=HN,y=R,mode="markers+text",
        text=[str(c) for c in casos],textposition="top right",textfont=dict(size=9),
        marker=dict(size=10,color="white",line=dict(color="black",width=2)),name="Experimental",
        hovertemplate="<b>%{text}</b><br>HN=%{x:.2f}<br>r1=%{y:.3f}<extra></extra>"))
    if hn_s is not None:
        fig.add_trace(go.Scatter(x=[hn_s],y=[r1_s],mode="markers+text",
            text=["P*"],textposition="top right",textfont=dict(size=11,color="red"),
            marker=dict(size=14,color="red",symbol="star",line=dict(color="black",width=1.5)),name="P*"))
    # Aspecto 1:1 real — mesmo mecanismo do parity plot que funcionou
    xvals = HG[0]; yvals = RG[:,0]
    xpad = (xvals.max()-xvals.min())*0.05; ypad = (yvals.max()-yvals.min())*0.05
    xlo=xvals.min()-xpad; xhi=xvals.max()+xpad
    ylo=yvals.min()-ypad; yhi=yvals.max()+ypad
    fig.update_layout(
        title=dict(text=titulo,font=dict(size=14,color="#1565c0")),
        xaxis=dict(title="HN (mm)",gridcolor="#e0e0e0",
                   range=[xlo,xhi], constrain="domain"),
        yaxis=dict(title="r1 (V/U)",gridcolor="#e0e0e0",
                   range=[ylo,yhi], scaleanchor="x", scaleratio=1, constrain="domain"),
        height=560, margin=dict(l=75,r=30,t=60,b=75),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#fafafa",
        legend=dict(bgcolor="rgba(255,255,255,0.85)"))
    return fig


def fig_parity(yobs,ypr,ypk,r2r,r2k,resp,unid,modo):
    # Range igual em X e Y: sem espaço vazio lateral
    av  = np.concatenate([yobs,ypr,ypk])
    pad = (av.max()-av.min())*0.05
    lo  = av.min()-pad; hi = av.max()+pad
    lims= [lo, hi]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=lims,y=lims,mode="lines",
        line=dict(color="black",dash="dash",width=2),showlegend=False,hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=yobs,y=ypr,mode="markers",name=f"RSM (R²={r2r:.4f})",
        marker=dict(size=10,color="#1565c0",line=dict(color="white",width=1.5)),
        hovertemplate="Obs=%{x:.4f}<br>Pred=%{y:.4f}<extra>RSM</extra>"))
    fig.add_trace(go.Scatter(x=yobs,y=ypk,mode="markers",name=f"Kriging (R²={r2k:.4f})",
        marker=dict(size=10,color="tomato",symbol="square",line=dict(color="white",width=1.5)),
        hovertemplate="Obs=%{x:.4f}<br>Pred=%{y:.4f}<extra>Kriging</extra>"))
    fig.update_layout(
        title=dict(text=f"Parity Plot — {modo} | {resp} ({unid})",font=dict(size=14,color="#1565c0")),
        xaxis=dict(title=f"{resp} observado ({unid})",gridcolor="#e0e0e0",
                   range=lims, constrain="domain"),
        yaxis=dict(title=f"{resp} predito ({unid})",gridcolor="#e0e0e0",
                   range=lims, scaleanchor="x", scaleratio=1, constrain="domain"),
        height=520,
        margin=dict(l=75,r=30,t=60,b=75),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#fafafa",
        legend=dict(bgcolor="rgba(255,255,255,0.85)",
                    yanchor="top",y=0.97,xanchor="left",x=0.03))
    return fig


def fig_res(ypr_cv,res_rc,ypk_cv,res_kc,kname,resp,unid):
    fig=make_subplots(rows=1,cols=2,
        subplot_titles=("Resíduos vs Predito (LOOCV)","Distribuição dos Resíduos (LOOCV)"))
    fig.add_trace(go.Scatter(x=ypr_cv,y=res_rc,mode="markers",name="RSM",
        marker=dict(size=9,color="#1565c0",line=dict(color="white",width=1))),row=1,col=1)
    fig.add_trace(go.Scatter(x=ypk_cv,y=res_kc,mode="markers",name=f"Kriging ({kname})",
        marker=dict(size=9,color="tomato",symbol="square",line=dict(color="white",width=1))),row=1,col=1)
    fig.add_hline(y=0,line_dash="dash",line_color="black",row=1,col=1)
    fig.update_xaxes(title_text=f"{resp} predito ({unid})",row=1,col=1)
    fig.update_yaxes(title_text="Resíduo",row=1,col=1)
    fig.add_trace(go.Histogram(x=res_rc,name="RSM",nbinsx=8,marker_color="#1565c0",opacity=0.65),row=1,col=2)
    fig.add_trace(go.Histogram(x=res_kc,name="Kriging",nbinsx=8,marker_color="tomato",opacity=0.65),row=1,col=2)
    fig.add_vline(x=0,line_dash="dash",line_color="black",row=1,col=2)
    fig.update_xaxes(title_text="Resíduo",row=1,col=2)
    fig.update_yaxes(title_text="Frequência",row=1,col=2)
    fig.update_layout(height=430,margin=dict(l=55,r=40,t=60,b=50),barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#fafafa",
        legend=dict(bgcolor="rgba(255,255,255,0.85)"))
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style='text-align:center;padding:.4rem 0 1rem;'>
        <div style='font-size:2.4rem;'>🔬</div>
        <div style='font-weight:700;color:#1565c0;font-size:1rem;'>RSM-Kriging Suite</div>
        <div style='font-size:.75rem;color:#666;'>UEM · Eng. Química · 3DCP Lab</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 📂 Dados de Entrada")
    arquivo=st.file_uploader("Carregar Excel (.xlsx)",type=["xlsx"],
        help="Colunas obrigatórias: caso, hn, r1 + respostas numéricas.")

    resposta_sel=kernel_sel=hn_novo=r1_novo=None; executar=False

    if arquivo:
        df0=pd.read_excel(arquivo); df0.columns=df0.columns.str.strip()
        variaveis=[c for c in df0.columns
                   if c not in EXCLUIR and pd.api.types.is_numeric_dtype(df0[c])]

        st.markdown("### ⚙️ Modelo")
        resposta_sel=st.selectbox("Variável Resposta",variaveis)
        kernel_sel=st.selectbox("Kernel Kriging",KERNELS)

        st.markdown("### 📍 Ponto de Análise (P*)")
        hn_novo=st.number_input("HN (mm)",
            value=round(float(df0["hn"].mean()),2),
            min_value=round(float(df0["hn"].min())*0.5,2),
            max_value=round(float(df0["hn"].max())*1.5,2),
            step=0.5,format="%.2f")
        r1_novo=st.number_input("r1 (V/U)",
            value=round(float(df0["r1"].mean()),3),
            min_value=round(float(df0["r1"].min())*0.5,3),
            max_value=round(float(df0["r1"].max())*1.5,3),
            step=0.01,format="%.3f")
        st.markdown("---")
        executar=st.button("▶  Executar Análise Completa",use_container_width=True)
        st.markdown("---")
        st.markdown(f"""<div style='font-size:.78rem;color:#555;'>
            <b>Dataset:</b> {df0.shape[0]} casos<br>
            HN: {df0["hn"].min():.1f}–{df0["hn"].max():.1f} mm<br>
            r1: {df0["r1"].min():.3f}–{df0["r1"].max():.3f}<br>
            Respostas: {", ".join(variaveis)}</div>""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""<div class="main-header">
    <h1>🔬 RSM-Kriging Analysis Suite</h1>
    <p>Plataforma interativa para ajuste, comparação e validação de metamodelos ·
       UEM · 3DCP Lab · Prof. Dr. Ricardo V. P. Rezende &amp; Doutoranda Allana Ribeiro Mendes</p>
</div>""", unsafe_allow_html=True)

if not arquivo:
    st.markdown("""<div style='background:#e3f2fd;border-radius:8px;padding:.9rem 1.1rem;
        border-left:4px solid #1565c0;font-size:.88rem;color:#0d47a1;'>
        👈 <b>Carregue o arquivo Excel</b> na barra lateral para iniciar.<br>
        Estrutura: <code>caso, hn, r1</code> + respostas numéricas
        (<code>largura, altura, area, ar, area_norm, perda_alt, espalhamento</code>).
    </div>""", unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: st.markdown("**Metamodelos**\n- RSM quadrática completa\n- Kriging/GPR (4 kernels)")
    with c2: st.markdown("**Validação**\n- LOOCV (Leave-One-Out)\n- 11 métricas estatísticas")
    with c3: st.markdown("**Visualizações**\n- 3D interativo e rotacionável\n- Contornos proporcionais")
    st.stop()

# ── Execução ──────────────────────────────────────────────────────────────────
if executar or "res" in st.session_state:
    if executar:
        arquivo.seek(0); raw=arquivo.read()
        with st.spinner("⏳ Ajustando RSM e Kriging + LOOCV… aguarde."):
            res=rodar(raw,resposta_sel,kernel_sel,hn_novo,r1_novo)
        st.session_state.update({"res":res,"resp":resposta_sel,
                                  "hn0":hn_novo,"r10":r1_novo})
    else:
        res=st.session_state["res"]
        resposta_sel=st.session_state.get("resp","y")
        hn_novo=st.session_state.get("hn0",15.0)
        r1_novo=st.session_state.get("r10",0.9)

    d=res; unid=UNIDADES.get(resposta_sel,"—")

    # KPIs
    k1,k2,k3,k4=st.columns(4)
    k1.metric("RSM — R² LOOCV",      f"{d['r2rc']:.4f}")
    k2.metric("RSM — RMSE LOOCV",    f"{d['mrc']:.4f}")
    k3.metric("Kriging — R² LOOCV",  f"{d['r2kc']:.4f}")
    k4.metric("Kriging — RMSE LOOCV",f"{d['mkc']:.4f}")

    tabs=st.tabs(["🌐 Superfícies 3D","🗺️ Contornos","📊 Parity Plots",
                  "📉 Resíduos","📋 Métricas","🔎 Análise Local","🏆 Veredicto"])

    # ── ABA 1: 3D ─────────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("### 🌐 Superfícies 3D Interativas")
        st.caption("Arraste para rotacionar · Scroll para zoom · Duplo clique para resetar câmera")

        # ── Seletor de superfície ──────────────────────────────────────────────
        modo=st.radio("Exibir:",["RSM",f"Kriging ({d['kname']})","Lado a lado","Incerteza σ"],
                      horizontal=True,key="m3d")

        # ── Opções de customização 3D ──────────────────────────────────────────
        with st.expander("⚙️ Opções de visualização 3D", expanded=True):
            oc1,oc2,oc3,oc4 = st.columns(4)
            opt_grade   = oc1.checkbox("📐 Grade na superfície", value=False, key="opt_grade",
                                        help="Linhas de wireframe suaves sobre a superfície")
            opt_lock    = oc2.checkbox("🔒 Sincronizar câmera",  value=False, key="opt_lock",
                                        help="Lado a lado: mesmo ângulo fixo nos dois gráficos")
            opt_esferas = oc3.checkbox("🔵 Shading esférico",    value=False, key="opt_esferas",
                                        help="Adiciona halo com gradiente radial em cada ponto, simulando volume esférico")
            opt_hastes  = oc4.checkbox("📌 Hastes até superfície",value=False, key="opt_hastes",
                                        help="Linha tracejada vermelha de cada ponto até a superfície ajustada")

            # Sliders de câmera (usados quando lock está ativo)
            if opt_lock:
                st.markdown("**Ângulo da câmera sincronizada:**")
                sc1,sc2,sc3 = st.columns(3)
                cam_x = sc1.slider("X",  min_value=-3.0, max_value=3.0, value=1.6,  step=0.1, key="cam_x")
                cam_y = sc2.slider("Y",  min_value=-3.0, max_value=3.0, value=-1.6, step=0.1, key="cam_y")
                cam_z = sc3.slider("Z",  min_value= 0.2, max_value=3.0, value=1.2,  step=0.1, key="cam_z")
                cam_lock = dict(x=cam_x, y=cam_y, z=cam_z)
            else:
                cam_lock = dict(x=1.6, y=-1.6, z=1.2)

        kw3d = dict(
            show_grid    = opt_grade,
            show_spheres = opt_esferas,
            show_stems   = opt_hastes,
        )

        if modo=="RSM":
            st.plotly_chart(fig3d(d["HG"],d["RG"],d["Zr"],f"RSM — {resposta_sel}",
                resposta_sel,unid,d["HN"],d["R"],d["y"],d["casos"],
                hn_novo,r1_novo,d["yrl"],cs="Blues",**kw3d),
                use_container_width=True,config={"scrollZoom":True})

        elif modo==f"Kriging ({d['kname']})":
            st.plotly_chart(fig3d(d["HG"],d["RG"],d["Zk"],
                f"Kriging [{d['kname']}] — {resposta_sel}",
                resposta_sel,unid,d["HN"],d["R"],d["y"],d["casos"],
                hn_novo,r1_novo,d["ykl"],cs="Reds",**kw3d),
                use_container_width=True,config={"scrollZoom":True})

        elif modo=="Lado a lado":
            if opt_lock:
                st.info("🔒 Câmera sincronizada — ambos os gráficos partem do mesmo ponto de vista. "
                        "Ajuste a câmera em qualquer um e re-execute para fixar o ângulo desejado.")
            c1,c2=st.columns(2)
            cam = cam_lock if opt_lock else None
            with c1:
                st.plotly_chart(fig3d(d["HG"],d["RG"],d["Zr"],f"RSM — {resposta_sel}",
                    resposta_sel,unid,d["HN"],d["R"],d["y"],d["casos"],
                    hn_novo,r1_novo,d["yrl"],cs="Blues",camera_eye=cam,**kw3d),
                    use_container_width=True,config={"scrollZoom":True})
            with c2:
                st.plotly_chart(fig3d(d["HG"],d["RG"],d["Zk"],f"Kriging — {resposta_sel}",
                    resposta_sel,unid,d["HN"],d["R"],d["y"],d["casos"],
                    hn_novo,r1_novo,d["ykl"],cs="Reds",camera_eye=cam,**kw3d),
                    use_container_width=True,config={"scrollZoom":True})

        else:  # Incerteza σ
            st.plotly_chart(fig3d(d["HG"],d["RG"],d["Zk"],f"Incerteza σ — {resposta_sel}",
                resposta_sel,unid,d["HN"],d["R"],d["y"],d["casos"],
                hn_novo,r1_novo,d["ykl"],cs="Viridis",sc=d["Zs"],**kw3d),
                use_container_width=True,config={"scrollZoom":True})

    # ── ABA 2: Contornos ──────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("### 🗺️ Mapas de Contorno 2D")
        st.caption("Aspecto 1:1 — proporção real entre HN e r1")

        ca, cb = st.columns([3,1])
        with ca:
            opcoes_cont = ["RSM", f"Kriging ({d['kname']})", "Incerteza σ", "Diferença (Kriging−RSM)"]
            mc = st.radio("Mapa:", opcoes_cont, horizontal=True, key="mc")
        with cb:
            PALETAS = {
                "🌈 Jet":        "Jet",
                "🔥 Inferno":    "Inferno",
                "🌊 Viridis":    "Viridis",
                "🎨 RdBu":       "RdBu",
                "🟫 YlOrRd":     "YlOrRd",
            }
            paleta_nome = st.selectbox("Paleta de cores", list(PALETAS.keys()), key="paleta_cont")
            cs_custom = PALETAS[paleta_nome]

        # Mapa de Z: usa paleta escolhida para RSM e Kriging; fixa Viridis/RdBu para σ e diff
        Z_map = {
            "RSM":                        (d["Zr"], cs_custom),
            f"Kriging ({d['kname']})":    (d["Zk"], cs_custom),
            "Incerteza σ":                (d["Zs"], "Viridis"),
            "Diferença (Kriging−RSM)":    (d["Zd"], "RdBu"),
        }
        Zsel, cssel = Z_map[mc]
        st.plotly_chart(fig_cont(d["HG"],d["RG"],Zsel,
            f"{mc} — {resposta_sel} ({unid})",resposta_sel,unid,
            d["HN"],d["R"],d["casos"],hn_novo,r1_novo,cssel),
            use_container_width=True)

    # ── ABA 3: Parity ─────────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("### 📊 Parity Plots")
        mp=st.radio("Conjunto:",["Treino","LOOCV"],horizontal=True,key="mp")
        if mp=="Treino":
            st.plotly_chart(fig_parity(d["y"],d["ypr"],d["ypk"],
                d["r2rt"],d["r2kt"],resposta_sel,unid,"Treino"),use_container_width=True)
        else:
            st.plotly_chart(fig_parity(d["y"],d["ypr_cv"],d["ypk_cv"],
                d["r2rc"],d["r2kc"],resposta_sel,unid,"LOOCV"),use_container_width=True)

    # ── ABA 4: Resíduos ───────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("### 📉 Resíduos LOOCV")
        st.plotly_chart(fig_res(d["ypr_cv"],d["res_rc"],d["ypk_cv"],d["res_kc"],
            d["kname"],resposta_sel,unid),use_container_width=True)
        with st.expander("📄 Tabela detalhada por caso"):
            st.dataframe(d["resultado"].round(5),use_container_width=True,hide_index=True)

    # ── ABA 5: Métricas ───────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("### 📋 Métricas Comparativas")
        met=d["tab_met"].copy()
        num=met.select_dtypes(include="number").columns.tolist()

        def cr2(v):
            try:
                v=float(v)
                if v>=0.98: return "background-color:#c8e6c9;color:#1b5e20"
                if v>=0.90: return "background-color:#fff9c4;color:#7c5900"
                return "background-color:#ffcdd2;color:#b71c1c"
            except: return ""

        def crm(v):
            try:
                v=float(v)
                if v<0.5:  return "background-color:#c8e6c9;color:#1b5e20"
                if v<2.0:  return "background-color:#fff9c4;color:#7c5900"
                return "background-color:#ffcdd2;color:#b71c1c"
            except: return ""

        # Compatível com pandas antigo (.applymap) e novo (.map)
        try:
            styled=(met.style
                    .map(cr2,subset=["R²","Q²"])
                    .map(crm,subset=["RMSE"])
                    .format({c:"{:.5f}" for c in num}))
        except AttributeError:
            styled=(met.style
                    .applymap(cr2,subset=["R²","Q²"])
                    .applymap(crm,subset=["RMSE"])
                    .format({c:"{:.5f}" for c in num}))

        st.dataframe(styled,use_container_width=True,hide_index=True)
        st.markdown("#### Resumo Final")
        st.dataframe(d["resumo"],use_container_width=True,hide_index=True)

        buf=io.BytesIO()
        with pd.ExcelWriter(buf,engine="openpyxl") as w:
            d["tab_met"].to_excel(w,sheet_name="Métricas",index=False)
            d["resultado"].to_excel(w,sheet_name="Por Caso",index=False)
            d["resumo"].to_excel(w,sheet_name="Resumo Final",index=False)
        st.download_button("⬇️ Baixar tabelas em Excel",data=buf.getvalue(),
            file_name=f"metricas_{resposta_sel}_{d['kname'].replace(' ','_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ── ABA 6: Análise Local ──────────────────────────────────────────────────
    with tabs[5]:
        st.markdown(f"### 🔎 Análise Local — P*(HN={hn_novo:.2f} mm, r1={r1_novo:.3f})")
        c1,c2=st.columns([3,2])
        with c1:
            st.markdown("#### Equação RSM ajustada")
            st.code(d["eq"],language="text")
        with c2:
            st.markdown("#### Ponto Crítico RSM")
            st.dataframe(d["pc_df"],use_container_width=True,hide_index=True)
        st.markdown("#### Análise Completa no Ponto P*")
        st.dataframe(d["al_df"],use_container_width=True,hide_index=True)
        m1,m2,m3,m4=st.columns(4)
        sig_v=d["pl"]["Kriging — σ"]
        ic_i=d["pl"]["IC 95% inferior"]; ic_s=d["pl"]["IC 95% superior"]
        m1.metric("RSM em P*",    f"{d['yrl']:.5f} {unid}")
        m2.metric("Kriging em P*",f"{d['ykl']:.5f} {unid}")
        m3.metric("Incerteza σ",  f"{sig_v:.5f} {unid}")
        m4.metric("IC 95%",       f"[{ic_i:.3f}, {ic_s:.3f}]")
        st.info(f"**Concordância local:** {d['pl']['Interpretação']}")

    # ── ABA 7: Veredicto ──────────────────────────────────────────────────────
    with tabs[6]:
        st.markdown("### 🏆 Veredicto Final")
        st.dataframe(d["of"],use_container_width=True,hide_index=True)
        st.markdown("---")
        venc=d["venc"]
        if venc=="Kriging":
            cls_v,txt_v="verdict-kriging",(
                f"🏆 <b>Modelo mais confiável: Kriging [{d['kname']}]</b><br><br>"
                f"R² LOOCV = {d['r2kc']:.4f} · RMSE LOOCV = {d['mkc']:.4f}<br>"
                f"Comportamento interpolativo não comprometeu a generalização.<br>"
                f"<b>Recomendação: Kriging como metamodelo primário para {resposta_sel}.</b>")
        else:
            cls_v,txt_v="verdict-rsm",(
                f"🏆 <b>Modelo mais confiável: RSM</b><br><br>"
                f"R² LOOCV = {d['r2rc']:.4f} · RMSE LOOCV = {d['mrc']:.4f}<br>"
                f"Maior robustez e menor risco de sobreajuste.<br>"
                f"<b>Recomendação: RSM como metamodelo primário para {resposta_sel}.</b>")
        st.markdown(f'<div class="verdict-box {cls_v}">{txt_v}</div>',unsafe_allow_html=True)
        fig_sc=px.bar(d["of"],x="Modelo",y="Score",color="Modelo",text="Score",
            title="Pontuação Final",color_discrete_sequence=["#1565c0","tomato"])
        fig_sc.update_traces(textposition="outside")
        fig_sc.update_layout(showlegend=False,height=350,
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#fafafa",
            yaxis=dict(gridcolor="#e0e0e0"))
        st.plotly_chart(fig_sc,use_container_width=True)

# ── Rodapé ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""<div style='text-align:center;color:#999;font-size:.78rem;padding:.4rem;'>
    RSM-Kriging Analysis Suite v3.1 · Prof. Dr. Ricardo V. P. Rezende &amp;
    Doutoranda Allana Ribeiro Mendes<br>
    Universidade Estadual de Maringá · Departamento de Engenharia Química · 3DCP Lab
</div>""", unsafe_allow_html=True)
