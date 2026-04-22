import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
import io
import pandas as pd

st.set_page_config(page_title="McCabe-Thiele", page_icon="⚗️", layout="wide")
st.title("⚗️ Diagrama de McCabe-Thiele")
st.markdown("Método gráfico para determinação de estágios teóricos em colunas de destilação.")

# ══════════════════════════════════════════════════════════
#  MODO DE EQUILÍBRIO — sidebar topo
# ══════════════════════════════════════════════════════════

st.sidebar.header("Curva de Equilíbrio")
modo_eq = st.sidebar.radio("Fonte dos dados:", ["Volatilidade relativa (α)", "Arquivo de dados"])

# ── Funções base ──────────────────────────────────────────

def y_eq_alpha(x, alpha):
    return alpha * x / (1.0 + (alpha - 1.0) * x)

def x_eq_alpha(y, alpha):
    return y / (alpha - (alpha - 1.0) * y)

# ══════════════════════════════════════════════════════════
#  BLOCO 1 — curva de equilíbrio
# ══════════════════════════════════════════════════════════

azeotropo   = None   # x do azeótropo, se detectado
interp_func = None   # função y_eq(x) unificada
x_eq_inv    = None   # função x_eq(y) unificada

if modo_eq == "Volatilidade relativa (α)":
    alpha = st.sidebar.slider("α — volatilidade relativa", 1.1, 6.0, 2.5, 0.05)
    x_arr = np.linspace(0, 1, 500)
    y_arr = y_eq_alpha(x_arr, alpha)

    # wrappers
    def interp_func(x):
        x = np.asarray(x)
        return y_eq_alpha(x, alpha)

    def x_eq_inv(y):
        y = np.asarray(y, dtype=float)
        return x_eq_alpha(y, alpha)

    eq_label = f"Equilíbrio  (α = {alpha:.2f})"

else:
    st.sidebar.markdown("**Upload do arquivo de equilíbrio**")
    st.sidebar.markdown("Formato: duas colunas `x` e `y`, separadas por vírgula, ponto-e-vírgula ou espaço.")
    arquivo = st.sidebar.file_uploader("Arquivo (.csv, .dat, .txt, .xls, .xlsx)",
                                        type=["csv","dat","txt","xls","xlsx"])

    if arquivo is None:
        st.info("👈 Faça o upload de um arquivo de equilíbrio na barra lateral para continuar.")
        st.stop()

    # leitura
    nome = arquivo.name.lower()
    try:
        if nome.endswith((".xls", ".xlsx")):
            df = pd.read_excel(arquivo, header=None)
        else:
            raw = arquivo.read().decode("utf-8", errors="replace")
            # detecta separador
            sep = "," if "," in raw else (";" if ";" in raw else r"\s+")
            df = pd.read_csv(io.StringIO(raw), sep=sep, header=None, engine="python")

        df = df.dropna().astype(float)
        if df.shape[1] < 2:
            st.error("O arquivo precisa ter pelo menos duas colunas: x e y.")
            st.stop()

        x_data = df.iloc[:, 0].values
        y_data = df.iloc[:, 1].values

        # ordena por x crescente
        ordem = np.argsort(x_data)
        x_data = x_data[ordem]
        y_data = y_data[ordem]

        # garante pontos extremos (0,0) e (1,1) se não existirem
        if x_data[0] > 0.01:
            x_data = np.concatenate([[0.0], x_data])
            y_data = np.concatenate([[0.0], y_data])
        if x_data[-1] < 0.99:
            x_data = np.concatenate([x_data, [1.0]])
            y_data = np.concatenate([y_data, [1.0]])

        n_pts = len(x_data)
        st.sidebar.success(f"{n_pts} pontos carregados.")

    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        st.stop()

    # interpolação PCHIP (monotônica, não oscila)
    pchip = PchipInterpolator(x_data, y_data)

    def interp_func(x):
        x = np.asarray(x, dtype=float)
        return np.clip(pchip(x), 0.0, 1.0)

    # inversa numérica: dado y, encontra x na curva de equilíbrio
    x_dense = np.linspace(0, 1, 2000)
    y_dense = interp_func(x_dense)

    def x_eq_inv(y_target):
        y_target = float(y_target)
        # busca intervalo onde y_dense - y_target muda de sinal
        diff = y_dense - y_target
        idx  = np.where(np.diff(np.sign(diff)))[0]
        if len(idx) == 0:
            return float(np.clip(np.interp(y_target, y_dense, x_dense), 0.0, 1.0))
        # pega o primeiro cruzamento (componente mais volátil)
        i = idx[-1]   # último cruzamento — lado correto da curva
        try:
            return brentq(lambda x: float(interp_func(x)) - y_target,
                          x_dense[i], x_dense[i+1])
        except Exception:
            return float(np.interp(y_target, y_dense, x_dense))

    # detecção de azeótropo: curva cruza a diagonal y=x entre 0 e 1
    diff_diag = y_dense - x_dense
    cross_idx  = np.where((np.diff(np.sign(diff_diag)) != 0) &
                          (x_dense[:-1] > 0.01) &
                          (x_dense[:-1] < 0.99))[0]
    if len(cross_idx) > 0:
        i = cross_idx[0]
        try:
            azeotropo = brentq(lambda x: float(interp_func(x)) - x,
                               x_dense[i], x_dense[i+1])
        except Exception:
            azeotropo = x_dense[i]

    eq_label = f"Equilíbrio  ({n_pts} pontos)"
    alpha = None   # não usado no modo dados

# ══════════════════════════════════════════════════════════
#  BLOCO 2 — parâmetros operacionais
# ══════════════════════════════════════════════════════════

st.sidebar.markdown("---")
st.sidebar.header("Parâmetros Operacionais")

xD = st.sidebar.slider("xD — destilado",    0.50, 0.99, 0.90, 0.01)
xB = st.sidebar.slider("xB — fundo",        0.01, 0.40, 0.05, 0.01)
zF = st.sidebar.slider("zF — alimentação",  0.10, 0.90, 0.45, 0.01)
q  = st.sidebar.slider("q — qualidade da alimentação", -2.0, 2.0, 1.0, 0.02)

st.sidebar.markdown("""
**Parâmetro q:**
- `> 1` → líquido sub-resfriado
- `1.0` → líquido saturado
- `0–1` → misto líq./vap.
- `0.0` → vapor saturado
- `< 0` → vapor superaquecido
""")

# ══════════════════════════════════════════════════════════
#  BLOCO 3 — Rmin e slider R/Rmin
# ══════════════════════════════════════════════════════════

def intersect_q_rect(R, xD, zF, q):
    mR = R / (R + 1.0)
    bR = xD / (R + 1.0)
    if abs(q - 1.0) < 1e-6:
        return zF, mR * zF + bR
    mQ = q / (q - 1.0)
    bQ = -zF / (q - 1.0)
    denom = mR - mQ
    if abs(denom) < 1e-12:
        return zF, mR * zF + bR
    xi = (bQ - bR) / denom
    return xi, mR * xi + bR

def pinch_point_numeric(zF, q):
    """
    Pinch = interseção da linha q com a curva de equilíbrio.
    Busca numérica — funciona para qualquer curva (alpha ou dados).
    """
    if abs(q - 1.0) < 1e-6:
        return zF, float(interp_func(zF))

    if abs(q) < 1e-6:
        # linha q horizontal: y = zF
        return float(x_eq_inv(zF)), zF

    mQ = q / (q - 1.0)
    bQ = -zF / (q - 1.0)

    x_s = np.linspace(0.001, 0.999, 5000)
    y_eq_s = interp_func(x_s)
    y_q_s  = mQ * x_s + bQ
    diff   = y_eq_s - y_q_s

    cross = np.where(np.diff(np.sign(diff)))[0]
    if len(cross) == 0:
        return zF, float(interp_func(zF))

    # escolhe o cruzamento mais próximo de zF
    best_i = cross[np.argmin(np.abs(x_s[cross] - zF))]
    try:
        xp = brentq(lambda x: float(interp_func(x)) - (mQ*x + bQ),
                    x_s[best_i], x_s[best_i+1])
    except Exception:
        xp = x_s[best_i]

    return float(np.clip(xp, 0.0, 1.0)), float(interp_func(xp))

def calc_rmin(xD, zF, q):
    xp, yp = pinch_point_numeric(zF, q)
    if abs(yp - xp) < 1e-12:
        return float('inf')
    val = (xD - yp) / (yp - xp)
    return max(0.0, val)

# calcula Rmin antes do slider
Rmin = calc_rmin(xD, zF, q)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**R_min calculado: `{Rmin:.3f}`**")

if Rmin == float('inf') or Rmin > 50:
    st.sidebar.warning("R_min muito alto ou indefinido. Verifique os parâmetros.")
    razao_R = st.sidebar.slider("R / R_min", 0.90, 5.0, 1.5, 0.05, disabled=True)
    R = 2.0
else:
    razao_R = st.sidebar.slider("R / R_min", 0.90, 5.0, 1.5, 0.05)
    R = razao_R * Rmin

st.sidebar.markdown(f"**R resultante: `{R:.3f}`**")

# ══════════════════════════════════════════════════════════
#  BLOCO 4 — estágios
# ══════════════════════════════════════════════════════════

def build_stages(R, xD, xB, zF, q):
    mR = R / (R + 1.0)
    bR = xD / (R + 1.0)
    xi, yi = intersect_q_rect(R, xD, zF, q)
    xi = float(np.clip(xi, xB, xD))
    yi = float(np.clip(yi, xB, xD))

    if abs(xi - xB) < 1e-10:
        mS, bS = 1e6, 0.0
    else:
        mS = (yi - xB) / (xi - xB)
        bS = xB - mS * xB

    def op_line(x, use_rect):
        return mR * x + bR if use_rect else mS * x + bS

    pts = [(xD, xD)]
    use_rect   = True
    feed_stage = -1
    stage      = 0
    y_cur      = xD

    while stage < 100:
        x_star = float(np.clip(x_eq_inv(y_cur), 0.0, 1.0))
        pts.append((x_star, y_cur))
        stage += 1
        if x_star <= xB + 1e-5:
            break
        if use_rect and x_star <= xi + 1e-4:
            use_rect = False
            if feed_stage < 0:
                feed_stage = stage
        y_op = op_line(x_star, use_rect)
        pts.append((x_star, y_op))
        y_cur = y_op
        if y_cur <= xB + 1e-5:
            break

    if feed_stage < 0:
        feed_stage = stage
    return pts, stage, feed_stage

# ══════════════════════════════════════════════════════════
#  BLOCO 5 — cálculos finais e avisos
# ══════════════════════════════════════════════════════════

xi, yi = intersect_q_rect(R, xD, zF, q)
xi = float(np.clip(xi, xB, xD))
yi = float(np.clip(yi, 0.0, 1.0))
xp, yp = pinch_point_numeric(zF, q)
pts, n_stages, feed_stage = build_stages(R, xD, xB, zF, q)

# avisos
if azeotropo is not None:
    st.warning(f"⚠️ Azeótropo detectado em x ≈ {azeotropo:.4f}. "
               f"xD e xB devem estar do mesmo lado do azeótropo. "
               f"A escada travará se tentar cruzá-lo.")

if xD < xB:
    st.error("xD deve ser maior que xB.")

# métricas
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Estágios teóricos",   n_stages)
col2.metric("Estágio alimentação", feed_stage)
col3.metric("R_min",  f"{Rmin:.3f}" if Rmin != float('inf') else "∞")
col4.metric("R / R_min", f"{razao_R:.2f}" if Rmin != float('inf') else "—")
col5.metric("R",      f"{R:.3f}"  if Rmin != float('inf') else "—")

# ══════════════════════════════════════════════════════════
#  BLOCO 6 — plot
# ══════════════════════════════════════════════════════════

_, col_plot, _ = st.columns([1, 2, 1])

with col_plot:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlabel("x  (fase líquida)", fontsize=11)
    ax.set_ylabel("y  (fase vapor)",   fontsize=11)
    ax.set_title("Diagrama de McCabe-Thiele", fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, linestyle="--")

    x_plot = np.linspace(0, 1, 500)

    # Diagonal
    ax.plot(x_plot, x_plot, color="gray", lw=1, linestyle="--",
            label="Diagonal  y = x", zorder=1)

    # Curva de equilíbrio
    ax.plot(x_plot, interp_func(x_plot), color="#3a8fd1", lw=2.5,
            label=eq_label, zorder=2)

    # Azeótropo
    if azeotropo is not None:
        ax.axvline(azeotropo, color="#e05c2a", lw=1, linestyle=":",
                   alpha=0.6, label=f"Azeótropo  x={azeotropo:.3f}")
        ax.plot(azeotropo, azeotropo, "^", color="#e05c2a", ms=8, zorder=6)

    # Linha de retificação
    ax.plot([xi, xD], [yi, xD], color="#e05c2a", lw=2,
            label=f"Retificação  (R = {R:.2f})", zorder=3)

    # Linha de esgotamento
    ax.plot([xB, xi], [xB, yi], color="#1a9e6e", lw=2,
            label="Esgotamento", zorder=3)

    # Linha q: de (zF,zF) até (xi,yi) + 20% extrapolação
    if abs(q - 1.0) < 1e-6:
        y_extra = float(np.clip(yi + 0.20*(yi - zF), 0.0, 1.0))
        ax.plot([zF, zF], [zF, y_extra], color="#c04ab0", lw=1.8,
                linestyle="-.", label=f"Linha q  (q = {q:.2f})", zorder=2)
    else:
        dx = xi - zF
        dy = yi - zF
        x_extra = float(np.clip(xi + 0.20*dx, 0.0, 1.0))
        y_extra = float(np.clip(yi + 0.20*dy, 0.0, 1.0))
        ax.plot([zF, x_extra], [zF, y_extra], color="#c04ab0", lw=1.8,
                linestyle="-.", label=f"Linha q  (q = {q:.2f})", zorder=2)

    # Ponto de interseção das linhas de operação
    ax.plot(xi, yi, "o", color="#555", ms=6, zorder=5)

    # Estágios
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color="#f0b429", lw=2.5,
            label=f"Estágios ({n_stages} teóricos)", zorder=4)

    # Verticais tracejadas xB, zF, xD → até diagonal
    for val, cor, lbl in [
            (xB, "#1a9e6e", f"xB={xB:.2f}"),
            (zF, "#c04ab0", f"zF={zF:.2f}"),
            (xD, "#e05c2a", f"xD={xD:.2f}")]:
        ax.plot([val, val], [0.0, val], color=cor, lw=1.0,
                linestyle=":", alpha=0.8, zorder=1)
        ax.text(val, -0.04, lbl, ha="center", va="top",
                color=cor, fontsize=8)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=3, fontsize=8, framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.07, 1.0)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    st.pyplot(fig)

# ══════════════════════════════════════════════════════════
#  Rodapé
# ══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
**Algoritmo da escada:** parte de $(x_D, x_D)$ na diagonal →
horizontal até a curva de equilíbrio → vertical até a linha de operação ativa →
troca para esgotamento ao cruzar o ponto de alimentação → repete até $x^* \\leq x_B$.

**R_min** pelo critério do pinch numérico:
$R_{min} = (x_D - y_p)\\,/\\,(y_p - x_p)$,
onde $(x_p, y_p)$ é a interseção da linha $q$ com a curva de equilíbrio.

**q < 0** corresponde a vapor superaquecido — a linha q tem inclinação positiva menor que 1.
""")
