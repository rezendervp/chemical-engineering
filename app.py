import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ══════════════════════════════════════════════════════════
#  Configuração da página
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="McCabe-Thiele",
    page_icon="⚗️",
    layout="wide",
)

st.title("⚗️ Diagrama de McCabe-Thiele")
st.markdown("Método gráfico para determinação de estágios teóricos em colunas de destilação.")

# ══════════════════════════════════════════════════════════
#  Funções auxiliares
# ══════════════════════════════════════════════════════════

def y_eq(x, alpha):
    """Curva de equilíbrio VLE: y = α·x / [1 + (α−1)·x]"""
    return alpha * x / (1.0 + (alpha - 1.0) * x)

def x_eq(y, alpha):
    """Inversa: x = y / [α − (α−1)·y]"""
    return y / (alpha - (alpha - 1.0) * y)

def intersect_q_rect(R, xD, zF, q):
    """
    Interseção entre linha q e linha de retificação.
    Este ponto é onde a linha de retificação TERMINA
    e a de esgotamento começa.
    """
    mR = R / (R + 1.0)
    bR = xD / (R + 1.0)
    if abs(q - 1.0) < 1e-6:
        xi = zF
        yi = mR * zF + bR
    else:
        mQ = q / (q - 1.0)
        bQ = -zF / (q - 1.0)
        xi = (bQ - bR) / (mR - mQ)
        yi = mR * xi + bR
    return xi, yi

def pinch_point(alpha, zF, q):
    """
    Ponto de pinch = interseção da linha q com a curva de equilíbrio.
    Com R = Rmin, as três curvas convergem aqui.

    Resolve a quadrática de igualar y_q = y_eq:
      (alpha-1)*mQ*x^2 + (mQ - (alpha-1)*bQ - 1)*x - bQ = 0
    """
    if abs(q - 1.0) < 1e-6:
        xp = zF
        yp = y_eq(zF, alpha)
        return xp, yp

    mQ = q / (q - 1.0)
    bQ = -zF / (q - 1.0)
    a  = alpha - 1.0
    A  = a * mQ
    B  = mQ - a * bQ - 1.0
    C  = -bQ
    disc = B**2 - 4*A*C

    if disc < 0 or abs(A) < 1e-12:
        xp = zF
    else:
        x1 = (-B + np.sqrt(disc)) / (2*A)
        x2 = (-B - np.sqrt(disc)) / (2*A)
        xp = x1 if 0.0 <= x1 <= 1.0 else x2

    xp = float(np.clip(xp, 0.0, 1.0))
    yp = y_eq(xp, alpha)
    return xp, yp

def calc_rmin(alpha, xD, zF, q):
    """
    R_min pelo critério do pinch:
        Rmin = (xD - yp) / (yp - xp)
    onde (xp, yp) é a interseção da linha q com a curva de equilíbrio.
    """
    xp, yp = pinch_point(alpha, zF, q)
    if abs(yp - xp) < 1e-12:
        return float('inf')
    return max(0.0, (xD - yp) / (yp - xp))

def build_stages(alpha, R, xD, xB, zF, q):
    """
    Escada de McCabe-Thiele.

    1. Parte de (xD, xD) na diagonal.
    2. Horizontal (y fixo) → x* na curva de equilíbrio.
    3. Vertical (x fixo)   → desce até a linha de operação ativa.
    4. Troca para esgotamento quando x* <= xi.
    5. Repete até x* <= xB.
    """
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
    stage = 0
    y_cur = xD

    while stage < 100:
        x_star = float(np.clip(x_eq(y_cur, alpha), 0.0, 1.0))
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
#  Sidebar — sliders
# ══════════════════════════════════════════════════════════

st.sidebar.header("Parâmetros")

alpha = st.sidebar.slider("α — volatilidade relativa", 1.1, 6.0, 2.5, 0.05)
R     = st.sidebar.slider("R — razão de refluxo",      0.5, 10.0, 2.0, 0.05)
xD    = st.sidebar.slider("xD — destilado",            0.50, 0.99, 0.90, 0.01)
xB    = st.sidebar.slider("xB — fundo",                0.01, 0.40, 0.05, 0.01)
zF    = st.sidebar.slider("zF — alimentação",          0.10, 0.90, 0.45, 0.01)
q     = st.sidebar.slider("q — qualidade da alimentação", 0.0, 2.0, 1.0, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**q:**
- `1.0` líquido saturado
- `0.0` vapor saturado
- `> 1` sub-resfriado
- `0–1` misto líq./vap.
""")

# ══════════════════════════════════════════════════════════
#  Cálculos
# ══════════════════════════════════════════════════════════

Rmin  = calc_rmin(alpha, xD, zF, q)
ratio = R / Rmin if Rmin > 0 else float('inf')
xp, yp = pinch_point(alpha, zF, q)
xi, yi = intersect_q_rect(R, xD, zF, q)
xi = float(np.clip(xi, xB, xD))
yi = float(np.clip(yi, 0.0, 1.0))
pts, n_stages, feed_stage = build_stages(alpha, R, xD, xB, zF, q)

# ── Métricas no topo ────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Estágios teóricos",    n_stages)
col2.metric("Estágio alimentação",  feed_stage)
col3.metric("R_min",                f"{Rmin:.3f}")
col4.metric("R / R_min",            f"{ratio:.2f}")

# ══════════════════════════════════════════════════════════
#  Plot
# ══════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(7, 6))
ax.set_xlabel("x  (fração molar — fase líquida)", fontsize=11)
ax.set_ylabel("y  (fração molar — fase vapor)",   fontsize=11)
ax.set_title("Diagrama de McCabe-Thiele", fontsize=13, fontweight="bold")
ax.set_aspect("equal")
ax.grid(True, alpha=0.2, linestyle="--")

x_arr = np.linspace(0, 1, 500)

# Diagonal
ax.plot(x_arr, x_arr, color="gray", lw=1, linestyle="--",
        label="Diagonal  y = x", zorder=1)

# Curva de equilíbrio
ax.plot(x_arr, y_eq(x_arr, alpha), color="#3a8fd1", lw=2.5,
        label=f"Equilíbrio  (α = {alpha:.2f})", zorder=2)

# Linha de retificação: de (xi,yi) até (xD,xD)
ax.plot([xi, xD], [yi, xD], color="#e05c2a", lw=2,
        label=f"Retificação  (R = {R:.2f})", zorder=3)

# Linha de esgotamento: de (xB,xB) até (xi,yi)
ax.plot([xB, xi], [xB, yi], color="#1a9e6e", lw=2,
        label="Esgotamento", zorder=3)

# Linha q
if abs(q - 1.0) < 1e-6:
    ax.plot([zF, zF], [0, yp], color="#c04ab0",
            lw=1.5, linestyle="-.", label=f"Alimentação  q = {q:.2f}", zorder=2)
else:
    mQ = q / (q - 1.0)
    bQ = -zF / (q - 1.0)
    x_q0 = float(np.clip(-bQ / mQ if abs(mQ) > 1e-10 else zF, 0.0, 1.0))
    ax.plot([x_q0, xp], [mQ*x_q0+bQ, yp], color="#c04ab0",
            lw=1.5, linestyle="-.", label=f"Alimentação  q = {q:.2f}", zorder=2)

# Ponto de pinch
ax.plot(xp, yp, "o", color="#c04ab0", ms=7, zorder=5,
        label=f"Pinch  ({xp:.3f}, {yp:.3f})")

# Estágios
xs = [p[0] for p in pts]
ys = [p[1] for p in pts]
ax.plot(xs, ys, color="#f0b429", lw=2.5,
        label=f"Estágios ({n_stages} teóricos)", zorder=4)

# Verticais tracejadas em xB, zF, xD até a diagonal
for val, cor, lbl in [
        (xB, "#1a9e6e", f"$x_B$={xB:.2f}"),
        (zF, "#c04ab0", f"$z_F$={zF:.2f}"),
        (xD, "#e05c2a", f"$x_D$={xD:.2f}")]:
    ax.plot([val, val], [0, val], color=cor, lw=1.0,
            linestyle=":", alpha=0.8, zorder=1)
    ax.text(val, -0.04, lbl, ha="center", va="top",
            color=cor, fontsize=9)

ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
ax.set_xlim(0, 1)
ax.set_ylim(-0.06, 1.0)
plt.tight_layout()

st.pyplot(fig)

# ══════════════════════════════════════════════════════════
#  Nota de rodapé
# ══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
**Algoritmo da escada:** parte de $(x_D, x_D)$ na diagonal →
horizontal até a curva de equilíbrio → vertical até a linha de operação ativa →
troca para esgotamento quando cruza o ponto de alimentação → repete até $x^* \\leq x_B$.

**R_min** é calculado pelo critério do pinch:
$R_{min} = (x_D - y_p)\\,/\\,(y_p - x_p)$,
onde $(x_p, y_p)$ é a interseção da linha $q$ com a curva de equilíbrio.
""")
