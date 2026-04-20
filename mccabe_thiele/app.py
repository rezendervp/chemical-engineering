import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="McCabe-Thiele", page_icon="⚗️", layout="wide")
st.title("⚗️ Diagrama de McCabe-Thiele")
st.markdown("Método gráfico para determinação de estágios teóricos em colunas de destilação.")

# ══════════════════════════════════════════════════════════
#  Funções auxiliares
# ══════════════════════════════════════════════════════════

def y_eq(x, alpha):
    return alpha * x / (1.0 + (alpha - 1.0) * x)

def x_eq(y, alpha):
    return y / (alpha - (alpha - 1.0) * y)

def intersect_q_rect(R, xD, zF, q):
    """
    Interseção linha q ∩ linha de retificação.
    Este ponto (xi, yi) é onde a retificação termina e o esgotamento começa.
    A linha q vai de (zF, zF) até (xi, yi) — podendo ser extrapolada além.
    Prova que a linha q passa por (zF, zF):
      y_q(zF) = [q/(q-1)]*zF - zF/(q-1) = zF*(q-1)/(q-1) = zF  ✓
    """
    mR = R / (R + 1.0)
    bR = xD / (R + 1.0)
    if abs(q - 1.0) < 1e-6:      # líquido saturado: linha q vertical x=zF
        xi = zF
        yi = mR * zF + bR
    else:
        mQ = q / (q - 1.0)
        bQ = -zF / (q - 1.0)
        xi = (bQ - bR) / (mR - mQ)
        yi = mR * xi + bR
    return xi, yi

def pinch_point(alpha, zF, q):
    """Interseção da linha q com a curva de equilíbrio — usado só para Rmin."""
    if abs(q - 1.0) < 1e-6:
        return zF, y_eq(zF, alpha)
    if abs(q) < 1e-6:
        xp = float(np.clip(x_eq(zF, alpha), 0.0, 1.0))
        return xp, zF
    mQ = q / (q - 1.0)
    bQ = -zF / (q - 1.0)
    a = alpha - 1.0
    A = a * mQ
    B = mQ - a * bQ - 1.0
    C = -bQ
    disc = B**2 - 4*A*C
    if disc < 0:
        return zF, y_eq(zF, alpha)
    if abs(A) < 1e-12:
        xp = -C / B if abs(B) > 1e-12 else zF
    else:
        x1 = (-B + np.sqrt(disc)) / (2*A)
        x2 = (-B - np.sqrt(disc)) / (2*A)
        best, best_res = zF, 1e10
        for xc in [x1, x2]:
            if 0.0 <= xc <= 1.0:
                res = abs(y_eq(xc, alpha) - (mQ*xc + bQ))
                if res < best_res:
                    best_res = res
                    best = xc
        xp = best
    return float(np.clip(xp, 0.0, 1.0)), y_eq(float(np.clip(xp, 0.0, 1.0)), alpha)

def calc_rmin(alpha, xD, zF, q):
    xp, yp = pinch_point(alpha, zF, q)
    if abs(yp - xp) < 1e-12:
        return float('inf')
    return max(0.0, (xD - yp) / (yp - xp))

def build_stages(alpha, R, xD, xB, zF, q):
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
    use_rect = True
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
#  Sidebar
# ══════════════════════════════════════════════════════════

st.sidebar.header("Parâmetros")
alpha = st.sidebar.slider("α — volatilidade relativa",    1.1,  6.0,  2.5,  0.05)
R     = st.sidebar.slider("R — razão de refluxo",         0.5,  10.0, 2.0,  0.05)
xD    = st.sidebar.slider("xD — destilado",               0.50, 0.99, 0.90, 0.01)
xB    = st.sidebar.slider("xB — fundo",                   0.01, 0.40, 0.05, 0.01)
zF    = st.sidebar.slider("zF — alimentação",             0.10, 0.90, 0.45, 0.01)
q     = st.sidebar.slider("q — qualidade da alimentação", 0.0,  2.0,  1.0,  0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Parâmetro q:**
- `1.0` → líquido saturado
- `0.0` → vapor saturado
- `> 1` → sub-resfriado
- `0–1` → misto líq./vap.
""")

# ══════════════════════════════════════════════════════════
#  Cálculos
# ══════════════════════════════════════════════════════════

Rmin  = calc_rmin(alpha, xD, zF, q)
ratio = R / Rmin if Rmin not in (0, float('inf')) else float('inf')
xi, yi = intersect_q_rect(R, xD, zF, q)
xi = float(np.clip(xi, xB, xD))
yi = float(np.clip(yi, 0.0, 1.0))
pts, n_stages, feed_stage = build_stages(alpha, R, xD, xB, zF, q)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Estágios teóricos",   n_stages)
col2.metric("Estágio alimentação", feed_stage)
col3.metric("R_min",               f"{Rmin:.3f}" if Rmin != float('inf') else "∞")
col4.metric("R / R_min",           f"{ratio:.2f}" if ratio != float('inf') else "∞")

# ══════════════════════════════════════════════════════════
#  Plot
# ══════════════════════════════════════════════════════════

_, col_plot, _ = st.columns([1, 2, 1])

with col_plot:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlabel("x  (fase líquida)", fontsize=11)
    ax.set_ylabel("y  (fase vapor)",   fontsize=11)
    ax.set_title("Diagrama de McCabe-Thiele", fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, linestyle="--")

    x_arr = np.linspace(0, 1, 500)

    # ── Diagonal y = x ──────────────────────────────────────
    ax.plot(x_arr, x_arr, color="gray", lw=1, linestyle="--",
            label="Diagonal  y = x", zorder=1)

    # ── Curva de equilíbrio ─────────────────────────────────
    ax.plot(x_arr, y_eq(x_arr, alpha), color="#3a8fd1", lw=2.5,
            label=f"Equilíbrio  (α = {alpha:.2f})", zorder=2)

    # ── Linha de retificação: (xi,yi) → (xD,xD) ────────────
    ax.plot([xi, xD], [yi, xD], color="#e05c2a", lw=2,
            label=f"Retificação  (R = {R:.2f})", zorder=3)

    # ── Linha de esgotamento: (xB,xB) → (xi,yi) ───────────
    ax.plot([xB, xi], [xB, yi], color="#1a9e6e", lw=2,
            label="Esgotamento", zorder=3)

    # ── Linha q: parte de (zF,zF) na diagonal,
    #    termina em (xi,yi) extrapolada 20% além para didática ──
    if abs(q - 1.0) < 1e-6:
        # líquido saturado: vertical de (zF,zF) até (zF, yi) + 20% extra
        y_extra = yi + 0.20 * (yi - zF)
        y_extra = float(np.clip(y_extra, 0.0, 1.0))
        ax.plot([zF, zF], [zF, y_extra], color="#c04ab0", lw=1.8,
                linestyle="-.", label=f"Linha q  (q = {q:.2f})", zorder=2)
    else:
        # caso geral: de (zF,zF) até (xi,yi) + 20% de extrapolação
        dx = xi - zF
        dy = yi - zF
        x_extra = xi + 0.20 * dx
        y_extra = yi + 0.20 * dy
        x_extra = float(np.clip(x_extra, 0.0, 1.0))
        y_extra = float(np.clip(y_extra, 0.0, 1.0))
        ax.plot([zF, x_extra], [zF, y_extra], color="#c04ab0", lw=1.8,
                linestyle="-.", label=f"Linha q  (q = {q:.2f})", zorder=2)

    # ── Ponto de interseção das linhas de operação ──────────
    ax.plot(xi, yi, "o", color="#888", ms=6, zorder=5)

    # ── Estágios (escada) ────────────────────────────────────
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color="#f0b429", lw=2.5,
            label=f"Estágios ({n_stages} teóricos)", zorder=4)

    # ── Verticais tracejadas: xB, zF, xD → de y=0 até diagonal ─
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

**R_min** pelo critério do pinch:
$R_{min} = (x_D - y_p)\\,/\\,(y_p - x_p)$,
onde $(x_p,\\,y_p)$ é a interseção da linha $q$ com a curva de equilíbrio.
""")
