"""
solver.py
---------
Núcleo numérico do simulador de condução de calor 2D transiente/permanente.

Equação governante (material homogêneo, k constante no domínio):

    rho*cp * dT/dt = k * (d2T/dx2 + d2T/dy2)   =>   dT/dt = alpha * grad2(T)

Discretização espacial: diferenças finitas centradas de 2a ordem, malha
cartesiana uniforme por direção (dx, dy podem diferir entre si).

Condições de contorno: incorporadas via nó fantasma (ver bc.py), o que
preserva a ordem 2 de precisão também nas fronteiras com Neumann/convecção.

Convenção geral usada neste módulo:

    Seja A a matriz esparsa (N x N) e F o vetor (N,) tais que, para todo nó
    que NÃO é Dirichlet:
        (A @ T)[p] + F[p]  ~  grad2(T) no nó p

    Nos nós Dirichlet, a linha de A é nula e F é nulo (tratados à parte,
    pois não entram na PDE -- o valor é imposto diretamente).

    - Regime permanente:      A @ T = -F        (nos nós não-Dirichlet)
    - Regime transiente:      dT/dt = alpha*(A @ T + F)

Os solvers iterativos (Jacobi, Gauss-Seidel, SOR) e o direto operam sobre um
sistema linear genérico M @ T = b -- a mesma rotina serve tanto para o passo
permanente quanto para cada passo implícito do transiente.

NOVO nesta versão: todos os solvers iterativos aceitam um parâmetro opcional
`callback(it, residuo)`, chamado a cada iteração (ou a cada `callback_step`
iterações). Isso permite que a interface gráfica desenhe o gráfico de
convergência EM TEMPO REAL durante a solução, sem precisar esperar o fim.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .bc import BC, ContornosRetangulo
from .mesh import Mesh2D


# ---------------------------------------------------------------------------
# Montagem do operador de Laplace com condições de contorno (nó fantasma)
# ---------------------------------------------------------------------------

def _ghost_coef(bc: BC, k: float, h_espac: float):
    """
    Para uma face com espaçamento normal h_espac (dx ou dy) e condutividade k,
    retorna (diag_extra, const) -- a contribuição extra na diagonal principal
    (multiplicando T da própria fronteira) e o termo constante (para F),
    resultantes da eliminação algébrica do nó fantasma.

    Dedução (face esquerda, normal externa = -x; análoga para as demais):
        Lei de Fourier na normal externa:  -k * dT/dn = q   =>   dT/dn = -q/k
        Nó fantasma:      T_fantasma = T_espelho - 2*h*q/k
        Convecção:        q = h_conv*(T_fronteira - Tinf)  (linear em T_fronteira)
    """
    if bc.tipo == "neumann":
        return 0.0, -2.0 * bc.q / (k * h_espac)
    elif bc.tipo == "conveccao":
        diag = -2.0 * bc.h / (k * h_espac)
        const = 2.0 * bc.h * bc.Tinf / (k * h_espac)
        return diag, const
    elif bc.tipo == "simetria":
        return 0.0, 0.0
    else:
        raise ValueError(f"_ghost_coef não trata o tipo '{bc.tipo}' (deveria ser Dirichlet à parte).")


def assemble_laplaciano(mesh: Mesh2D, contornos: ContornosRetangulo, k: float):
    """
    Monta o operador discreto do Laplaciano via diferenças finitas centradas
    de 2a ordem, com condições de contorno pelo método do nó fantasma.

    Em cada nó, a contribuição em x e a contribuição em y são somadas de forma
    INDEPENDENTE -- por isso os cantos (onde duas faces distintas se encontram)
    são tratados automaticamente sem lógica especial: o Laplaciano é separável,
    grad2(T) = d2T/dx2 + d2T/dy2.

    Retorna:
        A               : matriz esparsa CSR (N x N)
        F               : vetor (N,) parte constante (fluxo/convecção)
        dirichlet_mask  : array booleano (N,), True onde T é prescrita
        dirichlet_vals  : array (N,), valor prescrito onde a máscara é True
    """
    Nx, Ny, dx, dy = mesh.Nx, mesh.Ny, mesh.dx, mesh.dy
    N = mesh.n_nos
    rows, cols, data = [], [], []
    F = np.zeros(N)
    dirichlet_mask = np.zeros(N, dtype=bool)
    dirichlet_vals = np.zeros(N)

    def add(p, q_, val):
        rows.append(p); cols.append(q_); data.append(val)

    avisos_canto = []

    for j in range(Ny):
        for i in range(Nx):
            p = mesh.idx(i, j)

            candidatos = []
            if i == 0 and contornos.esquerda.tipo == "dirichlet":
                candidatos.append(contornos.esquerda.T)
            if i == Nx - 1 and contornos.direita.tipo == "dirichlet":
                candidatos.append(contornos.direita.T)
            if j == 0 and contornos.inferior.tipo == "dirichlet":
                candidatos.append(contornos.inferior.T)
            if j == Ny - 1 and contornos.superior.tipo == "dirichlet":
                candidatos.append(contornos.superior.T)

            if candidatos:
                if len(set(candidatos)) > 1:
                    avisos_canto.append((i, j, candidatos))
                dirichlet_mask[p] = True
                dirichlet_vals[p] = candidatos[0]
                continue

            # contribuição em x
            if i == 0:
                bc = contornos.esquerda
                add(p, mesh.idx(1, j), 2.0 / dx**2)
                diag, const = _ghost_coef(bc, k, dx)
                add(p, p, -2.0 / dx**2 + diag)
                F[p] += const
            elif i == Nx - 1:
                bc = contornos.direita
                add(p, mesh.idx(Nx - 2, j), 2.0 / dx**2)
                diag, const = _ghost_coef(bc, k, dx)
                add(p, p, -2.0 / dx**2 + diag)
                F[p] += const
            else:
                add(p, mesh.idx(i - 1, j), 1.0 / dx**2)
                add(p, mesh.idx(i + 1, j), 1.0 / dx**2)
                add(p, p, -2.0 / dx**2)

            # contribuição em y
            if j == 0:
                bc = contornos.inferior
                add(p, mesh.idx(i, 1), 2.0 / dy**2)
                diag, const = _ghost_coef(bc, k, dy)
                add(p, p, -2.0 / dy**2 + diag)
                F[p] += const
            elif j == Ny - 1:
                bc = contornos.superior
                add(p, mesh.idx(i, Ny - 2), 2.0 / dy**2)
                diag, const = _ghost_coef(bc, k, dy)
                add(p, p, -2.0 / dy**2 + diag)
                F[p] += const
            else:
                add(p, mesh.idx(i, j - 1), 1.0 / dy**2)
                add(p, mesh.idx(i, j + 1), 1.0 / dy**2)
                add(p, p, -2.0 / dy**2)

    if avisos_canto:
        for (i, j, vals) in avisos_canto:
            print(f"[AVISO] Nó de canto (i={i}, j={j}) tem BCs Dirichlet conflitantes {vals}; "
                  f"usando prioridade esquerda>direita>inferior>superior => T={vals[0]:g}")

    A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    return A, F, dirichlet_mask, dirichlet_vals


# ---------------------------------------------------------------------------
# Estabilidade do esquema explícito (regime transiente)
# ---------------------------------------------------------------------------

def dt_critico(alpha: float, dx: float, dy: float) -> float:
    """
    Passo de tempo crítico (limite de estabilidade) do esquema explícito FTCS
    para a equação de difusão 2D, obtido por análise de von Neumann:

        dt_crit = 1 / [ 2*alpha*(1/dx^2 + 1/dy^2) ]

    Válido para malha uniforme por direção; usado como referência prática
    mesmo com condições de contorno mistas. Recomenda-se margem de segurança,
    ex.: dt = 0.9 * dt_crit.
    """
    return 1.0 / (2.0 * alpha * (1.0 / dx**2 + 1.0 / dy**2))


# ---------------------------------------------------------------------------
# Sistemas lineares: regime permanente e passo implícito
# ---------------------------------------------------------------------------

def build_steady_system(A, F, dirichlet_mask, dirichlet_vals):
    """Monta M @ T = b para o regime permanente: A@T = -F, com linhas Dirichlet = identidade."""
    M = A.tolil()
    b = -F.copy()
    idx_dir = np.where(dirichlet_mask)[0]
    for p in idx_dir:
        M.rows[p] = [p]
        M.data[p] = [1.0]
        b[p] = dirichlet_vals[p]
    return M.tocsr(), b


def build_implicit_system(A, F, dirichlet_mask, dirichlet_vals, T_n, dt, alpha):
    """Monta M @ T_new = b para um passo de Euler implícito (backward Euler).

    ATENÇÃO -- uso recomendado apenas para UM passo isolado. Para uma série
    de passos de tempo (o caso comum), M não muda de um passo para o outro
    (só depende de A, dt, alpha -- todos constantes); rebuildar a matriz
    inteira a cada passo é desperdício e pode travar a interface em
    simulações longas. Para séries de passos, use
    `build_implicit_matrix` (uma vez) + `build_implicit_rhs` (a cada passo).
    """
    N = A.shape[0]
    I = sp.identity(N, format="csr")
    M = (I - dt * alpha * A).tolil()
    b = T_n + dt * alpha * F
    idx_dir = np.where(dirichlet_mask)[0]
    for p in idx_dir:
        M.rows[p] = [p]
        M.data[p] = [1.0]
        b[p] = dirichlet_vals[p]
    return M.tocsr(), b


def build_implicit_matrix(A, dirichlet_mask, dt, alpha):
    """
    Monta APENAS a matriz M = I - dt*alpha*A do esquema implícito, com as
    linhas Dirichlet substituídas por identidade -- construída UMA VEZ e
    reaproveitada em todos os passos de tempo (M não depende de T_n).

    NOTA CONCEITUAL (formulação única, ver residuo_pde): fazendo dt -> +inf
    nesta mesma equação, (I - dt*alpha*A)@T = T_ant + dt*alpha*F se reduz a
    A@T = -F, exatamente o sistema do regime permanente. Ou seja, regime
    permanente e transiente compartilham a MESMA formulação algébrica;
    a diferença é puramente de interpretação de dt (estabilização numérica
    vs. tempo físico) -- não de método.
    """
    N = A.shape[0]
    I = sp.identity(N, format="csr")
    M = (I - dt * alpha * A).tolil()
    idx_dir = np.where(dirichlet_mask)[0]
    for p in idx_dir:
        M.rows[p] = [p]
        M.data[p] = [1.0]
    return M.tocsr()


def build_implicit_rhs(F, dirichlet_mask, dirichlet_vals, T_n, dt, alpha):
    """
    Monta APENAS o vetor b = T_n + dt*alpha*F do esquema implícito (com os
    nós Dirichlet sobrescritos), para ser usado junto de uma matriz M já
    construída por `build_implicit_matrix`. Operação O(N), rápida --
    chamada a cada passo de tempo sem custo relevante.
    """
    b = T_n + dt * alpha * F
    b[dirichlet_mask] = dirichlet_vals[dirichlet_mask]
    return b


def fatorar_direto(M):
    """
    Fatoração LU esparsa reutilizável (scipy.sparse.linalg.splu). Para o
    esquema implícito com solver 'direto', M é CONSTANTE ao longo de todos
    os passos de tempo -- fatorar uma vez e reutilizar (apenas substituições
    triangulares, O(nnz), a cada passo) é ordens de magnitude mais rápido
    que refazer a fatoração completa a cada passo.
    """
    return spla.splu(M.tocsc())


def resolver_direto_fatorado(lu, b):
    """Resolve M @ T = b usando uma fatoração LU já calculada por `fatorar_direto`."""
    return lu.solve(b)


def passo_explicito(A, F, dirichlet_mask, dirichlet_vals, T_n, dt, alpha):
    """Um passo de Euler explícito (forward Euler): T^{n+1} = T^n + dt*alpha*(A@T^n + F)."""
    T_new = T_n + dt * alpha * (A @ T_n + F)
    T_new[dirichlet_mask] = dirichlet_vals[dirichlet_mask]
    return T_new


# ---------------------------------------------------------------------------
# Solvers para M @ T = b (genéricos: servem tanto ao regime permanente
# quanto a cada passo do Euler implícito). Todos aceitam `callback(it, res)`
# opcional, chamado a cada `callback_step` iterações -- usado pela GUI para
# desenhar o gráfico de convergência EM TEMPO REAL.
# ---------------------------------------------------------------------------

def resolver_jacobi(M, b, T0, tol=1e-8, max_iter=20000, callback=None, callback_step=5):
    M = M.tocsr()
    D = M.diagonal()
    R = M - sp.diags(D)
    T = T0.copy()
    hist = []
    nb = np.linalg.norm(b) + 1e-30
    for it in range(max_iter):
        T_new = (b - R @ T) / D
        res = np.linalg.norm(M @ T_new - b) / nb
        hist.append(res)
        T = T_new
        if callback is not None and (it % callback_step == 0 or res < tol):
            callback(it + 1, res, T_new)
        if res < tol:
            return T, np.array(hist), it + 1, True
    return T, np.array(hist), max_iter, False


def resolver_gauss_seidel(M, b, T0, tol=1e-8, max_iter=20000, callback=None, callback_step=5):
    M = M.tocsr()
    L = sp.tril(M, format="csr")   # inclui a diagonal
    U = sp.triu(M, k=1, format="csr")
    T = T0.copy()
    hist = []
    nb = np.linalg.norm(b) + 1e-30
    for it in range(max_iter):
        rhs = b - U @ T
        T_new = spla.spsolve_triangular(L, rhs, lower=True)
        res = np.linalg.norm(M @ T_new - b) / nb
        hist.append(res)
        T = T_new
        if callback is not None and (it % callback_step == 0 or res < tol):
            callback(it + 1, res, T_new)
        if res < tol:
            return T, np.array(hist), it + 1, True
    return T, np.array(hist), max_iter, False


def resolver_sor(M, b, T0, omega=1.7, tol=1e-8, max_iter=20000, callback=None, callback_step=5):
    M = M.tocsr()
    diag = M.diagonal()
    D = sp.diags(diag)
    L = sp.tril(M, k=-1, format="csr")
    U = sp.triu(M, k=1, format="csr")
    A_esq = (D + omega * L).tocsr()
    T = T0.copy()
    hist = []
    nb = np.linalg.norm(b) + 1e-30
    for it in range(max_iter):
        rhs = omega * b - (omega * U + (omega - 1.0) * D) @ T
        T_new = spla.spsolve_triangular(A_esq, rhs, lower=True)
        res = np.linalg.norm(M @ T_new - b) / nb
        hist.append(res)
        T = T_new
        if callback is not None and (it % callback_step == 0 or res < tol):
            callback(it + 1, res, T_new)
        if res < tol:
            return T, np.array(hist), it + 1, True
    return T, np.array(hist), max_iter, False


def resolver_direto(M, b, T0=None, callback=None, callback_step=5, **kwargs):
    """
    Solver direto (LU esparsa). ATENÇÃO CONCEITUAL: a ausência de iterações
    NÃO significa erro zero. O resíduo algébrico ||M@T - b|| ainda existe
    (arredondamento de ponto flutuante na fatoração/substituição, maior
    quanto pior condicionada for M -- ex.: malha muito fina, Δt muito
    pequeno/grande no implícito). Esse resíduo é calculado e reportado aqui
    como qualquer outro solver, em vez de assumir 0.0.

    IMPORTANTE, e é uma distinção conceitual central: este resíduo mede
    apenas o quão bem a equação DISCRETA (A T = b) foi satisfeita -- o
    chamado erro algébrico/de fechamento. Ele NÃO mede o erro de truncamento
    da discretização em si (O(dx^2) no espaço, O(dt) no Euler implícito)
    frente à EDP contínua -- esse é um erro completamente diferente, e
    existe mesmo com resíduo algébrico zero. Estimar o erro de truncamento
    exige um estudo de refino de malha/Δt (ex.: extrapolação de Richardson,
    índice GCI), não a leitura do resíduo do solver linear.
    """
    T = spla.spsolve(M.tocsc(), b)
    res = float(np.linalg.norm(M @ T - b) / (np.linalg.norm(b) + 1e-30))
    if callback is not None:
        callback(1, res, T)
    return T, np.array([res]), 1, True


SOLVERS = {
    "jacobi": resolver_jacobi,
    "gauss_seidel": resolver_gauss_seidel,
    "sor": resolver_sor,
    "direto": resolver_direto,
}


def resolver(metodo: str, M, b, T0, **kwargs):
    if metodo not in SOLVERS:
        raise ValueError(f"Método '{metodo}' desconhecido. Opções: {list(SOLVERS)}")
    return SOLVERS[metodo](M, b, T0, **kwargs)


# ---------------------------------------------------------------------------
# Fluxo de calor (Lei de Fourier), para pós-processamento/visualização
# ---------------------------------------------------------------------------

def norma_referencia_permanente(F, dirichlet_mask, dirichlet_vals):
    """
    Norma do vetor 'b' do sistema permanente (F com os nós Dirichlet
    sobrescritos pelos valores prescritos) -- usada como referência de
    normalização do resíduo em AMBOS os regimes, para que permanente e
    transiente reportem o erro na MESMA escala (ver residuo_pde abaixo).
    """
    b = -F.copy()
    b[dirichlet_mask] = dirichlet_vals[dirichlet_mask]
    return float(np.linalg.norm(b)) + 1e-30


def residuo_pde(A, F, T, norma_ref):
    """
    Resíduo unificado ||A@T + F|| / norma_ref -- MESMA fórmula usada em
    regime permanente e transiente (ver nota em build_implicit_matrix):

        Regime permanente: A@T = -F  é exatamente o limite Δt->infinito de
        (I - Δt*alpha*A)@T = T_ant + Δt*alpha*F. Por isso a formulação
        algébrica é sempre "transiente" -- em regime permanente, Δt é só um
        artifício de relaxação/estabilização numérica (cada iteração do
        Jacobi/GS/SOR/direto é puramente algébrica, sem significado
        temporal), enquanto em regime transiente Δt é físico, T carrega a
        condição inicial de verdade e evolui no tempo real.

        Este resíduo mede exatamente a mesma grandeza nos dois casos:
        o quão longe T está de satisfazer a equação de Laplace permanente.
        Em regime permanente, ele decai ITERAÇÃO a ITERAÇÃO até o sistema
        "fechar". Em regime transiente, ele decai PASSO DE TEMPO a PASSO DE
        TEMPO conforme o transiente se aproxima do regime permanente --
        calculável para o esquema explícito e o implícito igualmente, pois
        não depende de nenhum solver linear ter sido chamado naquele passo.
    """
    return float(np.linalg.norm(A @ T + F)) / norma_ref


def fluxo_calor(mesh: Mesh2D, T_campo: np.ndarray, k: float):
    """
    Calcula o vetor fluxo de calor q = -k*grad(T) por diferenças centradas
    (np.gradient: 2a ordem no interior, 1a ordem nas bordas do array).

    T_campo: array (Ny, Nx). Retorna qx, qy, |q| -- todos (Ny, Nx), em W/m2.
    """
    dTdy, dTdx = np.gradient(T_campo, mesh.dy, mesh.dx)
    qx = -k * dTdx
    qy = -k * dTdy
    q_mag = np.sqrt(qx**2 + qy**2)
    return qx, qy, q_mag
