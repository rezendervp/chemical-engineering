"""
worker.py
---------
Execução da simulação em uma thread separada da GUI.

Por que uma thread separada?
    O Tkinter roda um único loop de eventos. Se a solução numérica rodasse
    no mesmo thread da interface, a janela congelaria até o fim do cálculo.
    A comunicação de volta para a GUI passa por uma `queue.Queue` thread-safe;
    a janela principal faz polling periódico (ver main_window.py).

Mensagens colocadas na fila (tuplas, primeiro elemento = tipo):
    ("status", texto)
    ("progresso", fracao_0_a_1)            -- para a barra determinística
    ("convergencia", iteracao, residuo)     -- permanente: 1 ponto/iteração
    ("convergencia_passo", n_passo, residuo)-- transiente implícito: 1 ponto/passo
    ("amostra", x_ou_t, valor)              -- ponto monitorado (opcional)
    ("concluido", resultado_dict)
    ("cancelado", texto)
    ("erro", texto_excecao)

OTIMIZAÇÃO DE DESEMPENHO IMPORTANTE (esquema implícito):
    A matriz M = I - dt*alpha*A do esquema implícito é CONSTANTE ao longo de
    todos os passos de tempo (só depende de A, dt, alpha -- não de T_n).
    Antes, o código reconstruía a matriz esparsa inteira (incluindo a
    conversão .tolil(), cara) a cada passo -- para uma simulação com
    centenas de passos isso desperdiça a maior parte do tempo e podia travar
    a interface. Agora a matriz é montada UMA VEZ; a cada passo só o vetor
    b (O(N), barato) é recalculado. Quando o solver escolhido é "direto",
    também fatoramos a matriz uma única vez (splu) e reaproveitamos a
    fatoração em todos os passos (troca uma fatoração LU completa por
    simples substituições triangulares a cada passo).
"""

import threading
import queue
import time
import numpy as np

from app.core import solver as sv
from app.core.mesh import Mesh2D
from app.core.bc import ContornosRetangulo


class SimulacaoWorker(threading.Thread):
    def __init__(self, params: dict, fila: "queue.Queue"):
        super().__init__(daemon=True)
        self.params = params
        self.fila = fila
        self._cancelar = threading.Event()

    def cancelar(self):
        self._cancelar.set()

    def _checar_cancelamento(self):
        if self._cancelar.is_set():
            raise InterruptedError("Simulação cancelada pelo usuário.")

    # -----------------------------------------------------------------
    def run(self):
        try:
            p = self.params
            mesh: Mesh2D = p["mesh"]
            contornos: ContornosRetangulo = p["contornos"]
            k = p["material"].k
            alpha = p["material"].alpha
            idx_amostra = p.get("idx_amostra")  # índice global do nó monitorado, opcional

            self.fila.put(("status", "Montando o operador de diferenças finitas..."))
            A, F, dmask, dvals = sv.assemble_laplaciano(mesh, contornos, k)

            if p["regime"] == "permanente":
                self._rodar_permanente(p, mesh, A, F, dmask, dvals, idx_amostra)
            else:
                self._rodar_transiente(p, mesh, A, F, dmask, dvals, alpha, idx_amostra)

        except InterruptedError as e:
            self.fila.put(("cancelado", str(e)))
        except Exception as e:
            import traceback
            self.fila.put(("erro", f"{e}\n\n{traceback.format_exc()}"))

    # -----------------------------------------------------------------
    def _rodar_permanente(self, p, mesh, A, F, dmask, dvals, idx_amostra):
        """
        Regime permanente = mesma formulação algébrica do transiente,
        (I - Δt·α·A)@T = T_ant + Δt·α·F, com Δt como parâmetro de RELAXAÇÃO
        numérica (não físico) -- análogo ao ω do SOR: Δt grande ~ comporta-
        mento quase direto (poucos passos de relaxação); Δt pequeno ~
        relaxação mais lenta e gradual. O usuário controla Δt livremente.

        Cada "passo de relaxação" resolve o sistema daquele passo (via
        Jacobi/GS/SOR/direto, plenamente convergido dentro do passo) e então
        atualiza T_ant = T_novo para o próximo passo, até o resíduo
        estacionário ‖A·T+F‖ (mesma métrica usada no transiente) cair abaixo
        da tolerância. A contagem de iteração mostrada ao usuário é GLOBAL
        (soma das iterações internas de todos os passos de relaxação) --
        com o Δt padrão (grande), isso reproduz de perto o comportamento do
        sistema direto clássico A·T=-F em um único passo.
        """
        alpha = p["material"].alpha
        dt = p["dt_relaxacao"]
        max_passos = p.get("max_passos_relax", 200)
        norma_ref = sv.norma_referencia_permanente(F, dmask, dvals)

        self.fila.put(("status", f"Regime permanente via relaxação (Δt={dt:.4g} s, "
                                  f"{p['solver_metodo']})..."))
        M = sv.build_implicit_matrix(A, dmask, dt, alpha)

        T = np.full(mesh.n_nos, p["T_inicial"])
        T[dmask] = dvals[dmask]

        res0_ref = {"valor": None}
        it_global = {"valor": 0}

        def cb(it_local, res, T_atual):
            self._checar_cancelamento()
            it_global["valor"] += 1
            itg = it_global["valor"]
            if res0_ref["valor"] is None and res > 0:
                res0_ref["valor"] = res
            self.fila.put(("convergencia", itg, res))
            if itg % 5 == 0 or res < p["tolerancia"]:
                tol = max(p["tolerancia"], 1e-300)
                r0 = res0_ref["valor"] or res
                if res > 0 and r0 > tol:
                    frac = np.log10(r0 / max(res, tol)) / np.log10(r0 / tol)
                else:
                    frac = 1.0
                self.fila.put(("progresso", float(np.clip(frac, 0.0, 1.0))))
                self.fila.put(("status", f"Passo de relaxação {passo_atual[0]}  |  "
                                          f"iteração global {itg}  |  resíduo = {res:.3e}"))
            if idx_amostra is not None:
                self.fila.put(("amostra", itg, float(T_atual[idx_amostra])))

        passo_atual = [0]
        convergiu = False
        n_iter_total = 0
        for passo in range(1, max_passos + 1):
            self._checar_cancelamento()
            passo_atual[0] = passo
            b = sv.build_implicit_rhs(F, dmask, dvals, T, dt, alpha)
            kwargs = dict(tol=p["tolerancia"], max_iter=p["max_iter"], callback=cb)
            if p["solver_metodo"] == "sor":
                kwargs["omega"] = p["omega"]
            T, hist, n_iter, _ = sv.resolver(p["solver_metodo"], M, b, T.copy(), **kwargs)
            n_iter_total += n_iter

            residuo_estacionario = sv.residuo_pde(A, F, T, norma_ref)
            if residuo_estacionario < p["tolerancia"]:
                convergiu = True
                break

        resultado = {
            "tipo": "permanente",
            "T_campo": mesh.para_campo(T),
            "hist_convergencia": np.array([]),
            "n_iter": n_iter_total,
            "n_passos_relaxacao": passo,
            "convergiu": convergiu,
        }
        self.fila.put(("progresso", 1.0))
        self.fila.put(("concluido", resultado))

    # -----------------------------------------------------------------
    def _rodar_transiente(self, p, mesh, A, F, dmask, dvals, alpha, idx_amostra):
        dt = p["dt"]
        t_final = p["t_final"]
        esquema = p["esquema"]
        n_passos = int(np.ceil(t_final / dt))
        n_frames_alvo = p.get("n_frames_alvo", 80)
        passo_save = max(1, n_passos // n_frames_alvo)
        passo_status = max(1, n_passos // 200)  # status/progresso a cada ~0.5% dos passos, mín. 5 em 5
        passo_status = min(passo_status, 5) if n_passos >= 5 else 1

        T = np.full(mesh.n_nos, p["T_inicial"])
        T[dmask] = dvals[dmask]

        frames = [mesh.para_campo(T).copy()]
        tempos = [0.0]
        t = 0.0

        # resíduo = FECHAMENTO DO BALANÇO DE ENERGIA DISCRETO no passo (‖M·T-b‖
        # do sistema linear DAQUELE passo específico) -- isto sim é erro: mede
        # se acúmulo - (E_entra - E_sai) = 0 foi satisfeito numericamente.
        # NÃO é a distância do regime permanente (essa não é erro nenhum --
        # um transiente estar longe do equilíbrio não viola conservação de
        # energia alguma; o balanço pode fechar perfeitamente em cada passo
        # mesmo estando longe do estado estacionário).
        self.fila.put(("status", f"Regime transiente ({esquema}): {n_passos} passos..."))
        self.fila.put(("frame_transiente", 0.0, frames[0]))
        self.fila.put(("convergencia_passo", 0.0, 1e-300))  # t=0: T=condição inicial, sem passo ainda

        # --- pré-computação para o esquema implícito (feita UMA VEZ, fora do laço) ---
        M_implicito = None
        lu_implicito = None
        if esquema == "implicito":
            self.fila.put(("status", "Montando a matriz do esquema implícito (uma única vez)..."))
            M_implicito = sv.build_implicit_matrix(A, dmask, dt, alpha)
            if p["solver_metodo"] == "direto":
                self.fila.put(("status", "Fatorando a matriz (LU esparsa, reaproveitada em todos os passos)..."))
                lu_implicito = sv.fatorar_direto(M_implicito)

        residuo_maximo_observado = 0.0

        for n in range(1, n_passos + 1):
            self._checar_cancelamento()

            if esquema == "explicito":
                # M = I (ver discussão): b_passo é exatamente o que passo_explicito
                # atribui a T_novo -- residuo é por construção ~0 (só ruído de
                # ponto flutuante), pois não há aproximação alguma na álgebra
                # (o único erro do explícito é de truncamento no tempo, não
                # algébrico -- não aparece neste resíduo, ver nota em solver.py)
                b_passo = T + dt * alpha * (A @ T + F)
                b_passo[dmask] = dvals[dmask]
                T_novo = sv.passo_explicito(A, F, dmask, dvals, T, dt, alpha)
                residuo_passo = (float(np.linalg.norm(T_novo - b_passo)) /
                                  (float(np.linalg.norm(b_passo)) + 1e-30))
                T = T_novo
            else:  # implicito
                b = sv.build_implicit_rhs(F, dmask, dvals, T, dt, alpha)
                if lu_implicito is not None:
                    T_novo = sv.resolver_direto_fatorado(lu_implicito, b)
                else:
                    kwargs = dict(tol=p["tolerancia"], max_iter=p["max_iter"])
                    if p["solver_metodo"] == "sor":
                        kwargs["omega"] = p["omega"]
                    T_novo, hist_passo, _, _ = sv.resolver(p["solver_metodo"], M_implicito, b,
                                                            T.copy(), **kwargs)
                residuo_passo = (float(np.linalg.norm(M_implicito @ T_novo - b)) /
                                  (float(np.linalg.norm(b)) + 1e-30))
                T = T_novo

            residuo_maximo_observado = max(residuo_maximo_observado, residuo_passo)
            t += dt

            if idx_amostra is not None:
                self.fila.put(("amostra", t, float(T[idx_amostra])))

            if n % passo_save == 0 or n == n_passos:
                campo = mesh.para_campo(T).copy()
                frames.append(campo)
                tempos.append(t)
                self.fila.put(("frame_transiente", t, campo))

            if n % passo_status == 0 or n == n_passos:
                self.fila.put(("convergencia_passo", t, max(residuo_passo, 1e-300)))
                self.fila.put(("progresso", n / n_passos))
                self.fila.put(("status", f"Passo {n}/{n_passos}  |  t = {t:.3g} s / {t_final:.3g} s "
                                          f"({100*n/n_passos:.0f}%)  |  "
                                          f"resíduo do balanço = {residuo_passo:.2e}"))

        self.fila.put(("status", f"Concluído. Resíduo MÁXIMO observado em qualquer passo: "
                                  f"{residuo_maximo_observado:.3e}"))

        resultado = {
            "tipo": "transiente",
            "frames": frames,
            "tempos": tempos,
            "T_campo": frames[-1],
            "residuo_maximo": residuo_maximo_observado,
        }
        self.fila.put(("progresso", 1.0))
        self.fila.put(("concluido", resultado))
