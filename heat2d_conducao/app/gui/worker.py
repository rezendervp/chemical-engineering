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
        self.fila.put(("status", f"Resolvendo regime permanente ({p['solver_metodo']})..."))
        M, b = sv.build_steady_system(A, F, dmask, dvals)
        T0 = np.full(mesh.n_nos, p["T_inicial"])

        res0_ref = {"valor": None}  # primeiro resíduo, usado como referência p/ progresso em log

        def cb(it, res, T_atual):
            self._checar_cancelamento()
            if res0_ref["valor"] is None and res > 0:
                res0_ref["valor"] = res
            self.fila.put(("convergencia", it, res))
            if it % 5 == 0 or res < p["tolerancia"]:
                tol = max(p["tolerancia"], 1e-300)
                r0 = res0_ref["valor"] or res
                if res > 0 and r0 > tol:
                    frac = np.log10(r0 / max(res, tol)) / np.log10(r0 / tol)
                else:
                    frac = 1.0
                self.fila.put(("progresso", float(np.clip(frac, 0.0, 1.0))))
                self.fila.put(("status", f"Iteração {it}  |  resíduo = {res:.3e}"))
            if idx_amostra is not None:
                self.fila.put(("amostra", it, float(T_atual[idx_amostra])))

        kwargs = dict(tol=p["tolerancia"], max_iter=p["max_iter"], callback=cb)
        if p["solver_metodo"] == "sor":
            kwargs["omega"] = p["omega"]
        T, hist, n_iter, convergiu = sv.resolver(p["solver_metodo"], M, b, T0, **kwargs)

        resultado = {
            "tipo": "permanente",
            "T_campo": mesh.para_campo(T),
            "hist_convergencia": hist,
            "n_iter": n_iter,
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

        # norma de referência para o resíduo -- MESMA usada em regime permanente
        # (ver residuo_pde), garantindo que o erro seja reportado na mesma escala
        # nos dois regimes
        norma_ref = sv.norma_referencia_permanente(F, dmask, dvals)

        self.fila.put(("status", f"Regime transiente ({esquema}): {n_passos} passos..."))
        self.fila.put(("frame_transiente", 0.0, frames[0]))
        self.fila.put(("convergencia_passo", 0.0,
                        max(sv.residuo_pde(A, F, T, norma_ref), 1e-300)))

        # --- pré-computação para o esquema implícito (feita UMA VEZ, fora do laço) ---
        M_implicito = None
        lu_implicito = None
        if esquema == "implicito":
            self.fila.put(("status", "Montando a matriz do esquema implícito (uma única vez)..."))
            M_implicito = sv.build_implicit_matrix(A, dmask, dt, alpha)
            if p["solver_metodo"] == "direto":
                self.fila.put(("status", "Fatorando a matriz (LU esparsa, reaproveitada em todos os passos)..."))
                lu_implicito = sv.fatorar_direto(M_implicito)

        for n in range(1, n_passos + 1):
            self._checar_cancelamento()

            if esquema == "explicito":
                T = sv.passo_explicito(A, F, dmask, dvals, T, dt, alpha)
            else:  # implicito
                b = sv.build_implicit_rhs(F, dmask, dvals, T, dt, alpha)
                if lu_implicito is not None:
                    T = sv.resolver_direto_fatorado(lu_implicito, b)
                else:
                    kwargs = dict(tol=p["tolerancia"], max_iter=p["max_iter"])
                    if p["solver_metodo"] == "sor":
                        kwargs["omega"] = p["omega"]
                    T, hist_passo, _, _ = sv.resolver(p["solver_metodo"], M_implicito, b, T.copy(), **kwargs)

            t += dt

            if idx_amostra is not None:
                self.fila.put(("amostra", t, float(T[idx_amostra])))

            if n % passo_save == 0 or n == n_passos:
                campo = mesh.para_campo(T).copy()
                frames.append(campo)
                tempos.append(t)
                self.fila.put(("frame_transiente", t, campo))

            if n % passo_status == 0 or n == n_passos:
                # resíduo unificado ||A@T + F|| -- mede a distância do regime
                # permanente, calculável para QUALQUER esquema (explícito ou
                # implícito), pois não depende de nenhum solve linear ter sido
                # feito neste passo especificamente
                residuo = sv.residuo_pde(A, F, T, norma_ref)
                self.fila.put(("convergencia_passo", t, max(residuo, 1e-300)))
                self.fila.put(("progresso", n / n_passos))
                self.fila.put(("status", f"Passo {n}/{n_passos}  |  t = {t:.3g} s / {t_final:.3g} s "
                                          f"({100*n/n_passos:.0f}%)"))

        resultado = {
            "tipo": "transiente",
            "frames": frames,
            "tempos": tempos,
            "T_campo": frames[-1],
            "tem_residuo": True,  # resíduo unificado agora calculado p/ qualquer esquema
        }
        self.fila.put(("progresso", 1.0))
        self.fila.put(("concluido", resultado))
