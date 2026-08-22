"""
worker.py
---------
Execução da simulação em uma thread separada da GUI.

Por que uma thread separada?
    O Tkinter (e o CustomTkinter, que é construído sobre ele) roda um único
    loop de eventos. Se a solução numérica (iterações do solver, passos do
    transiente) rodasse no mesmo thread da interface, a janela congelaria
    até o fim do cálculo -- sem barra de progresso, sem gráfico de
    convergência ao vivo, sem poder cancelar.

    A solução: rodar `SimulacaoWorker` em uma `threading.Thread`, e toda
    comunicação de volta para a GUI passa por uma `queue.Queue` thread-safe.
    A janela principal faz polling da fila periodicamente via `.after(...)`
    (ver main_window.py) -- essa é a forma canônica e segura de integrar
    threads com Tkinter (nunca se deve tocar widgets a partir de outra
    thread diretamente).

Mensagens colocadas na fila (tuplas, primeiro elemento = tipo):
    ("status", texto)
    ("convergencia", iteracao, residuo)
    ("frame_transiente", tempo, T_campo)
    ("concluido", resultado_dict)
    ("erro", texto_excecao)
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

    # -----------------------------------------------------------------
    def run(self):
        try:
            p = self.params
            mesh: Mesh2D = p["mesh"]
            contornos: ContornosRetangulo = p["contornos"]
            k = p["material"].k
            alpha = p["material"].alpha

            self.fila.put(("status", "Montando o operador de diferenças finitas..."))
            A, F, dmask, dvals = sv.assemble_laplaciano(mesh, contornos, k)

            def cb(it, res):
                if self._cancelar.is_set():
                    raise InterruptedError("Simulação cancelada pelo usuário.")
                self.fila.put(("convergencia", it, res))

            if p["regime"] == "permanente":
                self.fila.put(("status", f"Resolvendo regime permanente ({p['solver_metodo']})..."))
                M, b = sv.build_steady_system(A, F, dmask, dvals)
                T0 = np.full(mesh.n_nos, p["T_inicial"])
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
                self.fila.put(("concluido", resultado))

            else:  # transiente
                dt = p["dt"]
                t_final = p["t_final"]
                esquema = p["esquema"]  # 'explicito' ou 'implicito'
                n_passos = int(np.ceil(t_final / dt))
                n_frames_alvo = p.get("n_frames_alvo", 80)
                passo_save = max(1, n_passos // n_frames_alvo)

                T = np.full(mesh.n_nos, p["T_inicial"])
                T[dmask] = dvals[dmask]

                frames = [mesh.para_campo(T).copy()]
                tempos = [0.0]
                t = 0.0

                self.fila.put(("status", f"Regime transiente ({esquema}): {n_passos} passos..."))
                self.fila.put(("frame_transiente", 0.0, frames[0]))

                for n in range(1, n_passos + 1):
                    if self._cancelar.is_set():
                        raise InterruptedError("Simulação cancelada pelo usuário.")

                    if esquema == "explicito":
                        T = sv.passo_explicito(A, F, dmask, dvals, T, dt, alpha)
                    else:  # implicito
                        M, b = sv.build_implicit_system(A, F, dmask, dvals, T, dt, alpha)
                        kwargs = dict(tol=p["tolerancia"], max_iter=p["max_iter"])
                        if p["solver_metodo"] == "sor":
                            kwargs["omega"] = p["omega"]
                        T, _, _, _ = sv.resolver(p["solver_metodo"], M, b, T.copy(), **kwargs)

                    t += dt
                    if n % passo_save == 0 or n == n_passos:
                        campo = mesh.para_campo(T).copy()
                        frames.append(campo)
                        tempos.append(t)
                        self.fila.put(("frame_transiente", t, campo))
                        self.fila.put(("status", f"t = {t:.3g} s / {t_final:.3g} s "
                                                  f"({100*t/t_final:.0f}%)"))

                resultado = {
                    "tipo": "transiente",
                    "frames": frames,
                    "tempos": tempos,
                    "T_campo": frames[-1],  # último instante, para as vistas estáticas
                }
                self.fila.put(("concluido", resultado))

        except InterruptedError as e:
            self.fila.put(("cancelado", str(e)))
        except Exception as e:
            import traceback
            self.fila.put(("erro", f"{e}\n\n{traceback.format_exc()}"))
