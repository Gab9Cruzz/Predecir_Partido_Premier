"""Modelo Dixon-Coles (1997) con ponderacion temporal.

Que problema resuelve
---------------------
El enfoque ingenuo ("promedio de goles a favor del local + promedio de goles
en contra del visitante") tiene un fallo grave: NO corrige por la calidad del
rival. Un equipo que acaba de jugar contra los cinco ultimos de la tabla
parece un equipo brutal, y uno que jugo contra los cinco primeros parece
malisimo. Ese sesgo es enorme y no se arregla con promedios.

Dixon-Coles estima a la vez, por maxima verosimilitud, un parametro de ataque
y uno de defensa para CADA equipo, de forma que los goles observados sean lo
mas probables posible. Como todos los equipos aparecen en la misma ecuacion,
la fuerza del rival queda descontada automaticamente.

    goles_local  ~ Poisson(exp(mu + ataque_local  - defensa_visita + ventaja))
    goles_visita ~ Poisson(exp(mu + ataque_visita - defensa_local))

Sobre eso se le anaden tres cosas:

1. Correccion tau para marcadores bajos. Un Poisson doble independiente
   subestima los 0-0 y 1-1 y sobreestima los 1-0 / 0-1. El parametro rho
   corrige exactamente esas cuatro celdas. Se traduce en empates mejor
   calibrados, que es donde mas fallan los modelos caseros.

2. Ponderacion temporal exponencial: peso = exp(-xi * dias_de_antiguedad).
   Un partido de hace un mes informa mas que uno de hace dos anos, pero el de
   hace dos anos sigue informando algo. Con xi = 0.0022 un partido pierde la
   mitad de su peso cada ~315 dias.

3. Regularizacion L2 (ridge) sobre ataque y defensa. Cumple dos funciones:
   fija la indeterminacion del modelo (se puede sumar una constante a todos
   los ataques sin cambiar nada) y encoge hacia la media a los equipos con
   pocos partidos: un recien ascendido con 3 jornadas no recibe un rating
   extremo solo por una racha corta.

Referencia: Dixon, M.J. y Coles, S.G. (1997), "Modelling Association Football
Scores and Inefficiencies in the Football Betting Market", Applied Statistics
46(2), 265-280.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from futbol.fuentes.base import Partido, buscar_equipo

# Valor elegido optimizando el RPS del backtest walk-forward en LaLiga,
# Premier, Bundesliga y Serie A (6 temporadas cada una). El optimo de cada liga
# cae entre 0.0015 y 0.0030; 0.0018 esta cerca del mejor en las cuatro.
# Puedes recalcularlo para tu liga con: python predecir.py optimizar
XI_POR_DEFECTO = 0.0018        # semivida ~385 dias
# Tambien elegida por backtest (LaLiga, Premier, Serie A). Entre 1 y 4 el
# resultado apenas cambia; por encima de 8 empeora claramente.
REGULARIZACION_POR_DEFECTO = 2.0
MAX_GOLES = 10


@dataclass
class Prediccion:
    """Resultado completo de predecir un partido."""

    local: str
    visitante: str
    goles_esperados_local: float
    goles_esperados_visitante: float
    prob_local: float
    prob_empate: float
    prob_visitante: float
    matriz: np.ndarray = field(repr=False)
    partidos_local: int = 0
    partidos_visitante: int = 0
    # Cuando el nombre pedido no coincidia exactamente con el del dataset,
    # aqui queda constancia de con que se emparejo y con cuanta confianza.
    emparejamientos: list[tuple[str, str, float]] = field(default_factory=list)

    # -------------------------------------------------------------- derivados

    @property
    def marcadores_probables(self) -> list[tuple[str, float]]:
        """Los marcadores exactos mas probables, de mayor a menor."""
        planos = [
            (f"{i}-{j}", float(self.matriz[i, j]))
            for i in range(self.matriz.shape[0])
            for j in range(self.matriz.shape[1])
        ]
        planos.sort(key=lambda x: -x[1])
        return planos[:10]

    def prob_mas_de(self, linea: float = 2.5) -> float:
        """Probabilidad de que el total de goles supere la linea."""
        n = self.matriz.shape[0]
        totales = np.add.outer(np.arange(n), np.arange(n))
        return float(self.matriz[totales > linea].sum())

    @property
    def prob_ambos_marcan(self) -> float:
        return float(self.matriz[1:, 1:].sum())

    @property
    def prob_local_o_empate(self) -> float:
        return self.prob_local + self.prob_empate

    @property
    def prob_visitante_o_empate(self) -> float:
        return self.prob_visitante + self.prob_empate

    def prob_handicap_asiatico(self, handicap: float) -> float:
        """Probabilidad de que el local cubra el handicap (ej. -1.5)."""
        n = self.matriz.shape[0]
        dif = np.subtract.outer(np.arange(n), np.arange(n))
        return float(self.matriz[dif + handicap > 0].sum())

    @property
    def cuotas_justas(self) -> tuple[float, float, float]:
        """Cuota decimal sin margen que corresponde a cada probabilidad."""
        return tuple(
            1.0 / p if p > 1e-9 else float("inf")
            for p in (self.prob_local, self.prob_empate, self.prob_visitante)
        )

    @property
    def probs(self) -> np.ndarray:
        return np.array([self.prob_local, self.prob_empate, self.prob_visitante])


def _tau(x: np.ndarray, y: np.ndarray, lh: np.ndarray, la: np.ndarray,
         rho: float) -> np.ndarray:
    """Correccion Dixon-Coles para los cuatro marcadores bajos."""
    t = np.ones_like(lh)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    t[m00] = 1.0 - lh[m00] * la[m00] * rho
    t[m01] = 1.0 + lh[m01] * rho
    t[m10] = 1.0 + la[m10] * rho
    t[m11] = 1.0 - rho
    return t


class DixonColes:
    """Ajusta el modelo sobre una lista de partidos y predice partidos nuevos."""

    def __init__(self, xi: float = XI_POR_DEFECTO,
                 regularizacion: float = REGULARIZACION_POR_DEFECTO,
                 max_goles: int = MAX_GOLES):
        self.xi = xi
        self.regularizacion = regularizacion
        self.max_goles = max_goles

        self.equipos: list[str] = []
        self._indice: dict[str, int] = {}
        self.ligas: list[str] = []
        self._indice_liga: dict[str, int] = {}

        self.ataque: np.ndarray | None = None
        self.defensa: np.ndarray | None = None
        self.ventaja_local: float = 0.0
        self.rho: float = 0.0
        self.mu: np.ndarray | None = None

        self.n_partidos: int = 0
        self.partidos_por_equipo: dict[str, int] = {}
        self.fecha_referencia: date | None = None
        self.convergio: bool = False
        self.log_verosimilitud: float = 0.0

    # ------------------------------------------------------------------ ajuste

    def ajustar(self, partidos: list[Partido],
                referencia: date | None = None) -> "DixonColes":
        """Estima los parametros a partir de los partidos ya jugados."""
        jugados = [p for p in partidos if p.jugado]
        if len(jugados) < 30:
            raise ValueError(
                f"Solo hay {len(jugados)} partidos jugados. Hacen falta al menos 30 "
                f"para ajustar el modelo; prueba con mas temporadas (--temporadas)."
            )

        self.fecha_referencia = referencia or max(p.fecha for p in jugados)

        self.equipos = sorted({p.local for p in jugados} | {p.visitante for p in jugados})
        self._indice = {e: i for i, e in enumerate(self.equipos)}
        self.ligas = sorted({p.liga for p in jugados})
        self._indice_liga = {lg: i for i, lg in enumerate(self.ligas)}

        n = len(self.equipos)
        n_ligas = len(self.ligas)
        self.n_partidos = len(jugados)

        idx_local = np.array([self._indice[p.local] for p in jugados])
        idx_visit = np.array([self._indice[p.visitante] for p in jugados])
        idx_liga = np.array([self._indice_liga[p.liga] for p in jugados])
        gl = np.array([p.goles_local for p in jugados], dtype=float)
        gv = np.array([p.goles_visitante for p in jugados], dtype=float)

        dias = np.array([(self.fecha_referencia - p.fecha).days for p in jugados],
                        dtype=float)
        peso = np.exp(-self.xi * np.maximum(dias, 0.0))

        self.partidos_por_equipo = {}
        for p in jugados:
            self.partidos_por_equipo[p.local] = self.partidos_por_equipo.get(p.local, 0) + 1
            self.partidos_por_equipo[p.visitante] = self.partidos_por_equipo.get(p.visitante, 0) + 1

        reg = self.regularizacion

        def desempaquetar(theta):
            ataque = theta[:n]
            defensa = theta[n:2 * n]
            ventaja = theta[2 * n]
            rho = theta[2 * n + 1]
            mu = theta[2 * n + 2:]
            return ataque, defensa, ventaja, rho, mu

        def objetivo_y_gradiente(theta):
            ataque, defensa, ventaja, rho, mu = desempaquetar(theta)

            log_lh = mu[idx_liga] + ataque[idx_local] - defensa[idx_visit] + ventaja
            log_la = mu[idx_liga] + ataque[idx_visit] - defensa[idx_local]
            # Los limites de los parametros ya acotan exp() a un rango sano,
            # asi que no hace falta recortar (recortar romperia el gradiente).
            lh = np.exp(log_lh)
            la = np.exp(log_la)

            tau = _tau(gl, gv, lh, la, rho)
            tau = np.maximum(tau, 1e-10)

            ll = peso * (np.log(tau) - lh + gl * log_lh - la + gv * log_la)
            objetivo = -ll.sum() + reg * (np.sum(ataque ** 2) + np.sum(defensa ** 2))

            # --- gradiente analitico ---
            # Derivada de log tau respecto a lh, la y rho (solo celdas bajas).
            dtau_dlh = np.zeros_like(lh)
            dtau_dla = np.zeros_like(la)
            dtau_drho = np.zeros_like(lh)
            m00 = (gl == 0) & (gv == 0)
            m01 = (gl == 0) & (gv == 1)
            m10 = (gl == 1) & (gv == 0)
            m11 = (gl == 1) & (gv == 1)
            dtau_dlh[m00] = -la[m00] * rho
            dtau_dla[m00] = -lh[m00] * rho
            dtau_drho[m00] = -lh[m00] * la[m00]
            dtau_dlh[m01] = rho
            dtau_drho[m01] = lh[m01]
            dtau_dla[m10] = rho
            dtau_drho[m10] = la[m10]
            dtau_drho[m11] = -1.0

            # d(loglik)/d(log lambda) para cada partido.
            d_lh = peso * ((gl - lh) + lh * dtau_dlh / tau)
            d_la = peso * ((gv - la) + la * dtau_dla / tau)
            d_rho = float((peso * dtau_drho / tau).sum())

            grad = np.zeros_like(theta)
            # ataque: aparece en +log_lh del local y +log_la del visitante
            np.add.at(grad, idx_local, -d_lh)
            np.add.at(grad, idx_visit, -d_la)
            # defensa: aparece con signo negativo
            np.add.at(grad, n + idx_visit, d_lh)
            np.add.at(grad, n + idx_local, d_la)
            grad[2 * n] = -d_lh.sum()
            grad[2 * n + 1] = -d_rho
            np.add.at(grad, 2 * n + 2 + idx_liga, -(d_lh + d_la))

            grad[:n] += 2 * reg * ataque
            grad[n:2 * n] += 2 * reg * defensa

            return objetivo, grad

        theta0 = np.concatenate([
            np.zeros(n),                                   # ataque
            np.zeros(n),                                   # defensa
            [0.25],                                        # ventaja de local
            [-0.05],                                       # rho
            np.full(n_ligas, np.log(max(gl.mean(), 0.3))),  # nivel de cada liga
        ])
        limites = (
            [(-2.0, 2.0)] * n
            + [(-2.0, 2.0)] * n
            + [(-1.0, 1.5)]      # ventaja de local
            + [(-0.35, 0.35)]    # rho
            + [(-1.5, 2.0)] * n_ligas
        )

        res = minimize(objetivo_y_gradiente, theta0, jac=True,
                       method="L-BFGS-B", bounds=limites,
                       options={"maxiter": 2000, "ftol": 1e-10})

        ataque, defensa, ventaja, rho, mu = desempaquetar(res.x)
        self.ataque = ataque
        self.defensa = defensa
        self.ventaja_local = float(ventaja)
        self.rho = float(rho)
        self.mu = mu
        self.convergio = bool(res.success)
        self.log_verosimilitud = float(-res.fun)
        return self

    # --------------------------------------------------------------- prediccion

    def _idx(self, equipo: str) -> int:
        if equipo in self._indice:
            return self._indice[equipo]
        sugerencias = buscar_equipo(equipo, self.equipos, n=5)
        if sugerencias:
            raise KeyError(
                f"'{equipo}' no esta en los datos. Quizas quisiste decir: "
                + ", ".join(f"'{nombre}'" for nombre, _ in sugerencias)
            )
        raise KeyError(f"'{equipo}' no esta en los datos de esta liga.")

    def resolver_equipo(self, consulta: str) -> tuple[str, float]:
        """Convierte un nombre aproximado en el exacto del dataset.

        Devuelve (nombre, confianza). Confianza 1.0 = coincidencia exacta.
        """
        if consulta in self._indice:
            return consulta, 1.0
        candidatos = buscar_equipo(consulta, self.equipos, n=2)
        if not candidatos:
            raise KeyError(
                f"No encuentro ningun equipo parecido a '{consulta}'. "
                f"Corre `python predecir.py equipos` para ver la lista."
            )
        return candidatos[0]

    def lambdas(self, local: str, visitante: str,
                liga: str | None = None) -> tuple[float, float]:
        """Goles esperados de cada equipo (sin convertir a probabilidades)."""
        i, j = self._idx(local), self._idx(visitante)
        if liga and liga in self._indice_liga:
            mu = float(self.mu[self._indice_liga[liga]])
        else:
            mu = float(np.mean(self.mu))
        lh = np.exp(mu + self.ataque[i] - self.defensa[j] + self.ventaja_local)
        la = np.exp(mu + self.ataque[j] - self.defensa[i])
        return float(lh), float(la)

    def predecir(self, local: str, visitante: str, liga: str | None = None,
                 campo_neutral: bool = False, estricto: bool = False) -> Prediccion:
        """Predice un partido y devuelve la distribucion completa de marcadores.

        estricto=True exige que los nombres coincidan exactamente con los del
        dataset. Lo usa el backtest: ahi un nombre desconocido significa equipo
        sin historico, y emparejarlo por parecido con otro equipo falsearia el
        resultado.
        """
        if self.ataque is None:
            raise RuntimeError("El modelo no esta ajustado: llama antes a ajustar().")

        emparejamientos: list[tuple[str, str, float]] = []
        if not estricto:
            for consulta, destino in (("local", local), ("visitante", visitante)):
                resuelto, confianza = self.resolver_equipo(destino)
                if resuelto != destino:
                    emparejamientos.append((destino, resuelto, confianza))
                if consulta == "local":
                    local = resuelto
                else:
                    visitante = resuelto
        i, j = self._idx(local), self._idx(visitante)

        if liga and liga in self._indice_liga:
            mu = float(self.mu[self._indice_liga[liga]])
        else:
            mu = float(np.mean(self.mu))

        # En campo neutral (finales a sede unica) la ventaja se reparte a medias
        # entre los dos: se mantiene el nivel de goles, se anula el desnivel.
        if campo_neutral:
            plus_local = plus_visita = self.ventaja_local / 2
        else:
            plus_local, plus_visita = self.ventaja_local, 0.0
        lh = float(np.exp(mu + self.ataque[i] - self.defensa[j] + plus_local))
        la = float(np.exp(mu + self.ataque[j] - self.defensa[i] + plus_visita))

        n = self.max_goles + 1
        goles = np.arange(n)
        pmf_local = poisson.pmf(goles, lh)
        pmf_visit = poisson.pmf(goles, la)
        matriz = np.outer(pmf_local, pmf_visit)

        # Correccion de marcadores bajos.
        matriz[0, 0] *= 1.0 - lh * la * self.rho
        matriz[0, 1] *= 1.0 + lh * self.rho
        matriz[1, 0] *= 1.0 + la * self.rho
        matriz[1, 1] *= 1.0 - self.rho
        matriz = np.maximum(matriz, 0.0)
        matriz /= matriz.sum()

        return Prediccion(
            local=local,
            visitante=visitante,
            goles_esperados_local=lh,
            goles_esperados_visitante=la,
            prob_local=float(np.tril(matriz, -1).sum()),
            prob_empate=float(np.trace(matriz)),
            prob_visitante=float(np.triu(matriz, 1).sum()),
            matriz=matriz,
            partidos_local=self.partidos_por_equipo.get(local, 0),
            partidos_visitante=self.partidos_por_equipo.get(visitante, 0),
            emparejamientos=emparejamientos,
        )

    # ------------------------------------------------------------------ ratings

    def ratings(self) -> list[dict]:
        """Fuerza estimada de cada equipo, ordenada de mejor a peor.

        `general` = ataque + defensa, en escala logaritmica. La diferencia entre
        dos equipos es directamente interpretable: +0.7 significa que el primero
        marca aproximadamente el doble de goles contra el mismo rival.
        """
        if self.ataque is None:
            raise RuntimeError("El modelo no esta ajustado.")
        filas = [
            {
                "equipo": equipo,
                "ataque": float(self.ataque[i]),
                "defensa": float(self.defensa[i]),
                "general": float(self.ataque[i] + self.defensa[i]),
                "partidos": self.partidos_por_equipo.get(equipo, 0),
            }
            for i, equipo in enumerate(self.equipos)
        ]
        filas.sort(key=lambda f: -f["general"])
        for posicion, fila in enumerate(filas, 1):
            fila["puesto"] = posicion
        return filas

    def resumen(self) -> str:
        semivida = np.log(2) / self.xi if self.xi > 0 else float("inf")
        return (
            f"Dixon-Coles ajustado sobre {self.n_partidos} partidos, "
            f"{len(self.equipos)} equipos, {len(self.ligas)} liga(s).\n"
            f"  ventaja de local : {self.ventaja_local:+.3f} log-goles "
            f"(x{np.exp(self.ventaja_local):.2f} de goles esperados)\n"
            f"  rho (empates)    : {self.rho:+.3f}\n"
            f"  semivida temporal: {semivida:.0f} dias\n"
            f"  convergencia     : {'ok' if self.convergio else 'NO convergio del todo'}"
        )
