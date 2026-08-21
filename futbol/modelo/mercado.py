"""Utilidades para trabajar con cuotas de casas de apuestas.

Las cuotas del mercado son la mejor prediccion publica que existe: incorporan
informacion que ningun modelo casero tiene (lesiones, rotaciones, motivacion,
dinero de gente que sabe). Sirven para dos cosas en este proyecto:

1. Como linea base honesta en el backtest. Si el modelo no le gana al mercado,
   hay que saberlo. Casi ningun modelo casero le gana.
2. Para detectar valor: si el modelo da 45% a un resultado que se paga a 2.60
   (implicito 38%), ahi hay una diferencia que puede ser real... o puede ser
   que el modelo se este equivocando. El backtest es lo que distingue una cosa
   de la otra.
"""

from __future__ import annotations

import numpy as np


def margen_casa(cuotas: tuple[float, ...] | list[float]) -> float:
    """Sobrerredondeo (overround) del mercado: 0.06 = la casa se lleva un 6%."""
    return float(sum(1.0 / c for c in cuotas) - 1.0)


def probabilidades_implicitas(cuotas, metodo: str = "proporcional") -> np.ndarray:
    """Convierte cuotas decimales en probabilidades que suman 1.

    metodo="proporcional": divide todo por la suma. Simple pero sesga hacia
        arriba a los favoritos.
    metodo="potencia": busca k tal que sum(p_i^k) = 1. Corrige el sesgo del
        favorito-perdedor (longshot bias) y calibra bastante mejor.
    """
    inversas = np.array([1.0 / c for c in cuotas], dtype=float)

    if metodo == "proporcional":
        return inversas / inversas.sum()

    if metodo != "potencia":
        raise ValueError(f"Metodo desconocido: {metodo!r}")

    # Biseccion sobre k: sum(p^k) es decreciente en k para p en (0, 1).
    lo, hi = 0.5, 2.0
    for _ in range(60):
        k = (lo + hi) / 2
        if np.sum(inversas ** k) > 1.0:
            lo = k
        else:
            hi = k
    probs = inversas ** ((lo + hi) / 2)
    return probs / probs.sum()


def valor_esperado(prob_modelo: float, cuota: float) -> float:
    """Beneficio esperado por unidad apostada. 0.05 = +5% de EV."""
    return prob_modelo * cuota - 1.0


def kelly(prob_modelo: float, cuota: float, fraccion: float = 0.25) -> float:
    """Fraccion del bankroll a apostar segun el criterio de Kelly.

    Se devuelve ya multiplicado por `fraccion` (Kelly fraccionado). Kelly
    completo es matematicamente optimo solo si tus probabilidades son exactas;
    como no lo son nunca, en la practica se usa un cuarto de Kelly para reducir
    la varianza y el riesgo de ruina.
    """
    b = cuota - 1.0
    if b <= 0:
        return 0.0
    f = (prob_modelo * cuota - 1.0) / b
    return max(0.0, f * fraccion)


def brier(probs: np.ndarray, reales: np.ndarray) -> float:
    """Brier score multiclase. Mas bajo es mejor. Azar puro (1/3 cada uno) = 0.667."""
    return float(((probs - reales) ** 2).sum(axis=1).mean())


def log_loss(probs: np.ndarray, reales: np.ndarray, eps: float = 1e-15) -> float:
    """Log-loss (entropia cruzada). Mas bajo es mejor. Azar puro = 1.099."""
    p = np.clip(probs, eps, 1.0)
    return float(-(reales * np.log(p)).sum(axis=1).mean())


def rps(probs: np.ndarray, reales: np.ndarray) -> float:
    """Ranked Probability Score: la metrica estandar en prediccion de futbol.

    A diferencia del Brier, tiene en cuenta que los resultados estan ordenados
    (local - empate - visitante): fallar prediciendo empate cuando gano el local
    se penaliza menos que predecir victoria visitante. Mas bajo es mejor.
    """
    acum_p = np.cumsum(probs[:, :-1], axis=1)
    acum_r = np.cumsum(reales[:, :-1], axis=1)
    return float(((acum_p - acum_r) ** 2).sum(axis=1).mean() / (probs.shape[1] - 1))


def a_one_hot(resultados: list[str]) -> np.ndarray:
    """['H','D','A'] -> matriz one-hot en el orden local/empate/visitante."""
    mapa = {"H": 0, "D": 1, "A": 2}
    salida = np.zeros((len(resultados), 3))
    for fila, res in enumerate(resultados):
        salida[fila, mapa[res]] = 1.0
    return salida
