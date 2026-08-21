"""Validacion walk-forward del modelo.

Un modelo de prediccion sin backtest es una opinion con decimales. Aqui se
mide de la unica forma valida: recorriendo el historico hacia adelante y
prediciendo cada partido usando SOLO los partidos anteriores a el.

Eso descarta el error mas comun de los predictores caseros, que es entrenar
con datos que incluyen el partido que se quiere predecir (o la clasificacion
final de la temporada, que ya contiene el resultado). El script antiguo de
este proyecto lo hacia: usaba la tabla de posiciones ACTUAL para "predecir"
partidos ya jugados de esa misma temporada.

Metricas
--------
RPS      metrica estandar en futbol; tiene en cuenta el orden 1-X-2. Menor mejor.
Brier    error cuadratico medio sobre las tres probabilidades. Menor mejor.
LogLoss  penaliza mucho la confianza equivocada. Menor mejor.
Acierto  % de veces que el resultado mas probable fue el que paso.

Todas se comparan contra dos lineas base:
  * azar    -> 1/3, 1/3, 1/3
  * mercado -> probabilidades implicitas en las cuotas de cierre
El mercado es un rival durisimo. Quedarse cerca ya es un buen resultado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from futbol.fuentes.base import Partido
from futbol.modelo.dixon_coles import (
    REGULARIZACION_POR_DEFECTO,
    XI_POR_DEFECTO,
    DixonColes,
)
from futbol.modelo.mercado import (
    a_one_hot,
    brier,
    kelly,
    log_loss,
    probabilidades_implicitas,
    rps,
    valor_esperado,
)


@dataclass
class ResultadoBacktest:
    n_partidos: int
    rps_modelo: float
    brier_modelo: float
    logloss_modelo: float
    acierto_modelo: float
    rps_mercado: float | None
    brier_mercado: float | None
    logloss_mercado: float | None
    acierto_mercado: float | None
    n_con_cuotas: int
    rps_azar: float = 0.25
    brier_azar: float = 0.6667
    logloss_azar: float = 1.0986
    apuestas: dict = field(default_factory=dict)
    calibracion: list[dict] = field(default_factory=list)

    def texto(self) -> str:
        lineas = [
            f"Partidos evaluados: {self.n_partidos}"
            f"  ({self.n_con_cuotas} con cuotas de mercado)",
            "",
            f"{'metrica':<10}{'modelo':>11}{'mercado':>11}{'azar':>11}",
            "-" * 43,
        ]

        def fila(nombre, mod, mer, azar):
            m = f"{mod:>11.4f}"
            k = f"{mer:>11.4f}" if mer is not None else f"{'-':>11}"
            a = f"{azar:>11.4f}"
            return f"{nombre:<10}{m}{k}{a}"

        lineas.append(fila("RPS", self.rps_modelo, self.rps_mercado, self.rps_azar))
        lineas.append(fila("Brier", self.brier_modelo, self.brier_mercado, self.brier_azar))
        lineas.append(fila("LogLoss", self.logloss_modelo, self.logloss_mercado, self.logloss_azar))
        lineas.append(
            f"{'Acierto':<10}{self.acierto_modelo:>10.1%}"
            + (f"{self.acierto_mercado:>11.1%}" if self.acierto_mercado is not None
               else f"{'-':>11}")
            + f"{0.3333:>11.1%}"
        )

        if self.rps_mercado is not None:
            dif = self.rps_modelo - self.rps_mercado
            if dif < -0.001:
                juicio = "el modelo le gana al mercado (raro: revisa que no haya fugas de datos)"
            elif dif < 0.005:
                juicio = "el modelo esta a la altura del mercado"
            elif dif < 0.015:
                juicio = "el modelo esta algo por detras del mercado, pero es util"
            else:
                juicio = "el mercado es claramente mejor que el modelo"
            lineas += ["", f"RPS modelo - RPS mercado = {dif:+.4f}  ->  {juicio}"]

        if self.calibracion:
            lineas += ["", "Calibracion (de todas las probabilidades predichas):",
                       f"  {'rango':<14}{'n':>6}{'predicho':>11}{'observado':>11}"]
            for tramo in self.calibracion:
                lineas.append(
                    f"  {tramo['rango']:<14}{tramo['n']:>6}"
                    f"{tramo['predicho']:>10.1%}{tramo['observado']:>11.1%}"
                )
            lineas.append("  (bien calibrado = las dos ultimas columnas se parecen)")

        if self.apuestas:
            a = self.apuestas
            lineas += [
                "",
                f"Simulacion de apuestas de valor (EV > {a['umbral_ev']:+.0%}, "
                f"cuarto de Kelly):",
                f"  apuestas encontradas : {a['n']} de {self.n_con_cuotas} partidos",
                f"  aciertos             : {a['aciertos']} ({a['tasa_acierto']:.1%})",
                f"  unidades apostadas   : {a['apostado']:.2f}",
                f"  resultado            : {a['beneficio']:+.2f} unidades "
                f"(ROI {a['roi']:+.1%})",
                f"  bankroll compuesto   : 1.000 -> {a['bankroll']:.3f}",
                "  Ojo: un ROI positivo en una muestra corta no demuestra nada.",
                "  Hacen falta cientos de apuestas para distinguir habilidad de suerte.",
            ]

        return "\n".join(lineas)


def _calibracion(probs: np.ndarray, reales: np.ndarray, n_tramos: int = 5) -> list[dict]:
    """Agrupa todas las probabilidades predichas y compara con la frecuencia real."""
    p = probs.ravel()
    r = reales.ravel()
    bordes = np.linspace(0.0, 1.0, n_tramos + 1)
    salida = []
    for i in range(n_tramos):
        lo, hi = bordes[i], bordes[i + 1]
        mask = (p >= lo) & (p < hi if i < n_tramos - 1 else p <= hi)
        if mask.sum() < 5:
            continue
        salida.append({
            "rango": f"{lo:.0%}-{hi:.0%}",
            "n": int(mask.sum()),
            "predicho": float(p[mask].mean()),
            "observado": float(r[mask].mean()),
        })
    return salida


def ejecutar(partidos: list[Partido], xi: float = XI_POR_DEFECTO,
             regularizacion: float = REGULARIZACION_POR_DEFECTO,
             dias_minimos: int = 400,
             refit_cada_dias: int = 7, umbral_ev: float = 0.05,
             verboso: bool = True) -> ResultadoBacktest:
    """Backtest walk-forward.

    dias_minimos     dias de historico que se reservan para entrenar antes de
                     empezar a evaluar.
    refit_cada_dias  cada cuantos dias se reajusta el modelo. Reajustar todos
                     los dias es lo mas correcto pero tarda mas; con 7 dias el
                     resultado apenas cambia.
    """
    jugados = sorted([p for p in partidos if p.jugado], key=lambda p: p.fecha)
    if not jugados:
        raise ValueError("No hay partidos jugados para el backtest.")

    inicio_eval = jugados[0].fecha + timedelta(days=dias_minimos)
    a_evaluar = [p for p in jugados if p.fecha >= inicio_eval]
    if len(a_evaluar) < 30:
        raise ValueError(
            f"Solo quedan {len(a_evaluar)} partidos para evaluar. "
            f"Carga mas temporadas (--temporadas) o baja --dias-minimos."
        )

    probs_modelo: list[np.ndarray] = []
    probs_mercado: list[np.ndarray] = []
    reales: list[str] = []
    reales_mercado: list[str] = []
    apuestas: list[dict] = []

    modelo: DixonColes | None = None
    ultimo_ajuste: date | None = None
    sin_historico = 0

    for i, partido in enumerate(a_evaluar):
        necesita_ajuste = (
            modelo is None
            or ultimo_ajuste is None
            or (partido.fecha - ultimo_ajuste).days >= refit_cada_dias
        )
        if necesita_ajuste:
            historico = [p for p in jugados if p.fecha < partido.fecha]
            try:
                modelo = DixonColes(xi=xi, regularizacion=regularizacion).ajustar(
                    historico, referencia=partido.fecha
                )
                ultimo_ajuste = partido.fecha
            except ValueError:
                continue

        if modelo is None:
            continue

        try:
            pred = modelo.predecir(partido.local, partido.visitante,
                                   liga=partido.liga, estricto=True)
        except KeyError:
            # Equipo recien ascendido: sin historico previo no se puede predecir.
            sin_historico += 1
            continue

        probs_modelo.append(pred.probs)
        reales.append(partido.resultado)

        if partido.tiene_cuotas:
            cuotas = (partido.cuota_local, partido.cuota_empate, partido.cuota_visitante)
            p_mercado = probabilidades_implicitas(cuotas, metodo="potencia")
            probs_mercado.append(p_mercado)
            reales_mercado.append(partido.resultado)

            # Buscar valor: la mejor de las tres opciones si supera el umbral.
            evs = [valor_esperado(pred.probs[k], cuotas[k]) for k in range(3)]
            mejor = int(np.argmax(evs))
            if evs[mejor] >= umbral_ev:
                stake = kelly(pred.probs[mejor], cuotas[mejor], fraccion=0.25)
                if stake > 0:
                    gano = ["H", "D", "A"][mejor] == partido.resultado
                    apuestas.append({
                        "stake": stake,
                        "cuota": cuotas[mejor],
                        "gano": gano,
                        "beneficio": stake * (cuotas[mejor] - 1) if gano else -stake,
                    })

        if verboso and (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{len(a_evaluar)} partidos evaluados")

    if not probs_modelo:
        raise ValueError("No se pudo evaluar ningun partido.")

    if verboso and sin_historico:
        print(f"  ({sin_historico} partidos omitidos: algun equipo aun no tenia historico)")

    P = np.array(probs_modelo)
    R = a_one_hot(reales)
    acierto = float((P.argmax(axis=1) == R.argmax(axis=1)).mean())

    if probs_mercado:
        M = np.array(probs_mercado)
        RM = a_one_hot(reales_mercado)
        met_mercado = (
            rps(M, RM), brier(M, RM), log_loss(M, RM),
            float((M.argmax(axis=1) == RM.argmax(axis=1)).mean()),
        )
    else:
        met_mercado = (None, None, None, None)

    resumen_apuestas: dict = {}
    if apuestas:
        apostado = sum(a["stake"] for a in apuestas)
        beneficio = sum(a["beneficio"] for a in apuestas)
        aciertos = sum(1 for a in apuestas if a["gano"])
        # Bankroll compuesto: cada stake es una fraccion del capital del momento.
        bankroll = 1.0
        for a in apuestas:
            monto = bankroll * a["stake"]
            bankroll += monto * (a["cuota"] - 1) if a["gano"] else -monto
        resumen_apuestas = {
            "n": len(apuestas),
            "aciertos": aciertos,
            "tasa_acierto": aciertos / len(apuestas),
            "apostado": apostado,
            "beneficio": beneficio,
            "roi": beneficio / apostado if apostado else 0.0,
            "bankroll": bankroll,
            "umbral_ev": umbral_ev,
        }

    # Linea base "azar": 1/3 en cada resultado, sobre los mismos partidos.
    azar = np.full_like(P, 1 / 3)

    return ResultadoBacktest(
        n_partidos=len(probs_modelo),
        rps_modelo=rps(P, R),
        brier_modelo=brier(P, R),
        logloss_modelo=log_loss(P, R),
        acierto_modelo=acierto,
        rps_mercado=met_mercado[0],
        brier_mercado=met_mercado[1],
        logloss_mercado=met_mercado[2],
        acierto_mercado=met_mercado[3],
        n_con_cuotas=len(probs_mercado),
        rps_azar=rps(azar, R),
        brier_azar=brier(azar, R),
        logloss_azar=log_loss(azar, R),
        apuestas=resumen_apuestas,
        calibracion=_calibracion(P, R),
    )


def optimizar_xi(partidos: list[Partido],
                 valores: list[float] | None = None,
                 dias_minimos: int = 400,
                 refit_cada_dias: int = 14,
                 verboso: bool = True) -> tuple[float, list[tuple[float, float]]]:
    """Busca el mejor factor de olvido temporal minimizando el RPS del backtest.

    Es el unico hiperparametro que de verdad mueve la aguja: controla cuanto
    pesa el pasado lejano frente a la forma reciente.
    """
    valores = valores or [0.0, 0.0008, 0.0015, 0.0018, 0.0022, 0.0030, 0.0045, 0.0065]
    resultados: list[tuple[float, float]] = []

    for xi in valores:
        try:
            res = ejecutar(partidos, xi=xi, dias_minimos=dias_minimos,
                           refit_cada_dias=refit_cada_dias, verboso=False)
        except ValueError as exc:
            if verboso:
                print(f"  xi={xi:.4f}  fallo: {exc}")
            continue
        resultados.append((xi, res.rps_modelo))
        if verboso:
            semivida = np.log(2) / xi if xi > 0 else float("inf")
            etiqueta = f"{semivida:.0f} dias" if xi > 0 else "sin olvido"
            print(f"  xi={xi:.4f}  (semivida {etiqueta:>11})  RPS {res.rps_modelo:.5f}")

    if not resultados:
        raise ValueError("Ninguna configuracion de xi pudo evaluarse.")

    mejor = min(resultados, key=lambda r: r[1])[0]
    return mejor, resultados
