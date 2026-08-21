#!/usr/bin/env python
"""Predictor de partidos de futbol.

Uso rapido:
    python predecir.py ligas
    python predecir.py partido --liga SP1 --local Barcelona --visitante "Real Madrid"
    python predecir.py proximos --liga SP1
    python predecir.py ratings --liga SP1
    python predecir.py backtest --liga SP1
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from futbol.backtest import ejecutar as ejecutar_backtest, optimizar_xi
from futbol.fuentes import footballdata_uk as fuente
from futbol.modelo.dixon_coles import (
    REGULARIZACION_POR_DEFECTO,
    XI_POR_DEFECTO,
    DixonColes,
    Prediccion,
)
from futbol.modelo.mercado import (
    kelly,
    margen_casa,
    probabilidades_implicitas,
    valor_esperado,
)

BARRA = "=" * 68


# --------------------------------------------------------------------- helpers

def _cargar_modelo(args) -> tuple[DixonColes, list]:
    codigos = [c.strip().upper() for c in args.liga.split(",") if c.strip()]
    if len(codigos) > 1:
        print("Cargando datos...")
    partidos = fuente.cargar_varias(
        codigos, temporadas=args.temporadas, forzar=args.actualizar,
        verboso=len(codigos) > 1,
    )
    if not partidos:
        print(f"No hay datos para {args.liga}. Prueba con --temporadas mayor.")
        sys.exit(1)

    modelo = DixonColes(xi=args.xi, regularizacion=args.regularizacion)
    modelo.ajustar(partidos)
    return modelo, partidos


def _mostrar_prediccion(pred: Prediccion, cuotas: tuple | None = None,
                        umbral_ev: float = 0.03) -> None:
    print()
    print(BARRA)
    print(f"  {pred.local}  vs  {pred.visitante}")
    print(BARRA)
    for pedido, resuelto, confianza in pred.emparejamientos:
        print(f"  nota: '{pedido}' se interpreto como '{resuelto}' "
              f"(confianza {confianza:.0%})")
    print(f"\nGoles esperados : {pred.goles_esperados_local:.2f} - "
          f"{pred.goles_esperados_visitante:.2f}")
    print(f"Marcador top    : {pred.marcadores_probables[0][0]} "
          f"({pred.marcadores_probables[0][1]:.1%})")

    print("\nResultado                probabilidad   cuota justa")
    print("-" * 52)
    etiquetas = [
        (f"Gana {pred.local}", pred.prob_local),
        ("Empate", pred.prob_empate),
        (f"Gana {pred.visitante}", pred.prob_visitante),
    ]
    justas = pred.cuotas_justas
    for (texto, prob), justa in zip(etiquetas, justas):
        barra = "#" * int(round(prob * 34))
        print(f"{texto[:22]:<22} {prob:>7.1%}   {justa:>7.2f}  {barra}")

    print("\nOtros mercados")
    print("-" * 52)
    print(f"Doble oportunidad 1X   {pred.prob_local_o_empate:>7.1%}"
          f"     X2   {pred.prob_visitante_o_empate:>7.1%}")
    print(f"Mas de 1.5 goles       {pred.prob_mas_de(1.5):>7.1%}"
          f"     -1.5 {1 - pred.prob_mas_de(1.5):>7.1%}")
    print(f"Mas de 2.5 goles       {pred.prob_mas_de(2.5):>7.1%}"
          f"     -2.5 {1 - pred.prob_mas_de(2.5):>7.1%}")
    print(f"Mas de 3.5 goles       {pred.prob_mas_de(3.5):>7.1%}")
    print(f"Ambos marcan           {pred.prob_ambos_marcan:>7.1%}")
    print(f"Local -1.5 handicap    {pred.prob_handicap_asiatico(-1.5):>7.1%}")

    print("\nMarcadores mas probables")
    print("-" * 52)
    for i in range(0, 8, 2):
        izq = pred.marcadores_probables[i]
        der = pred.marcadores_probables[i + 1]
        print(f"  {izq[0]:<6} {izq[1]:>6.1%}        {der[0]:<6} {der[1]:>6.1%}")

    muestra_baja = min(pred.partidos_local, pred.partidos_visitante) < 10
    print(f"\nPartidos en la muestra: {pred.local} {pred.partidos_local}, "
          f"{pred.visitante} {pred.partidos_visitante}")
    if muestra_baja:
        print("  AVISO: uno de los dos equipos tiene muy pocos partidos en los datos.")
        print("  Su rating esta muy encogido hacia la media y la prediccion es floja.")

    if cuotas:
        _comparar_con_mercado(pred, cuotas, umbral_ev)


def _comparar_con_mercado(pred: Prediccion, cuotas: tuple,
                          umbral_ev: float = 0.03) -> None:
    p_mercado = probabilidades_implicitas(cuotas, metodo="potencia")
    print("\nComparacion con el mercado"
          f"   (margen de la casa: {margen_casa(cuotas):.1%})")
    print("-" * 52)
    print(f"{'':<10}{'modelo':>9}{'mercado':>10}{'cuota':>9}{'EV':>9}{'Kelly':>8}")
    hay_valor = False
    for k, nombre in enumerate(["Local", "Empate", "Visita"]):
        ev = valor_esperado(pred.probs[k], cuotas[k])
        f = kelly(pred.probs[k], cuotas[k], fraccion=0.25)
        marca = "  <-- valor" if ev >= umbral_ev else ""
        hay_valor = hay_valor or ev >= umbral_ev
        print(f"{nombre:<10}{pred.probs[k]:>8.1%}{p_mercado[k]:>10.1%}"
              f"{cuotas[k]:>9.2f}{ev:>+9.1%}{f:>8.1%}{marca}")
    if hay_valor:
        print("\n  'Valor' solo significa que el modelo discrepa del mercado.")
        print("  El mercado acierta mas que este modelo (mira `backtest`),")
        print("  asi que lo mas probable es que el equivocado sea el modelo.")


# -------------------------------------------------------------------- comandos

def cmd_ligas(args) -> None:
    print("\nLigas disponibles (football-data.co.uk, gratis y sin API key)\n")
    print(f"  {'cod':<5}{'pais':<12}{'liga':<24}{'datos'}")
    print("  " + "-" * 62)
    for liga in fuente.LIGAS.values():
        extras = ("historico completo + cuotas" if liga.formato == "extra"
                  else "cuotas + tiros, corners y tarjetas")
        print(f"  {liga.codigo:<5}{liga.pais:<12}{liga.nombre:<24}{extras}")
    print("\n  Puedes combinar varias con coma: --liga E0,E1")
    print("  Los datos llegan hasta hace uno o dos dias.\n")


def cmd_equipos(args) -> None:
    partidos = fuente.cargar_varias(
        [c.strip().upper() for c in args.liga.split(",")],
        temporadas=args.temporadas, forzar=args.actualizar, verboso=False,
    )
    equipos = fuente.equipos_de(partidos)
    print(f"\n{len(equipos)} equipos en {args.liga.upper()} "
          f"(ultimas {args.temporadas} temporadas):\n")
    for i in range(0, len(equipos), 3):
        print("   " + "".join(f"{e:<26}" for e in equipos[i:i + 3]))
    print()


def cmd_partido(args) -> None:
    modelo, _ = _cargar_modelo(args)
    cuotas = tuple(args.cuotas) if args.cuotas else None
    codigos = [c.strip().upper() for c in args.liga.split(",")]
    pred = modelo.predecir(
        args.local, args.visitante,
        liga=codigos[0] if len(codigos) == 1 else None,
        campo_neutral=args.neutral,
    )
    if args.json:
        print(json.dumps({
            "local": pred.local,
            "visitante": pred.visitante,
            "goles_esperados": [pred.goles_esperados_local,
                                pred.goles_esperados_visitante],
            "prob_local": pred.prob_local,
            "prob_empate": pred.prob_empate,
            "prob_visitante": pred.prob_visitante,
            "mas_de_2_5": pred.prob_mas_de(2.5),
            "ambos_marcan": pred.prob_ambos_marcan,
            "marcadores": pred.marcadores_probables[:5],
        }, indent=2, ensure_ascii=False))
        return
    print(f"\n{modelo.resumen()}")
    _mostrar_prediccion(pred, cuotas)
    print()


def cmd_ratings(args) -> None:
    modelo, _ = _cargar_modelo(args)
    print(f"\n{modelo.resumen()}\n")
    print(f"  {'#':>3} {'equipo':<24}{'ataque':>9}{'defensa':>9}"
          f"{'general':>9}{'pj':>6}")
    print("  " + "-" * 60)
    for fila in modelo.ratings():
        aviso = " *" if fila["partidos"] < 10 else ""
        print(f"  {fila['puesto']:>3} {fila['equipo']:<24}"
              f"{fila['ataque']:>+9.2f}{fila['defensa']:>+9.2f}"
              f"{fila['general']:>+9.2f}{fila['partidos']:>6}{aviso}")
    print("\n  ataque  = capacidad de marcar (log-goles sobre la media de la liga)")
    print("  defensa = capacidad de evitar goles; mas alto es mejor defensa")
    print("  general = ataque + defensa. Diferencia de +0.7 ~ el doble de goles")
    print("  * = pocos partidos, rating encogido hacia la media\n")


def cmd_proximos(args) -> None:
    codigos = [c.strip().upper() for c in args.liga.split(",")] if args.liga else None
    proximos = fuente.cargar_proximos(codigos)
    if not proximos:
        print("\nNo hay partidos proximos publicados ahora mismo para esa seleccion.")
        print("El archivo de proximos partidos cubre solo los siguientes dias.\n")
        return

    por_liga: dict[str, list] = {}
    for p in proximos:
        por_liga.setdefault(p.liga, []).append(p)

    print(f"\n{len(proximos)} partidos proximos en {len(por_liga)} liga(s)\n")

    for codigo, lista in sorted(por_liga.items()):
        historico = fuente.cargar(codigo, temporadas=args.temporadas,
                                  forzar=args.actualizar, verboso=False)
        try:
            modelo = DixonColes(xi=args.xi,
                                regularizacion=args.regularizacion).ajustar(historico)
        except ValueError as exc:
            print(f"{fuente.LIGAS[codigo].nombre}: sin datos suficientes ({exc})")
            continue

        print(BARRA)
        print(f"  {fuente.LIGAS[codigo].nombre} ({fuente.LIGAS[codigo].pais})")
        print(BARRA)
        print(f"{'fecha':<12}{'partido':<40}{'1':>7}{'X':>7}{'2':>7}{'  xG':>10}")
        print("-" * 84)

        con_valor = []
        sin_historico: list[str] = []
        hay_poca_muestra = False
        for p in lista:
            try:
                pred = modelo.predecir(p.local, p.visitante, liga=codigo)
            except KeyError:
                sin_historico.append(f"{p.local} vs {p.visitante}")
                continue
            # Uso los nombres ya resueltos: el archivo de proximos partidos
            # escribe algunos equipos distinto que el de resultados.
            enfrentamiento = f"{pred.local} vs {pred.visitante}"
            # Un equipo con pocos partidos tiene el rating muy encogido: sus
            # probabilidades son poco de fiar y generan falsos "valores".
            poca_muestra = min(pred.partidos_local, pred.partidos_visitante) < 10
            marca = " *" if poca_muestra else ""
            print(f"{str(p.fecha):<12}{(enfrentamiento + marca)[:39]:<40}"
                  f"{pred.prob_local:>7.1%}{pred.prob_empate:>7.1%}"
                  f"{pred.prob_visitante:>7.1%}"
                  f"{pred.goles_esperados_local:>6.2f}-"
                  f"{pred.goles_esperados_visitante:.2f}")
            if poca_muestra:
                hay_poca_muestra = True
            if p.tiene_cuotas and not poca_muestra:
                cuotas = (p.cuota_local, p.cuota_empate, p.cuota_visitante)
                evs = [valor_esperado(pred.probs[k], cuotas[k]) for k in range(3)]
                mejor = int(np.argmax(evs))
                if evs[mejor] >= 0.05:
                    con_valor.append((enfrentamiento, ["1", "X", "2"][mejor],
                                      cuotas[mejor], evs[mejor], pred.probs[mejor]))

        if con_valor:
            print(f"\n  Donde mas discrepa el modelo del mercado:")
            for enf, signo, cuota, ev, prob in sorted(con_valor, key=lambda x: -x[3])[:5]:
                print(f"    {enf[:38]:<40} {signo}  cuota {cuota:.2f}  "
                      f"modelo {prob:.1%}  EV {ev:+.1%}")
        print()

    print("Recuerda: en el backtest el mercado acierta mas que el modelo.")
    print("Estas discrepancias son puntos a revisar, no recomendaciones.\n")


def cmd_backtest(args) -> None:
    codigos = [c.strip().upper() for c in args.liga.split(",")]
    partidos = fuente.cargar_varias(codigos, temporadas=args.temporadas,
                                    forzar=args.actualizar, verboso=True)
    print(f"\nBacktest walk-forward sobre {len(partidos)} partidos "
          f"({partidos[0].fecha} -> {partidos[-1].fecha})")
    print(f"xi={args.xi}  regularizacion={args.regularizacion}  "
          f"reajuste cada {args.refit} dias\n")
    resultado = ejecutar_backtest(
        partidos, xi=args.xi, regularizacion=args.regularizacion,
        dias_minimos=args.dias_minimos, refit_cada_dias=args.refit,
        umbral_ev=args.umbral_ev,
    )
    print()
    print(resultado.texto())
    print()


def cmd_optimizar(args) -> None:
    codigos = [c.strip().upper() for c in args.liga.split(",")]
    partidos = fuente.cargar_varias(codigos, temporadas=args.temporadas,
                                    forzar=args.actualizar, verboso=True)
    print(f"\nBuscando el mejor factor de olvido temporal para {args.liga.upper()}")
    print("(cada linea es un backtest completo; tarda un rato)\n")
    mejor, _ = optimizar_xi(partidos, dias_minimos=args.dias_minimos,
                            refit_cada_dias=max(args.refit, 14))
    semivida = np.log(2) / mejor if mejor > 0 else float("inf")
    print(f"\nMejor xi para esta liga: {mejor}"
          f"  (semivida {semivida:.0f} dias)" if mejor > 0
          else f"\nMejor xi para esta liga: {mejor} (sin olvido temporal)")
    print(f"Usalo asi:  python predecir.py partido --liga {args.liga} "
          f"--xi {mejor} --local ... --visitante ...\n")


# ----------------------------------------------------------------------- main

def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="predecir.py",
        description="Predice resultados de futbol con un modelo Dixon-Coles "
                    "sobre datos gratuitos y actualizados.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="comando", required=True)

    def comunes(sp, con_liga=True, liga_obligatoria=True):
        if con_liga:
            sp.add_argument("--liga", required=liga_obligatoria,
                            help="codigo de liga, o varios separados por coma")
        sp.add_argument("--temporadas", type=int, default=4,
                        help="temporadas de historico a usar (por defecto 4)")
        sp.add_argument("--xi", type=float, default=XI_POR_DEFECTO,
                        help=f"factor de olvido temporal (por defecto {XI_POR_DEFECTO})")
        sp.add_argument("--regularizacion", type=float,
                        default=REGULARIZACION_POR_DEFECTO,
                        help="fuerza del encogimiento hacia la media")
        sp.add_argument("--actualizar", action="store_true",
                        help="ignora la cache y vuelve a descargar los CSV")
        return sp

    sp = sub.add_parser("ligas", help="lista las ligas disponibles")
    sp.set_defaults(func=cmd_ligas)

    sp = comunes(sub.add_parser("equipos", help="lista los equipos de una liga"))
    sp.set_defaults(func=cmd_equipos)

    sp = comunes(sub.add_parser("partido", help="predice un partido concreto"))
    sp.add_argument("--local", required=True, help="equipo local")
    sp.add_argument("--visitante", required=True, help="equipo visitante")
    sp.add_argument("--neutral", action="store_true",
                    help="campo neutral: reparte la ventaja de local")
    sp.add_argument("--cuotas", type=float, nargs=3, metavar=("1", "X", "2"),
                    help="cuotas decimales del mercado para comparar")
    sp.add_argument("--json", action="store_true", help="salida en JSON")
    sp.set_defaults(func=cmd_partido)

    sp = comunes(sub.add_parser("ratings", help="tabla de fuerza de los equipos"))
    sp.set_defaults(func=cmd_ratings)

    sp = comunes(sub.add_parser("proximos", help="predice los partidos de los proximos dias"),
                 liga_obligatoria=False)
    sp.set_defaults(func=cmd_proximos)

    sp = comunes(sub.add_parser("backtest", help="valida el modelo contra el historico"))
    sp.add_argument("--dias-minimos", type=int, default=400, dest="dias_minimos",
                    help="dias de historico reservados para entrenar")
    sp.add_argument("--refit", type=int, default=7,
                    help="cada cuantos dias se reajusta el modelo")
    sp.add_argument("--umbral-ev", type=float, default=0.05, dest="umbral_ev",
                    help="EV minimo para simular una apuesta")
    sp.set_defaults(func=cmd_backtest, temporadas=6)

    sp = comunes(sub.add_parser("optimizar", help="busca el mejor xi para una liga"))
    sp.add_argument("--dias-minimos", type=int, default=400, dest="dias_minimos")
    sp.add_argument("--refit", type=int, default=14)
    sp.set_defaults(func=cmd_optimizar, temporadas=6)

    return p


def main() -> None:
    args = construir_parser().parse_args()
    try:
        args.func(args)
    except KeyError as exc:
        print(f"\n{exc.args[0]}\n")
        sys.exit(1)
    except (ValueError, RuntimeError) as exc:
        print(f"\nError: {exc}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nCancelado.")
        sys.exit(130)


if __name__ == "__main__":
    main()
