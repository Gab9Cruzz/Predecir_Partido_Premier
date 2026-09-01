"""Deteccion de equipos recien ascendidos.

"Recien ascendido" = un equipo que jugo la liga inferior (E1) la
temporada anterior y NO jugo la liga superior (E0) esa temporada anterior,
pero si juega E0 esta temporada -- un ascenso real, detectado directamente
en los datos, cada vez que ocurre (no solo la primera vez que un club pisa
la liga superior en todo el historico: un club que sube y baja varias veces
sufre el mismo "shock" de nivel cada vez que vuelve a subir).

La deteccion solo usa en que temporada/liga aparecio cada equipo
(informacion de calendario, publica de antemano) -- nunca resultados de
partidos futuros, asi que es segura de usar dentro de un backtest
walk-forward sin fuga de datos.

Limitacion explicita: si no hay datos de la liga inferior para la temporada
anterior (la primera temporada del dataset, 2000/2001, no tiene 1999/2000
antes), no se puede verificar el ascenso -- se devuelve vacio en vez de
adivinar.
"""

from __future__ import annotations

from futbol.fuentes.base import Partido


def _temporada_anterior(temporada: str) -> str:
    """'2005/2006' -> '2004/2005'."""
    inicio, fin = temporada.split("/")
    return f"{int(inicio) - 1}/{int(fin) - 1}"


def equipos_por_temporada(partidos: list[Partido], liga: str) -> dict[str, set[str]]:
    """temporada -> conjunto de equipos que juegan `liga` esa temporada.

    No exige que el partido este jugado -- pertenecer al fixture de una
    liga esa temporada es un hecho de calendario, conocido de antemano
    (decidido por el ascenso/descenso de la temporada anterior), no algo
    que dependa de resultados. Filtrar solo por jugados dejaria a un equipo
    recien ascendido sin detectar hasta que jugara su primer partido -- justo
    cuando mas hace falta la correccion (predicciones en vivo al arranque
    de temporada, no solo backtest sobre historico ya completo).
    """
    salida: dict[str, set[str]] = {}
    for p in partidos:
        if p.liga != liga:
            continue
        salida.setdefault(p.temporada, set()).add(p.local)
        salida[p.temporada].add(p.visitante)
    return salida


def equipos_ascendidos_en_temporada(temporada: str, equipos_e0: dict[str, set[str]],
                                    equipos_e1: dict[str, set[str]]) -> set[str]:
    """Equipos que juegan la liga superior en `temporada` y jugaron la
    inferior la temporada anterior sin haber jugado la superior esa
    temporada anterior -- ascenso real detectado en los datos."""
    anterior = _temporada_anterior(temporada)
    e1_anterior = equipos_e1.get(anterior, set())
    e0_anterior = equipos_e0.get(anterior, set())
    e0_actual = equipos_e0.get(temporada, set())
    return (e0_actual - e0_anterior) & e1_anterior


def mapa_ascendidos(partidos: list[Partido], liga: str = "E0",
                    liga_inferior: str = "E1") -> dict[str, set[str]]:
    """temporada -> equipos ascendidos esa temporada, para todas las
    temporadas presentes en `liga`."""
    equipos_e0 = equipos_por_temporada(partidos, liga)
    equipos_e1 = equipos_por_temporada(partidos, liga_inferior)
    return {
        temporada: equipos_ascendidos_en_temporada(temporada, equipos_e0, equipos_e1)
        for temporada in equipos_e0
    }


def es_ascendido(equipo: str, temporada: str, mapa: dict[str, set[str]]) -> bool:
    return equipo in mapa.get(temporada, set())
