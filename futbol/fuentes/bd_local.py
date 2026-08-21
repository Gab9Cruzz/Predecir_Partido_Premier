"""Fuente de datos: base SQLite local (BD_SQLITE/futbol_predicciones.db).

Unica fuente de datos del proyecto -- no queda ningun camino que descargue
CSV en vivo. `BD_SQLITE/cargar_datos.py` y `actualizar_resultados.py` son
los que mantienen la base al dia (por fuera de este modulo); este archivo
solo lee.

Carga el historico de Premier League (E0) + Championship (E1). Se cargan
siempre juntos: sin el historico de Championship, un equipo recien
ascendido llega cada temporada con cero partidos previos y el modelo
pierde la conectividad que necesita para estimarlo. El modelo Dixon-Coles
ya separa el nivel de cada division solo con el campo `liga` (un mu por
liga) -- no hace falta logica nueva para eso.

Deliberadamente fuera de esta primera fase: mercados de cuotas distintos de
1x2 (over/under, handicap asiatico) y las estadisticas extendidas (tiros,
corners, tarjetas) -- `Partido` no tiene campos para eso todavia.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from futbol.fuentes.base import Partido

DB_PATH = Path(__file__).resolve().parents[2] / "BD_SQLITE" / "futbol_predicciones.db"

# Premier League + Championship. Ver docstring del modulo: E1 no es opcional,
# es lo que le da conectividad al grafo de partidos para los ascendidos.
LIGAS_CARGADAS = ("E0", "E1")

# Nombres legibles. Antes se sacaban de footballdata_uk.LIGAS, pero ese
# modulo (el pipeline CSV en vivo) ya no lo usa nada del proyecto -- se
# hardcodean aca las dos que hacen falta en vez de mantener esa dependencia
# solo por dos strings.
_NOMBRES_LIGA = {
    "E0": "Premier League (Inglaterra)",
    "E1": "Championship (Inglaterra)",
}


def _liga_nombre(codigo: str) -> str:
    return _NOMBRES_LIGA.get(codigo, codigo)


def _conectar(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(
            f"No encuentro la base en {db_path}. Corre "
            f"'python BD_SQLITE/init_db.py' y 'python BD_SQLITE/cargar_datos.py' "
            f"primero para crearla y poblarla."
        )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _fila_a_partido(fila: sqlite3.Row) -> Partido:
    return Partido(
        fecha=datetime.strptime(fila["fecha"], "%Y-%m-%d").date(),
        liga=fila["liga"],
        liga_nombre=_liga_nombre(fila["liga"]),
        temporada=fila["temporada"],
        local=fila["local"],
        visitante=fila["visitante"],
        goles_local=fila["goles_local"],
        goles_visitante=fila["goles_visitante"],
        cuota_local=fila["cuota_local"],
        cuota_empate=fila["cuota_empate"],
        cuota_visitante=fila["cuota_visitante"],
    )


def cargar_para_ajuste(db_path: Path = DB_PATH) -> list[Partido]:
    """Carga el historico completo -- solo partidos ya jugados -- de E0
    (Premier) + E1 (Championship), listo para DixonColes.ajustar().
    """
    conn = _conectar(db_path)
    try:
        placeholders = ",".join("?" * len(LIGAS_CARGADAS))
        filas = conn.execute(
            f"""
            SELECT
                p.fecha, p.temporada, p.liga,
                el.nombre AS local, ev.nombre AS visitante,
                p.goles_local, p.goles_visitante,
                c.cuota_local, c.cuota_empate, c.cuota_visitante
            FROM partidos p
            JOIN equipos el ON el.id = p.equipo_local_id
            JOIN equipos ev ON ev.id = p.equipo_visitante_id
            LEFT JOIN cuotas_cierre c
                ON c.partido_id = p.id AND c.mercado = '1x2'
            WHERE p.liga IN ({placeholders}) AND p.goles_local IS NOT NULL
            ORDER BY p.fecha
            """,
            LIGAS_CARGADAS,
        ).fetchall()
    finally:
        conn.close()

    return [_fila_a_partido(f) for f in filas]


def cargar_proximos(db_path: Path = DB_PATH) -> list[Partido]:
    """Partidos de Premier League que todavia no se jugaron (goles NULL),
    con las cuotas que tenga cargadas `actualizar_resultados.py`.

    Los trae `BD_SQLITE/actualizar_resultados.py --proximos` (o corriendo
    el script sin flags, que sincroniza las dos cosas). Si nunca se corrio,
    esto devuelve una lista vacia -- no descarga nada por su cuenta.
    """
    conn = _conectar(db_path)
    try:
        filas = conn.execute(
            """
            SELECT
                p.fecha, p.temporada, p.liga,
                el.nombre AS local, ev.nombre AS visitante,
                p.goles_local, p.goles_visitante,
                c.cuota_local, c.cuota_empate, c.cuota_visitante
            FROM partidos p
            JOIN equipos el ON el.id = p.equipo_local_id
            JOIN equipos ev ON ev.id = p.equipo_visitante_id
            LEFT JOIN cuotas_cierre c
                ON c.partido_id = p.id AND c.mercado = '1x2'
            WHERE p.liga = 'E0' AND p.goles_local IS NULL
            ORDER BY p.fecha
            """
        ).fetchall()
    finally:
        conn.close()

    return [_fila_a_partido(f) for f in filas]
