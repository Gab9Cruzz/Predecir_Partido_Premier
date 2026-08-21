"""Fuente principal de datos: football-data.co.uk

Por que esta y no API-Football:
  * Es gratis y NO necesita API key ni registro.
  * Se actualiza cada pocas horas: hay resultados de partidos de anteayer.
  * Trae historico completo (10+ temporadas) para poder entrenar y validar.
  * Incluye cuotas de cierre del mercado, que sirven de linea base honesta
    para saber si el modelo aporta algo o no.

Hay dos formatos de archivo:
  * "main"  -> https://www.football-data.co.uk/mmz4281/<TEMP>/<COD>.csv
               Un archivo por temporada. Ligas europeas. Incluye tiros,
               corners y tarjetas ademas del resultado.
  * "extra" -> https://www.football-data.co.uk/new/<COD>.csv
               Un unico archivo con todo el historico. Resto del mundo.
               Solo resultado y cuotas.
"""

from __future__ import annotations

import csv
import io
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

from futbol.fuentes.base import Partido

BASE = "https://www.football-data.co.uk"
CACHE = Path(__file__).resolve().parents[2] / "datos" / "cache"
CACHE_TTL_HORAS = 6
TIMEOUT = 30


@dataclass(frozen=True, slots=True)
class Liga:
    codigo: str
    nombre: str
    pais: str
    formato: str  # "main" | "extra"


LIGAS: dict[str, Liga] = {
    # --- formato "main": un CSV por temporada, con estadisticas de tiros ---
    "E0":  Liga("E0",  "Premier League",      "Inglaterra", "main"),
    "E1":  Liga("E1",  "Championship",        "Inglaterra", "main"),
    "E2":  Liga("E2",  "League One",          "Inglaterra", "main"),
    "E3":  Liga("E3",  "League Two",          "Inglaterra", "main"),
    "EC":  Liga("EC",  "National League",     "Inglaterra", "main"),
    "SC0": Liga("SC0", "Premiership",         "Escocia",    "main"),
    "SC1": Liga("SC1", "Championship",        "Escocia",    "main"),
    "SC2": Liga("SC2", "League One",          "Escocia",    "main"),
    "SC3": Liga("SC3", "League Two",          "Escocia",    "main"),
    "D1":  Liga("D1",  "Bundesliga",          "Alemania",   "main"),
    "D2":  Liga("D2",  "2. Bundesliga",       "Alemania",   "main"),
    "I1":  Liga("I1",  "Serie A",             "Italia",     "main"),
    "I2":  Liga("I2",  "Serie B",             "Italia",     "main"),
    "SP1": Liga("SP1", "LaLiga",              "Espana",     "main"),
    "SP2": Liga("SP2", "LaLiga 2",            "Espana",     "main"),
    "F1":  Liga("F1",  "Ligue 1",             "Francia",    "main"),
    "F2":  Liga("F2",  "Ligue 2",             "Francia",    "main"),
    "N1":  Liga("N1",  "Eredivisie",          "Holanda",    "main"),
    "B1":  Liga("B1",  "Pro League",          "Belgica",    "main"),
    "P1":  Liga("P1",  "Primeira Liga",       "Portugal",   "main"),
    "T1":  Liga("T1",  "Super Lig",           "Turquia",    "main"),
    "G1":  Liga("G1",  "Super League",        "Grecia",     "main"),
    # --- formato "extra": historico completo en un solo CSV ---
    "ARG": Liga("ARG", "Liga Profesional",    "Argentina",  "extra"),
    "BRA": Liga("BRA", "Brasileirao Serie A", "Brasil",     "extra"),
    "MEX": Liga("MEX", "Liga MX",             "Mexico",     "extra"),
    "USA": Liga("USA", "MLS",                 "EEUU",       "extra"),
    "JPN": Liga("JPN", "J1 League",           "Japon",      "extra"),
    "CHN": Liga("CHN", "Super League",        "China",      "extra"),
    "AUT": Liga("AUT", "Bundesliga",          "Austria",    "extra"),
    "DNK": Liga("DNK", "Superliga",           "Dinamarca",  "extra"),
    "FIN": Liga("FIN", "Veikkausliiga",       "Finlandia",  "extra"),
    "IRL": Liga("IRL", "Premier Division",    "Irlanda",    "extra"),
    "NOR": Liga("NOR", "Eliteserien",         "Noruega",    "extra"),
    "POL": Liga("POL", "Ekstraklasa",         "Polonia",    "extra"),
    "ROU": Liga("ROU", "Liga I",              "Rumania",    "extra"),
    "RUS": Liga("RUS", "Premier Liga",        "Rusia",      "extra"),
    "SWE": Liga("SWE", "Allsvenskan",         "Suecia",     "extra"),
    "SWZ": Liga("SWZ", "Super League",        "Suiza",      "extra"),
}


# Nombre de pais tal y como aparece en los CSV "extra" -> codigo de liga.
PAIS_CSV: dict[str, str] = {
    "argentina": "ARG", "brazil": "BRA", "mexico": "MEX", "usa": "USA",
    "japan": "JPN", "china": "CHN", "austria": "AUT", "denmark": "DNK",
    "finland": "FIN", "ireland": "IRL", "norway": "NOR", "poland": "POL",
    "romania": "ROU", "russia": "RUS", "sweden": "SWE", "switzerland": "SWZ",
}


# ---------------------------------------------------------------- temporadas

def temporada_actual(hoy: date | None = None) -> int:
    """Ano en que arranca la temporada europea vigente (2026 => 2026/27)."""
    hoy = hoy or date.today()
    return hoy.year if hoy.month >= 7 else hoy.year - 1


def codigo_temporada(ano_inicio: int) -> str:
    """2025 -> '2526' (formato de carpeta de football-data.co.uk)."""
    return f"{ano_inicio % 100:02d}{(ano_inicio + 1) % 100:02d}"


# ------------------------------------------------------------------- descarga

def _ruta_cache(nombre: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / nombre


def _descargar(url: str, nombre_cache: str, forzar: bool = False) -> str | None:
    """Descarga un CSV con cache en disco. Devuelve None si no existe."""
    destino = _ruta_cache(nombre_cache)

    if destino.exists() and not forzar:
        edad = time.time() - destino.stat().st_mtime
        if edad < CACHE_TTL_HORAS * 3600:
            return destino.read_text(encoding="utf-8-sig", errors="replace")

    try:
        resp = requests.get(url, timeout=TIMEOUT)
    except requests.RequestException as exc:
        # Sin red: si hay cache vieja, mejor usarla que fallar.
        if destino.exists():
            print(f"  aviso: sin conexion ({type(exc).__name__}), uso cache de {nombre_cache}")
            return destino.read_text(encoding="utf-8-sig", errors="replace")
        raise

    # El servidor responde con una pagina HTML de error para temporadas futuras.
    es_html = "<html" in resp.text[:300].lower()
    if resp.status_code != 200 or len(resp.content) < 500 or es_html:
        if destino.exists():
            return destino.read_text(encoding="utf-8-sig", errors="replace")
        return None

    texto = resp.content.decode("utf-8-sig", errors="replace")
    destino.write_text(texto, encoding="utf-8")
    return texto


# --------------------------------------------------------------------- parseo

def _a_fecha(txt: str) -> date | None:
    txt = (txt or "").strip()
    for formato in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(txt, formato).date()
        except ValueError:
            continue
    return None


def _a_int(txt: str) -> int | None:
    try:
        return int(float(txt))
    except (TypeError, ValueError):
        return None


def _a_float(txt: str) -> float | None:
    try:
        valor = float(txt)
    except (TypeError, ValueError):
        return None
    return valor if valor > 1.0 else None


def _primera_cuota(fila: dict, *grupos: tuple[str, str, str]):
    """Toma el primer trio de columnas de cuotas que este completo.

    Prioridad: promedio de cierre (AvgC*) > Pinnacle cierre (PSC*) > Bet365.
    Las cuotas de cierre son las mas informativas: incorporan alineaciones,
    lesiones y todo lo que el mercado sabia justo antes del pitazo inicial.
    """
    for cols in grupos:
        valores = tuple(_a_float(fila.get(c, "")) for c in cols)
        if all(v is not None for v in valores):
            return valores
    return (None, None, None)


def _parsear_main(texto: str, liga: Liga, temporada: str) -> list[Partido]:
    partidos: list[Partido] = []
    for fila in csv.DictReader(io.StringIO(texto)):
        fecha = _a_fecha(fila.get("Date", ""))
        local = (fila.get("HomeTeam") or "").strip()
        visitante = (fila.get("AwayTeam") or "").strip()
        if not fecha or not local or not visitante:
            continue

        cuotas = _primera_cuota(
            fila,
            ("AvgCH", "AvgCD", "AvgCA"),
            ("PSCH", "PSCD", "PSCA"),
            ("AvgH", "AvgD", "AvgA"),
            ("B365H", "B365D", "B365A"),
        )
        partidos.append(
            Partido(
                fecha=fecha,
                liga=liga.codigo,
                liga_nombre=f"{liga.nombre} ({liga.pais})",
                temporada=temporada,
                local=local,
                visitante=visitante,
                goles_local=_a_int(fila.get("FTHG", "")),
                goles_visitante=_a_int(fila.get("FTAG", "")),
                cuota_local=cuotas[0],
                cuota_empate=cuotas[1],
                cuota_visitante=cuotas[2],
            )
        )
    return partidos


def _parsear_extra(texto: str, liga: Liga, desde: date) -> list[Partido]:
    partidos: list[Partido] = []
    for fila in csv.DictReader(io.StringIO(texto)):
        fecha = _a_fecha(fila.get("Date", ""))
        local = (fila.get("Home") or "").strip()
        visitante = (fila.get("Away") or "").strip()
        if not fecha or fecha < desde or not local or not visitante:
            continue

        cuotas = _primera_cuota(
            fila,
            ("AvgCH", "AvgCD", "AvgCA"),
            ("PSCH", "PSCD", "PSCA"),
            ("B365CH", "B365CD", "B365CA"),
        )
        partidos.append(
            Partido(
                fecha=fecha,
                liga=liga.codigo,
                liga_nombre=f"{liga.nombre} ({liga.pais})",
                temporada=(fila.get("Season") or "").strip(),
                local=local,
                visitante=visitante,
                goles_local=_a_int(fila.get("HG", "")),
                goles_visitante=_a_int(fila.get("AG", "")),
                cuota_local=cuotas[0],
                cuota_empate=cuotas[1],
                cuota_visitante=cuotas[2],
            )
        )
    return partidos


# ------------------------------------------------------------------- carga

def cargar(codigo: str, temporadas: int = 4, forzar: bool = False,
           verboso: bool = True) -> list[Partido]:
    """Carga los partidos jugados de una liga en las ultimas N temporadas."""
    codigo = codigo.upper()
    if codigo not in LIGAS:
        raise ValueError(
            f"Liga '{codigo}' desconocida. "
            f"Corre `python predecir.py ligas` para ver las opciones."
        )
    liga = LIGAS[codigo]
    partidos: list[Partido] = []

    if liga.formato == "main":
        ano_actual = temporada_actual()
        for ano in range(ano_actual - temporadas + 1, ano_actual + 1):
            cod_temp = codigo_temporada(ano)
            url = f"{BASE}/mmz4281/{cod_temp}/{liga.codigo}.csv"
            texto = _descargar(url, f"{liga.codigo}_{cod_temp}.csv", forzar)
            if texto is None:
                if verboso and ano == ano_actual:
                    print(f"  aviso: {liga.nombre} {ano}/{ano + 1} aun no publicada")
                continue
            partidos.extend(_parsear_main(texto, liga, f"{ano}/{ano + 1}"))
    else:
        desde = date.today() - timedelta(days=int(365.25 * temporadas))
        url = f"{BASE}/new/{liga.codigo}.csv"
        texto = _descargar(url, f"{liga.codigo}_new.csv", forzar)
        if texto is not None:
            partidos.extend(_parsear_extra(texto, liga, desde))

    jugados = [p for p in partidos if p.jugado]
    jugados.sort(key=lambda p: p.fecha)
    return jugados


def cargar_varias(codigos: list[str], temporadas: int = 4, forzar: bool = False,
                  verboso: bool = True) -> list[Partido]:
    """Carga y mezcla varias ligas (para el modo multi-liga)."""
    todos: list[Partido] = []
    for codigo in codigos:
        partidos = cargar(codigo, temporadas, forzar, verboso)
        if verboso:
            liga = LIGAS[codigo.upper()]
            rango = (f"{partidos[0].fecha} -> {partidos[-1].fecha}"
                     if partidos else "sin datos")
            print(f"  {codigo.upper():<4} {liga.nombre:<22} "
                  f"{len(partidos):>5} partidos   {rango}")
        todos.extend(partidos)
    todos.sort(key=lambda p: p.fecha)
    return todos


def equipos_de(partidos: list[Partido]) -> list[str]:
    return sorted({p.local for p in partidos} | {p.visitante for p in partidos})


# ------------------------------------------------- proximos partidos (fixtures)

def cargar_proximos(codigos: list[str] | None = None) -> list[Partido]:
    """Partidos de los proximos dias, con las cuotas que ofrece el mercado.

    football-data.co.uk publica dos archivos de proximos partidos: uno para las
    ligas europeas y otro para el resto. Si `codigos` es None se devuelven todos.
    """
    filtro = {c.upper() for c in codigos} if codigos else None
    proximos: list[Partido] = []

    texto = _descargar(f"{BASE}/fixtures.csv", "fixtures_main.csv")
    if texto:
        for fila in csv.DictReader(io.StringIO(texto)):
            codigo = (fila.get("Div") or "").strip().upper()
            fecha = _a_fecha(fila.get("Date", ""))
            if codigo not in LIGAS or not fecha:
                continue
            if filtro and codigo not in filtro:
                continue
            liga = LIGAS[codigo]
            cuotas = _primera_cuota(
                fila, ("AvgH", "AvgD", "AvgA"), ("MaxH", "MaxD", "MaxA"),
                ("B365H", "B365D", "B365A"),
            )
            proximos.append(Partido(
                fecha=fecha, liga=codigo,
                liga_nombre=f"{liga.nombre} ({liga.pais})", temporada="proxima",
                local=(fila.get("HomeTeam") or "").strip(),
                visitante=(fila.get("AwayTeam") or "").strip(),
                cuota_local=cuotas[0], cuota_empate=cuotas[1], cuota_visitante=cuotas[2],
            ))

    # El archivo "extra" identifica la liga por el nombre ingles del pais.
    texto = _descargar(f"{BASE}/new_league_fixtures.csv", "fixtures_extra.csv")
    if texto:
        for fila in csv.DictReader(io.StringIO(texto)):
            pais = (fila.get("Country") or "").strip().lower()
            fecha = _a_fecha(fila.get("Date", ""))
            codigo = PAIS_CSV.get(pais)
            if not fecha or codigo is None:
                continue
            if filtro and codigo not in filtro:
                continue
            liga = LIGAS[codigo]
            cuotas = _primera_cuota(
                fila, ("AvgH", "AvgD", "AvgA"), ("PSH", "PSD", "PSA"),
                ("B365H", "B365D", "B365A"),
            )
            proximos.append(Partido(
                fecha=fecha, liga=liga.codigo,
                liga_nombre=f"{liga.nombre} ({liga.pais})", temporada="proxima",
                local=(fila.get("Home") or "").strip(),
                visitante=(fila.get("Away") or "").strip(),
                cuota_local=cuotas[0], cuota_empate=cuotas[1], cuota_visitante=cuotas[2],
            ))

    proximos.sort(key=lambda p: (p.fecha, p.liga))
    return proximos
