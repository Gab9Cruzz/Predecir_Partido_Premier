import io
import sqlite3
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path

from equipos_premier import resolver_equipo

DB_PATH = Path("futbol_predicciones.db")
TIMEOUT = 30

# Temporada actual y código en football-data (Ej: 2026/2027 -> '2627')
TEMPORADA_LABEL = "2026/2027"
TEMPORADA_CODE = "2627"
LIGA_CODIGO = "E0"
URL_CSV_ACTUAL = f"https://www.football-data.co.uk/mmz4281/{TEMPORADA_CODE}/{LIGA_CODIGO}.csv"
URL_PROXIMOS = "https://www.football-data.co.uk/fixtures.csv"

def parsear_fecha(fecha_str):
    if pd.isna(fecha_str):
        return None
    fecha_str = str(fecha_str).strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(fecha_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None

def temporada_en_curso() -> str:
    """'2026/2027' si hoy es agosto 2026 o despues, '2025/2026' si es antes de julio."""
    hoy = datetime.now()
    inicio = hoy.year if hoy.month >= 7 else hoy.year - 1
    return f"{inicio}/{inicio + 1}"

def safe_int(val):
    try:
        if pd.notna(val):
            return int(float(val))
    except (ValueError, TypeError):
        pass
    return None

def safe_float(val):
    try:
        if pd.notna(val):
            return float(val)
    except (ValueError, TypeError):
        pass
    return None

def actualizar_cuotas(cursor, partido_id, row):
    cuotas = []

    # 1. 1X2
    h = safe_float(row.get("PSH")) or safe_float(row.get("B365H")) or safe_float(row.get("AvgH"))
    d = safe_float(row.get("PSD")) or safe_float(row.get("B365D")) or safe_float(row.get("AvgD"))
    a = safe_float(row.get("PSA")) or safe_float(row.get("B365A")) or safe_float(row.get("AvgA"))
    if h and d and a:
        # linea = 0.0 (no None): mismo motivo que en cargar_datos.py -- NULL
        # nunca deduplica en un indice UNIQUE de SQLite.
        cuotas.append((partido_id, "1x2", 0.0, h, d, a))

    # 2. Over / Under 2.5
    over = safe_float(row.get("B365>2.5")) or safe_float(row.get("Avg>2.5")) or safe_float(row.get("BbAv>2.5"))
    under = safe_float(row.get("B365<2.5")) or safe_float(row.get("Avg<2.5")) or safe_float(row.get("BbAv<2.5"))
    if over and under:
        cuotas.append((partido_id, "over_under", 2.5, over, None, under))

    # 3. Hándicap Asiático
    linea_ah = safe_float(row.get("AHh")) or safe_float(row.get("BbAHh"))
    ah_h = safe_float(row.get("B365AHH")) or safe_float(row.get("AvgAHH"))
    ah_a = safe_float(row.get("B365AHA")) or safe_float(row.get("AvgAHA"))
    if linea_ah is not None and ah_h and ah_a:
        cuotas.append((partido_id, "handicap_asiatico", linea_ah, ah_h, None, ah_a))

    if cuotas:
        cursor.executemany("""
            INSERT OR REPLACE INTO cuotas_cierre (partido_id, mercado, linea, cuota_local, cuota_empate, cuota_visitante)
            VALUES (?, ?, ?, ?, ?, ?)
        """, cuotas)

def _procesar_partido(cursor, fecha: str, local_csv: str, visitante_csv: str,
                      temporada: str, row) -> int:
    """Inserta/actualiza un partido -- jugado o pendiente -- y sus cuotas.

    Sirve para las dos fuentes (jornada jugada y proximos partidos): la
    unica diferencia entre ambas es si `row` trae FTHG/FTAG o no. Devuelve
    el id del partido.
    """
    codigo_local, local = resolver_equipo(local_csv)
    codigo_visitante, visitante = resolver_equipo(visitante_csv)

    # nombre = local_csv/visitante_csv (el string crudo del CSV) -- el
    # SELECT de abajo tiene que buscar por lo mismo que se inserta aca,
    # no por el nombre ya resuelto del catalogo (ver nota de bug al pie).
    cursor.execute(
        "INSERT OR IGNORE INTO equipos (nombre, nombre_normalizado, codigo, liga_actual) VALUES (?, ?, ?, ?)",
        (local_csv, local, codigo_local, LIGA_CODIGO))
    cursor.execute(
        "INSERT OR IGNORE INTO equipos (nombre, nombre_normalizado, codigo, liga_actual) VALUES (?, ?, ?, ?)",
        (visitante_csv, visitante, codigo_visitante, LIGA_CODIGO))

    cursor.execute("SELECT id FROM equipos WHERE nombre = ?", (local_csv,))
    local_id = cursor.fetchone()[0]
    cursor.execute("SELECT id FROM equipos WHERE nombre = ?", (visitante_csv,))
    visitante_id = cursor.fetchone()[0]

    goles_l = safe_int(row.get("FTHG"))
    goles_v = safe_int(row.get("FTAG"))

    # UPSERT nativo de SQLite: si el partido ya estaba (por ejemplo, cargado
    # antes como "proximo" con goles NULL), esto lo actualiza in-place sin
    # duplicar la fila cuando el resultado real llega despues.
    cursor.execute("""
        INSERT INTO partidos (fecha, temporada, liga, equipo_local_id, equipo_visitante_id, goles_local, goles_visitante)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(fecha, equipo_local_id, equipo_visitante_id)
        DO UPDATE SET
            goles_local = excluded.goles_local,
            goles_visitante = excluded.goles_visitante
    """, (fecha, temporada, LIGA_CODIGO, local_id, visitante_id, goles_l, goles_v))

    cursor.execute("""
        SELECT id FROM partidos WHERE fecha = ? AND equipo_local_id = ? AND equipo_visitante_id = ?
    """, (fecha, local_id, visitante_id))
    partido_id = cursor.fetchone()[0]

    if pd.notna(row.get("HS")):
        cursor.execute("""
            INSERT OR REPLACE INTO estadisticas_partido (
                partido_id, tiros_local, tiros_visitante, tiros_puerta_local, tiros_puerta_visitante,
                corners_local, corners_visitante, faltas_local, faltas_visitante,
                tarjetas_am_local, tarjetas_am_visitante, tarjetas_roj_local, tarjetas_roj_visitante
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            partido_id,
            safe_int(row.get("HS")), safe_int(row.get("AS")),
            safe_int(row.get("HST")), safe_int(row.get("AST")),
            safe_int(row.get("HC")), safe_int(row.get("AC")),
            safe_int(row.get("HF")), safe_int(row.get("AF")),
            safe_int(row.get("HY")), safe_int(row.get("AY")),
            safe_int(row.get("HR")), safe_int(row.get("AR"))
        ))

    actualizar_cuotas(cursor, partido_id, row)
    return partido_id

def _descargar_csv(url: str, etiqueta: str) -> pd.DataFrame | None:
    """Descarga y valida un CSV de football-data.co.uk. None si no esta
    disponible (temporada no publicada, red caida, etc.) -- nunca crashea."""
    try:
        resp = requests.get(url, timeout=TIMEOUT)
    except requests.RequestException as e:
        print(f"[!] Error al descargar {etiqueta}: {e}")
        return None

    # football-data.co.uk responde con una pagina HTML (a veces con status 300
    # "Multiple Choices", que requests NO trata como error) cuando el archivo
    # no esta disponible. Detectarlo antes de parsear.
    # utf-8-sig (no latin1): football-data.co.uk manda el CSV con BOM al
    # principio. Con latin1 el BOM queda pegado al nombre de la primera
    # columna ("Div" pasa a ser un string distinto), lo que rompe cualquier
    # lookup contra esa columna en particular sin avisar (ver bug real
    # encontrado en sincronizar_proximos: df.get("Div") devolvia None).
    cuerpo = resp.content.decode("utf-8-sig", errors="replace")
    es_html = "<html" in cuerpo[:300].lower()
    if resp.status_code != 200 or len(resp.content) < 500 or es_html:
        print(f"[!] {etiqueta} aun no disponible (status {resp.status_code})")
        return None

    try:
        return pd.read_csv(io.StringIO(cuerpo), on_bad_lines="skip")
    except pd.errors.ParserError as e:
        print(f"[!] No se pudo interpretar el CSV de {etiqueta}: {e}")
        return None

def sincronizar_jornada():
    """Trae los resultados de la temporada en curso y actualiza la base."""
    print(f"Descargando última actualización de {TEMPORADA_LABEL}...")
    df = _descargar_csv(URL_CSV_ACTUAL, TEMPORADA_LABEL)
    if df is None:
        return

    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    cursor = conn.cursor()

    procesados = 0
    for _, row in df.iterrows():
        fecha = parsear_fecha(row.get("Date"))
        if not fecha:
            continue
        local_csv = str(row["HomeTeam"]).strip()
        visitante_csv = str(row["AwayTeam"]).strip()
        _procesar_partido(cursor, fecha, local_csv, visitante_csv, TEMPORADA_LABEL, row)
        procesados += 1

    conn.commit()
    conn.close()
    print(f"[OK] Base de datos sincronizada: {procesados} partidos procesados/actualizados.")

def sincronizar_proximos():
    """Trae los partidos de Premier League que todavia no se jugaron, con
    las cuotas actuales del mercado, y los deja en la base con goles NULL.
    Cuando el partido se juega, sincronizar_jornada() completa el resultado
    en la misma fila (mismo UPSERT por fecha+equipos)."""
    print("Descargando proximos partidos...")
    df = _descargar_csv(URL_PROXIMOS, "proximos partidos")
    if df is None:
        return

    df = df[df.get("Div") == LIGA_CODIGO]
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    if df.empty:
        print("  No hay proximos partidos de Premier League publicados ahora mismo.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    cursor = conn.cursor()
    temporada = temporada_en_curso()

    procesados = 0
    for _, row in df.iterrows():
        fecha = parsear_fecha(row.get("Date"))
        if not fecha:
            continue
        local_csv = str(row["HomeTeam"]).strip()
        visitante_csv = str(row["AwayTeam"]).strip()
        _procesar_partido(cursor, fecha, local_csv, visitante_csv, temporada, row)
        procesados += 1

    conn.commit()
    conn.close()
    print(f"[OK] {procesados} proximos partidos sincronizados.")

if __name__ == "__main__":
    sincronizar_jornada()
    sincronizar_proximos()
