import io
import sqlite3
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path

from equipos_premier import resolver_equipo

# Anclado al directorio del script -- ver nota en actualizar_resultados.py.
DB_PATH = Path(__file__).resolve().parent / "futbol_predicciones.db"
TIMEOUT = 30  # segundos. 26 temporadas = 26 descargas; una que se cuelgue no debe trabar todo.

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

def generar_temporadas(anio_inicio=2000, anio_fin=2026):
    temporadas = []
    for y in range(anio_inicio, anio_fin):
        s_code = f"{str(y)[-2:]}{str(y+1)[-2:]}"
        s_label = f"{y}/{y+1}"
        temporadas.append((s_code, s_label))
    return temporadas

def procesar_cuotas(cursor, partido_id, row):
    cuotas = []
    
    # 1. Mercado 1X2 (Prioridad Pinnacle/Bet365/Promedio)
    h = safe_float(row.get("PSH")) or safe_float(row.get("B365H")) or safe_float(row.get("AvgH"))
    d = safe_float(row.get("PSD")) or safe_float(row.get("B365D")) or safe_float(row.get("AvgD"))
    a = safe_float(row.get("PSA")) or safe_float(row.get("B365A")) or safe_float(row.get("AvgA"))
    if h and d and a:
        # linea = 0.0 (no None): el mercado 1x2 no tiene linea, y NULL en una
        # columna UNIQUE nunca deduplica en SQLite (cada NULL cuenta como
        # distinto). 0.0 es el centinela de "sin linea" para toda la tabla.
        cuotas.append((partido_id, "1x2", 0.0, h, d, a))

    # 2. Mercado Over / Under 2.5
    over = safe_float(row.get("B365>2.5")) or safe_float(row.get("Avg>2.5")) or safe_float(row.get("BbAv>2.5"))
    under = safe_float(row.get("B365<2.5")) or safe_float(row.get("Avg<2.5")) or safe_float(row.get("BbAv<2.5"))
    if over and under:
        cuotas.append((partido_id, "over_under", 2.5, over, None, under))

    # 3. Mercado Hándicap Asiático
    linea_ah = safe_float(row.get("AHh")) or safe_float(row.get("BbAHh"))
    ah_h = safe_float(row.get("B365AHH")) or safe_float(row.get("AvgAHH"))
    ah_a = safe_float(row.get("B365AHA")) or safe_float(row.get("AvgAHA"))
    if linea_ah is not None and ah_h and ah_a:
        cuotas.append((partido_id, "handicap_asiatico", linea_ah, ah_h, None, ah_a))

    if cuotas:
        cursor.executemany("""
            INSERT OR IGNORE INTO cuotas_cierre (partido_id, mercado, linea, cuota_local, cuota_empate, cuota_visitante)
            VALUES (?, ?, ?, ?, ?, ?)
        """, cuotas)

def poblar_base_datos():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    cursor = conn.cursor()

    liga_codigo = "E0"  # Premier League
    temporadas = generar_temporadas(2000, 2026)

    print("Iniciando descarga e ingesta de datos históricos (2000 - 2026)...")

    for s_code, s_label in temporadas:
        url = f"https://www.football-data.co.uk/mmz4281/{s_code}/E0.csv"
        try:
            resp = requests.get(url, timeout=TIMEOUT)
        except requests.RequestException as e:
            print(f"[!] No se pudo descargar {s_label}: {e}")
            continue

        # football-data.co.uk responde con una pagina HTML (a veces con
        # status 300 "Multiple Choices", que requests NO trata como error --
        # solo levanta excepcion en 4xx/5xx) cuando la temporada todavia no
        # esta publicada. Detectarlo aca evita mandarle HTML a pd.read_csv.
        # utf-8-sig, no latin1: football-data.co.uk manda el CSV con BOM al
        # principio, que con latin1 queda pegado al nombre de la primera
        # columna ("Div") en vez de sacarse.
        cuerpo = resp.content.decode("utf-8-sig", errors="replace")
        es_html = "<html" in cuerpo[:300].lower()
        if resp.status_code != 200 or len(resp.content) < 500 or es_html:
            print(f"[!] {s_label} aun no disponible en football-data.co.uk (status {resp.status_code})")
            continue

        try:
            df = pd.read_csv(io.StringIO(cuerpo), on_bad_lines="skip")
        except pd.errors.ParserError as e:
            print(f"[!] No se pudo interpretar el CSV de {s_label}: {e}")
            continue

        df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        partidos_insertados = 0

        filas_con_error = 0

        for _, row in df.iterrows():
            fecha = parsear_fecha(row.get("Date"))
            if not fecha:
                continue

            local_csv = str(row["HomeTeam"]).strip()
            visitante_csv = str(row["AwayTeam"]).strip()

            try:
                codigo_local, local = resolver_equipo(local_csv)
                codigo_visitante, visitante = resolver_equipo(visitante_csv)

                # 1. Inserción de Equipos (nombre = string crudo del CSV, para
                # que el lookup de mas abajo siga encontrando la fila incluso
                # si el equipo no esta catalogado).
                cursor.execute(
                    "INSERT OR IGNORE INTO equipos (nombre, nombre_normalizado, codigo, liga_actual) VALUES (?, ?, ?, ?)",
                    (local_csv, local, codigo_local, liga_codigo))
                cursor.execute(
                    "INSERT OR IGNORE INTO equipos (nombre, nombre_normalizado, codigo, liga_actual) VALUES (?, ?, ?, ?)",
                    (visitante_csv, visitante, codigo_visitante, liga_codigo))

                cursor.execute("SELECT id FROM equipos WHERE nombre = ?", (local_csv,))
                local_id = cursor.fetchone()[0]
                cursor.execute("SELECT id FROM equipos WHERE nombre = ?", (visitante_csv,))
                visitante_id = cursor.fetchone()[0]

                # 2. Inserción de Partidos
                cursor.execute("""
                    INSERT OR IGNORE INTO partidos (fecha, temporada, liga, equipo_local_id, equipo_visitante_id, goles_local, goles_visitante)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (fecha, s_label, liga_codigo, local_id, visitante_id, safe_int(row.get("FTHG")), safe_int(row.get("FTAG"))))

                cursor.execute("""
                    SELECT id FROM partidos WHERE fecha = ? AND equipo_local_id = ? AND equipo_visitante_id = ?
                """, (fecha, local_id, visitante_id))
                partido_id = cursor.fetchone()[0]

                # 3. Inserción de Estadísticas Detalladas
                cursor.execute("""
                    INSERT OR IGNORE INTO estadisticas_partido (
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

                # 4. Inserción de Cuotas de Cierre
                procesar_cuotas(cursor, partido_id, row)
                partidos_insertados += 1

            except Exception as e:
                # Una fila mala no debe tirar la temporada entera: se loguea,
                # se sigue con la siguiente, y lo ya insertado en esta
                # temporada se conserva (el commit es por temporada, mas abajo).
                filas_con_error += 1
                print(f"[!] Fila con error ({fecha} {local_csv} vs {visitante_csv}): {e}")
                continue

        conn.commit()
        msg = f"[OK] Temporada {s_label}: {partidos_insertados} partidos procesados."
        if filas_con_error:
            msg += f" ({filas_con_error} filas con error, ver arriba)"
        print(msg)

    conn.close()
    print("\nProceso finalizado. Base de datos actualizada con éxito.")

if __name__ == "__main__":
    poblar_base_datos()