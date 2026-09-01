import sqlite3
from pathlib import Path

# Anclado al directorio del script -- ver nota en actualizar_resultados.py.
DB_PATH = Path(__file__).resolve().parent / "futbol_predicciones.db"

DDL_SCHEMA = """
PRAGMA foreign_keys = ON;

-- 1. Equipos
CREATE TABLE IF NOT EXISTS equipos (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre             TEXT NOT NULL UNIQUE,   -- tal como aparece en el CSV (ej. "Man City")
    nombre_normalizado TEXT NOT NULL,          -- nombre corto oficial (ej. "Manchester City")
    codigo             TEXT,                   -- trigrama del catalogo (ej. "MCI"); NULL si no catalogado
    liga_actual        TEXT
);

-- 2. Partidos (Tabla ancla)
CREATE TABLE IF NOT EXISTS partidos (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha               DATE NOT NULL,
    temporada           TEXT NOT NULL,
    liga                TEXT NOT NULL,
    equipo_local_id     INTEGER NOT NULL REFERENCES equipos(id),
    equipo_visitante_id INTEGER NOT NULL REFERENCES equipos(id),
    goles_local         INTEGER,
    goles_visitante     INTEGER,
    fuente              TEXT NOT NULL DEFAULT 'football-data.co.uk',
    UNIQUE(fecha, equipo_local_id, equipo_visitante_id)
);

-- 3. Estadísticas detalladas del partido
CREATE TABLE IF NOT EXISTS estadisticas_partido (
    partido_id              INTEGER PRIMARY KEY REFERENCES partidos(id) ON DELETE CASCADE,
    tiros_local             INTEGER,
    tiros_visitante         INTEGER,
    tiros_puerta_local      INTEGER,
    tiros_puerta_visitante  INTEGER,
    corners_local           INTEGER,
    corners_visitante       INTEGER,
    faltas_local            INTEGER,
    faltas_visitante        INTEGER,
    tarjetas_am_local       INTEGER,
    tarjetas_am_visitante   INTEGER,
    tarjetas_roj_local      INTEGER,
    tarjetas_roj_visitante  INTEGER,
    posesion_local          REAL,
    posesion_visitante      REAL
);

-- 4. Cuotas de cierre de mercado
CREATE TABLE IF NOT EXISTS cuotas_cierre (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    partido_id      INTEGER NOT NULL REFERENCES partidos(id) ON DELETE CASCADE,
    mercado         TEXT NOT NULL,
    -- 0 = mercado sin linea (1x2, btts). SQLite trata cada NULL como distinto
    -- de cualquier otro NULL en un indice UNIQUE, asi que un NULL real aqui
    -- rompe la deduplicacion de INSERT OR IGNORE en cada re-ingesta.
    linea           REAL NOT NULL DEFAULT 0,
    cuota_local     REAL,
    cuota_empate    REAL,
    cuota_visitante REAL,
    UNIQUE(partido_id, mercado, linea)
);

-- Índices para optimizar consultas de modelado
CREATE INDEX IF NOT EXISTS idx_partidos_fecha ON partidos(fecha);
CREATE INDEX IF NOT EXISTS idx_partidos_temporada_liga ON partidos(temporada, liga);
CREATE INDEX IF NOT EXISTS idx_cuotas_partido_mercado ON cuotas_cierre(partido_id, mercado);
"""

def inicializar_base_datos(db_path: Path = DB_PATH):
    with sqlite3.connect(db_path) as conn:
        conn.executescript(DDL_SCHEMA)
        conn.commit()
    print(f"[OK] Base de datos inicializada en: {db_path.resolve()}")

if __name__ == "__main__":
    inicializar_base_datos()