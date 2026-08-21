"""Catalogo de equipos de Premier League: nombre oficial, nombre corto y
codigo de 3 letras (trigrama).

Cubre los clubes que jugaron en la Premier League (o bajaron/subieron de
ella) desde la temporada 2000/01 en adelante -- la ventana que descarga
cargar_datos.py. Fuente: tabla provista por el usuario.

El "nombre corto" coincide con el que usa football-data.co.uk en las
columnas HomeTeam/AwayTeam para la enorme mayoria de los equipos -- se
verifico contra los CSV de E0 ya cacheados en datos/cache/ (temporadas
2021/22 a 2025/26). Los unicos tres que NO coinciden se resuelven via
ALIAS_CSV, con evidencia real de los CSV:

    CSV dice "Man City"       -> corto real "Manchester City" (MCI)
    CSV dice "Man United"     -> corto real "Manchester United" (MUN)
    CSV dice "Nott'm Forest"  -> corto real "Nottingham Forest" (NFO)

Si aparece un nombre de equipo que no resuelve ni por alias ni por match
exacto contra el catalogo (por ejemplo un club de temporadas muy viejas
que no esta en esta lista), resolver_equipo() no rompe la ingesta: avisa
por consola y el equipo entra a la base sin codigo, para que se pueda
completar el catalogo despues sin perder el partido.
"""

from __future__ import annotations

# codigo -> (nombre_oficial, nombre_corto)
CATALOGO: dict[str, tuple[str, str]] = {
    "ARS": ("Arsenal FC", "Arsenal"),
    "AVL": ("Aston Villa FC", "Aston Villa"),
    "BOU": ("AFC Bournemouth", "Bournemouth"),
    "BIR": ("Birmingham City FC", "Birmingham"),
    "BLA": ("Blackburn Rovers FC", "Blackburn"),
    "BKP": ("Blackpool FC", "Blackpool"),
    "BOL": ("Bolton Wanderers FC", "Bolton"),
    "BRE": ("Brentford FC", "Brentford"),
    "BHA": ("Brighton & Hove Albion FC", "Brighton"),
    "BUR": ("Burnley FC", "Burnley"),
    "CAR": ("Cardiff City FC", "Cardiff"),
    "CHA": ("Charlton Athletic FC", "Charlton"),
    "CHE": ("Chelsea FC", "Chelsea"),
    "COV": ("Coventry City FC", "Coventry"),
    "CRY": ("Crystal Palace FC", "Crystal Palace"),
    "DER": ("Derby County FC", "Derby"),
    "EVE": ("Everton FC", "Everton"),
    "FUL": ("Fulham FC", "Fulham"),
    "HUD": ("Huddersfield Town AFC", "Huddersfield"),
    "HUL": ("Hull City AFC", "Hull City"),
    "IPS": ("Ipswich Town FC", "Ipswich"),
    "LEE": ("Leeds United FC", "Leeds"),
    "LEI": ("Leicester City FC", "Leicester"),
    "LIV": ("Liverpool FC", "Liverpool"),
    "LUT": ("Luton Town FC", "Luton"),
    "MCI": ("Manchester City FC", "Manchester City"),
    "MUN": ("Manchester United FC", "Manchester United"),
    "MID": ("Middlesbrough FC", "Middlesbrough"),
    "NEW": ("Newcastle United FC", "Newcastle"),
    "NOR": ("Norwich City FC", "Norwich"),
    "NFO": ("Nottingham Forest FC", "Nottingham Forest"),
    "POR": ("Portsmouth FC", "Portsmouth"),
    "QPR": ("Queens Park Rangers FC", "QPR"),
    "RDG": ("Reading FC", "Reading"),
    "SHU": ("Sheffield United FC", "Sheffield United"),
    "SHW": ("Sheffield Wednesday FC", "Sheffield Wednesday"),
    "SOU": ("Southampton FC", "Southampton"),
    "STK": ("Stoke City FC", "Stoke"),
    "SUN": ("Sunderland AFC", "Sunderland"),
    "SWA": ("Swansea City AFC", "Swansea"),
    "TOT": ("Tottenham Hotspur FC", "Tottenham"),
    "WAT": ("Watford FC", "Watford"),
    "WBA": ("West Bromwich Albion FC", "West Brom"),
    "WHU": ("West Ham United FC", "West Ham"),
    "WOL": ("Wolverhampton Wanderers FC", "Wolves"),
}

# Nombre EXACTO como aparece en las columnas HomeTeam/AwayTeam de
# football-data.co.uk -> codigo. Solo para los casos donde no coincide con
# el "nombre corto" del catalogo. Verificado contra CSV reales -- no
# adivinado (ver docstring del modulo).
ALIAS_CSV: dict[str, str] = {
    "Man City": "MCI",
    "Man United": "MUN",
    "Nott'm Forest": "NFO",
    "Hull": "HUL",  # visto en fixtures.csv real (proximos partidos 2026/27)
}

# Indice inverso nombre_corto -> codigo, para el caso comun donde el CSV
# ya trae el nombre corto tal cual (la mayoria de los equipos).
_POR_NOMBRE_CORTO: dict[str, str] = {
    corto: codigo for codigo, (_, corto) in CATALOGO.items()
}


def resolver_equipo(nombre_csv: str, silencioso: bool = False) -> tuple[str | None, str]:
    """Resuelve un nombre de equipo tal como viene del CSV de football-data.co.uk.

    Devuelve (codigo, nombre_corto). Si el nombre no esta catalogado,
    codigo es None y nombre_corto es el nombre del CSV tal cual (recortado) --
    el partido igual se guarda, no se pierde por un equipo sin catalogar.

    silencioso=True evita el aviso por consola -- para usos interactivos
    (resolver lo que escribio un usuario) donde un "no catalogado" es normal
    y no un problema de ingesta que haya que loguear.
    """
    nombre_csv = nombre_csv.strip()

    codigo = ALIAS_CSV.get(nombre_csv)
    if codigo is None:
        codigo = _POR_NOMBRE_CORTO.get(nombre_csv)

    if codigo is None:
        if not silencioso:
            print(f"[!] Equipo no catalogado: '{nombre_csv}' -- se guarda sin codigo. "
                  f"Agregalo a equipos_premier.py si vas a verlo seguido.")
        return None, nombre_csv

    return codigo, CATALOGO[codigo][1]


# codigo -> nombre exacto tal como aparece en el CSV de football-data.co.uk.
# Para los 3 alias (ver ALIAS_CSV), el nombre corto del catalogo ("Manchester
# City") no es el que trae el CSV ("Man City") -- esto invierte ALIAS_CSV
# para recuperar el nombre real del dataset. Para el resto, corto == CSV.
_NOMBRE_CSV_POR_CODIGO: dict[str, str] = {v: k for k, v in ALIAS_CSV.items()}


def nombre_en_datos(codigo: str) -> str | None:
    """Nombre exacto que usa el dataset (Partido.local/visitante) para este codigo.

    None si el codigo no existe en el catalogo.
    """
    if codigo not in CATALOGO:
        return None
    return _NOMBRE_CSV_POR_CODIGO.get(codigo, CATALOGO[codigo][1])


def resolver_a_nombre_en_datos(consulta: str) -> str | None:
    """Atajo: de un nombre cualquiera (oficial, corto, o el del CSV) al nombre
    exacto que usa el dataset. None si no esta catalogado -- el llamador
    decide el fallback (por ejemplo, el fuzzy-matching de base.py).
    """
    codigo, _ = resolver_equipo(consulta, silencioso=True)
    return nombre_en_datos(codigo) if codigo else None
