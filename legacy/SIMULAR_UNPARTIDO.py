"""
predict_match_apifootball_v2_corregido.py
-----------------------------------------
Versión mejorada del predictor de partidos usando API-Football.

Mejoras sobre la v1:
  1. Head-to-head directo entre los dos equipos (pesa en la predicción)
  2. Standings actuales (puntos por partido = indicador de fuerza real)
  3. Muestra más grande de partidos recientes, separados correctamente
     por localía (últimos N como local / últimos N como visitante)
  4. Fallback completo con promedios de la liga (si faltan datos).
  5. Manejo de errores de la API.

CÓMO CAMBIAR DE PARTIDO O TORNEO:
    Edita las variables en la sección CONFIGURACIÓN.

CÓMO ENCONTRAR EL league_id DE TU TORNEO:
    Corre primero buscar_liga_id("Copa Libertadores") -- imprime opciones
    con su id y temporada disponible. Ya está incluido como paso 0 al
    correr este script directamente.
"""

import requests
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize

# ============================================================
# CONFIGURACIÓN — cambia esto para cada partido que quieras analizar
# ============================================================
# Usar únicamente IDs: configura HOME_ID y AWAY_ID (ej: HOME_ID = 34)
HOME_ID = 473
AWAY_ID = 124
LEAGUE_NAME = "Copa Libertadores"
SEASON = 2026                # Si es None, se usará la última temporada con partidos finalizados

API_KEY = "2ea1ff78a5c5f279d729509d550bcf61"
N_PARTIDOS_FORMA = 20     # partidos recientes a analizar por equipo (total, se filtran por local/visita)
N_PARTIDOS_H2H = 10        # partidos entre ellos a considerar

# Peso relativo de cada componente en el cálculo final (deben sumar 1.0)
PESO_FORMA_RECIENTE = 0.55
PESO_H2H = 0.25
PESO_STANDINGS = 0.20

# ============================================================
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}


def buscar_liga_id(nombre_liga: str) -> list:
    """Busca una liga/torneo por nombre. Útil para encontrar el league_id correcto."""
    resp = requests.get(f"{BASE_URL}/leagues", headers=HEADERS, params={"name": nombre_liga})
    resp.raise_for_status()
    data = resp.json()
    if data.get("errors"):
        print(f"Error API al buscar liga '{nombre_liga}': {data['errors']}")
        return []
    return [
        {
            "id": item["league"]["id"],
            "name": item["league"]["name"],
            "temporadas_disponibles": [s["year"] for s in item["seasons"]],
        }
        for item in data.get("response", [])
    ]


def obtener_ultima_temporada(league_id: int) -> int:
    """Obtiene la temporada más reciente con partidos finalizados para una liga."""
    resp = requests.get(f"{BASE_URL}/leagues", headers=HEADERS, params={"id": league_id})
    resp.raise_for_status()
    data = resp.json()
    if not data.get("response"):
        return 2025  # valor por defecto
    seasons = data["response"][0]["seasons"]
    seasons_sorted = sorted(seasons, key=lambda s: s["year"], reverse=True)
    for s in seasons_sorted:
        resp2 = requests.get(
            f"{BASE_URL}/fixtures",
            headers=HEADERS,
            params={"league": league_id, "season": s["year"], "status": "FT", "limit": 1},
        )
        if resp2.json().get("response"):
            return s["year"]
    return seasons_sorted[0]["year"] if seasons_sorted else 2025


def obtener_ultimos_partidos(team_id: int, n: int = 20) -> list:
    """Trae los últimos N partidos finalizados de un equipo, en cualquier competición."""
    resp = requests.get(
        f"{BASE_URL}/fixtures",
        headers=HEADERS,
        params={"team": team_id, "last": n, "status": "FT"},
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("errors"):
        print(f"Error API al obtener partidos del equipo {team_id}: {data['errors']}")
        return []
    return data.get("response", [])


def obtener_head_to_head(id1: int, id2: int, n: int = 10) -> list:
    """Trae los últimos N enfrentamientos directos entre dos equipos."""
    resp = requests.get(
        f"{BASE_URL}/fixtures/headtohead",
        headers=HEADERS,
        params={"h2h": f"{id1}-{id2}", "last": n, "status": "FT"},
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("errors"):
        print(f"Error API al obtener H2H entre {id1} y {id2}: {data['errors']}")
        return []
    return data.get("response", [])


def obtener_standings(league_id: int, season: int, team_id: int) -> dict | None:
    """Trae la posición y puntos por partido de un equipo en la tabla actual."""
    resp = requests.get(
        f"{BASE_URL}/standings",
        headers=HEADERS,
        params={"league": league_id, "season": season},
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("errors") or not data.get("response"):
        print(f"Sin standings para liga {league_id} temporada {season}")
        return None

    for bloque in data["response"][0]["league"]["standings"]:
        for equipo in bloque:
            if equipo["team"]["id"] == team_id:
                jugados = equipo["all"]["played"] or 1
                return {
                    "posicion": equipo["rank"],
                    "puntos_por_partido": equipo["points"] / jugados,
                    "forma": equipo.get("form", ""),
                }
    return None


def obtener_promedios_liga(league_id: int, season: int) -> dict:
    """Calcula promedios de goles como local y visitante a partir de partidos de la liga."""
    resp = requests.get(
        f"{BASE_URL}/fixtures",
        headers=HEADERS,
        params={"league": league_id, "season": season, "status": "FT", "limit": 200},
    )
    resp.raise_for_status()
    fixtures = resp.json().get("response", [])
    if not fixtures:
        # valores neutrales si no hay partidos
        return {"gf_local": 1.3, "gc_local": 1.1, "gf_visita": 1.0, "gc_visita": 1.3}

    home_goals = []
    away_goals = []
    for f in fixtures:
        gh = f["goals"]["home"]
        ga = f["goals"]["away"]
        if gh is not None and ga is not None:
            home_goals.append(gh)
            away_goals.append(ga)

    if not home_goals or not away_goals:
        return {"gf_local": 1.3, "gc_local": 1.1, "gf_visita": 1.0, "gc_visita": 1.3}

    return {
        "gf_local": float(np.mean(home_goals)),   # goles a favor como local
        "gc_local": float(np.mean(away_goals)),   # goles en contra como local (promedio del visitante)
        "gf_visita": float(np.mean(away_goals)),  # goles a favor como visitante
        "gc_visita": float(np.mean(home_goals)),  # goles en contra como visitante (promedio del local)
    }


def calcular_promedio_goles(team_id: int, partidos: list, liga_stats: dict) -> dict:
    """Calcula promedios de goles a favor y en contra separados por localía, con fallback a liga."""
    gf_local, gc_local, gf_visita, gc_visita = [], [], [], []

    for p in partidos:
        home_id = p["teams"]["home"]["id"]
        away_id = p["teams"]["away"]["id"]
        gh, ga = p["goals"]["home"], p["goals"]["away"]
        if gh is None or ga is None:
            continue

        if home_id == team_id:
            gf_local.append(gh)
            gc_local.append(ga)
        elif away_id == team_id:
            gf_visita.append(ga)
            gc_visita.append(gh)

    def prom(lista, fallback):
        return float(np.mean(lista)) if lista else fallback

    return {
        "gf_local": prom(gf_local, liga_stats["gf_local"]),
        "gc_local": prom(gc_local, liga_stats["gc_local"]),
        "gf_visita": prom(gf_visita, liga_stats["gf_visita"]),
        "gc_visita": prom(gc_visita, liga_stats["gc_visita"]),
        "muestra_local": len(gf_local),
        "muestra_visita": len(gf_visita),
    }


def calcular_goles_h2h(home_id: int, away_id: int, h2h: list) -> dict | None:
    """Promedio de goles del 'home_id' contra el 'away_id' en enfrentamientos previos."""
    goles_home_equipo, goles_away_equipo = [], []

    for p in h2h:
        gh, ga = p["goals"]["home"], p["goals"]["away"]
        if gh is None or ga is None:
            continue
        if p["teams"]["home"]["id"] == home_id:
            goles_home_equipo.append(gh)
            goles_away_equipo.append(ga)
        else:  # localías invertidas
            goles_home_equipo.append(ga)
            goles_away_equipo.append(gh)

    if not goles_home_equipo:
        return None

    return {
        "goles_home_prom": float(np.mean(goles_home_equipo)),
        "goles_away_prom": float(np.mean(goles_away_equipo)),
        "muestra": len(goles_home_equipo),
    }


def buscar_equipo_id(nombre_equipo: str) -> list:
    """Busca equipos por nombre para verificar IDs."""
    resp = requests.get(f"{BASE_URL}/teams", headers=HEADERS, params={"search": nombre_equipo})
    resp.raise_for_status()
    data = resp.json()
    if data.get("errors"):
        print(f"Error API al buscar equipo '{nombre_equipo}': {data['errors']}")
        return []
    return [{"id": t["team"]["id"], "name": t["team"]["name"]} for t in data.get("response", [])]


def predecir_partido_ids(home_id: int, away_id: int, league_name: str, season: int, pesos: dict | None = None) -> dict:
    """Predice un partido usando IDs de equipos y nombre de liga. Devuelve probabilidades y detalles."""
    if pesos is None:
        p_forma = PESO_FORMA_RECIENTE
        p_h2h = PESO_H2H
        p_stand = PESO_STANDINGS
    else:
        p_forma = pesos.get("forma", PESO_FORMA_RECIENTE)
        p_h2h = pesos.get("h2h", PESO_H2H)
        p_stand = pesos.get("standings", PESO_STANDINGS)

    # 1. Obtener league_id
    liga_opciones = buscar_liga_id(league_name)
    if not liga_opciones:
        raise ValueError(f"No se encontró la liga '{league_name}'")
    league_id = liga_opciones[0]["id"]
    print(f"Usando liga: {liga_opciones[0]['name']} (ID {league_id}), temporada {season}")

    # 2. Obtener promedios de la liga como fallback
    liga_stats = obtener_promedios_liga(league_id, season)

    # 3. Componente 1: forma reciente
    partidos_home = obtener_ultimos_partidos(home_id, N_PARTIDOS_FORMA)
    partidos_away = obtener_ultimos_partidos(away_id, N_PARTIDOS_FORMA)

    stats_home = calcular_promedio_goles(home_id, partidos_home, liga_stats)
    stats_away = calcular_promedio_goles(away_id, partidos_away, liga_stats)

    # Avisos si no hay datos
    if stats_home["muestra_local"] == 0 and stats_home["muestra_visita"] == 0:
        print(f"⚠️  El equipo local (ID {home_id}) no tiene partidos recientes. Verifica el ID.")
    if stats_away["muestra_local"] == 0 and stats_away["muestra_visita"] == 0:
        print(f"⚠️  El equipo visitante (ID {away_id}) no tiene partidos recientes. Verifica el ID.")

    lambda_home_forma = (stats_home["gf_local"] + stats_away["gc_visita"]) / 2
    lambda_away_forma = (stats_away["gf_visita"] + stats_home["gc_local"]) / 2

    # 4. Componente 2: head-to-head
    h2h_partidos = obtener_head_to_head(home_id, away_id, N_PARTIDOS_H2H)
    h2h_goles = calcular_goles_h2h(home_id, away_id, h2h_partidos)

    if h2h_goles:
        lambda_home_h2h = h2h_goles["goles_home_prom"]
        lambda_away_h2h = h2h_goles["goles_away_prom"]
    else:
        lambda_home_h2h = lambda_home_forma
        lambda_away_h2h = lambda_away_forma

    # 5. Componente 3: standings
    standings_home = obtener_standings(league_id, season, home_id)
    standings_away = obtener_standings(league_id, season, away_id)

    if standings_home and standings_away:
        ppp_home = standings_home["puntos_por_partido"]
        ppp_away = standings_away["puntos_por_partido"]
        total = ppp_home + ppp_away
        if total > 0:
            factor_home = (ppp_home / total) * 2
            factor_away = (ppp_away / total) * 2
        else:
            factor_home = factor_away = 1.0
    else:
        factor_home = factor_away = 1.0

    lambda_home_standings = lambda_home_forma * factor_home
    lambda_away_standings = lambda_away_forma * factor_away

    # 6. Cálculo final de lambdas combinadas
    lambda_home = (
        p_forma * lambda_home_forma
        + p_h2h * lambda_home_h2h
        + p_stand * lambda_home_standings
    )
    lambda_away = (
        p_forma * lambda_away_forma
        + p_h2h * lambda_away_h2h
        + p_stand * lambda_away_standings
    )

    # 7. Probabilidades con Poisson
    max_goals = 8
    matriz = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            matriz[i, j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
    matriz /= matriz.sum()

    prob_home = np.tril(matriz, -1).sum()
    prob_draw = np.trace(matriz)
    prob_away = np.triu(matriz, 1).sum()

    marcador = np.unravel_index(np.argmax(matriz), matriz.shape)

    return {
        "local_id": int(home_id),
        "visitante_id": int(away_id),
        "goles_esperados_local": float(round(lambda_home, 2)),
        "goles_esperados_visitante": float(round(lambda_away, 2)),
        "prob_gana_local": float(round(prob_home * 100, 1)),
        "prob_empate": float(round(prob_draw * 100, 1)),
        "prob_gana_visitante": float(round(prob_away * 100, 1)),
        "marcador_mas_probable": f"{marcador[0]}-{marcador[1]}",
        "detalle": {
            "forma_reciente": {
                "local": float(round(lambda_home_forma, 2)),
                "visitante": float(round(lambda_away_forma, 2)),
                "muestra_local": stats_home["muestra_local"],
                "muestra_visitante": stats_away["muestra_visita"],
            },
            "head_to_head": {
                "local": float(round(lambda_home_h2h, 2)),
                "visitante": float(round(lambda_away_h2h, 2)),
                "partidos_directos": h2h_goles["muestra"] if h2h_goles else 0,
            },
            "standings": {"local": standings_home, "visitante": standings_away},
        },
    }


def brier_score_for_probs(probs: np.ndarray, truths: np.ndarray) -> float:
    return float(((probs - truths) ** 2).sum(axis=1).mean())


def tune_pesos_por_brier(league_name: str, season: int, n_muestras: int = 200) -> dict:
    """Optimiza los pesos para minimizar Brier score usando partidos pasados de la liga."""
    opciones = buscar_liga_id(league_name)
    if not opciones:
        raise ValueError("No se encontró la liga para tunear pesos")
    league_id = opciones[0]["id"]

    resp = requests.get(
        f"{BASE_URL}/fixtures",
        headers=HEADERS,
        params={"league": league_id, "season": season, "status": "FT", "limit": n_muestras},
    )
    resp.raise_for_status()
    fixtures = resp.json().get("response", [])
    if not fixtures:
        raise ValueError("No hay fixtures para la liga/temporada solicitada")

    probs_list = []
    truths = []

    for f in fixtures:
        home_id = f["teams"]["home"]["id"]
        away_id = f["teams"]["away"]["id"]
        gh = f["goals"]["home"]
        ga = f["goals"]["away"]
        if gh is None or ga is None:
            continue
        if gh > ga:
            truths.append([1, 0, 0])
        elif gh == ga:
            truths.append([0, 1, 0])
        else:
            truths.append([0, 0, 1])
        probs_list.append((home_id, away_id))

    truths = np.array(truths)

    def loss_from_x(x):
        x = np.maximum(x, 1e-8)
        w = x / x.sum()
        pesos = {"forma": float(w[0]), "h2h": float(w[1]), "standings": float(w[2])}
        probs = []
        for home_id, away_id in probs_list:
            p = predecir_partido_ids(home_id, away_id, league_name, season, pesos)
            probs.append([
                p["prob_gana_local"] / 100.0,
                p["prob_empate"] / 100.0,
                p["prob_gana_visitante"] / 100.0,
            ])
        probs = np.array(probs)
        return brier_score_for_probs(probs, truths)

    x0 = np.array([PESO_FORMA_RECIENTE, PESO_H2H, PESO_STANDINGS])
    res = minimize(
        lambda x: loss_from_x(x),
        x0 + 1e-3,
        method="Nelder-Mead",
        options={"maxiter": 200, "disp": False},
    )
    x_opt = np.maximum(res.x, 1e-8)
    w_opt = (x_opt / x_opt.sum()).tolist()
    return {
        "forma": w_opt[0],
        "h2h": w_opt[1],
        "standings": w_opt[2],
        "brier": float(res.fun),
        "success": res.success,
    }


def verificar_liga(nombre_liga: str = LEAGUE_NAME):
    opciones = buscar_liga_id(nombre_liga)
    if not opciones:
        print(f"No se encontró ninguna liga con el nombre '{nombre_liga}'")
        return
    print(f"\nOpciones encontradas para '{nombre_liga}':\n")
    for op in opciones:
        print(f"  id={op['id']:<6} nombre='{op['name']}'  temporadas: {op['temporadas_disponibles']}")
    print(f"\nEl script usará automáticamente: id={opciones[0]['id']} ('{opciones[0]['name']}')")


if __name__ == "__main__":
    import sys

    # Comandos especiales
    if "--verificar-liga" in sys.argv:
        idx = sys.argv.index("--verificar-liga")
        termino = None
        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-"):
            termino = sys.argv[idx + 1]
        verificar_liga(termino if termino else LEAGUE_NAME)
        sys.exit(0)

    if "--buscar-equipo" in sys.argv:
        idx = sys.argv.index("--buscar-equipo")
        if idx + 1 < len(sys.argv):
            nombre = sys.argv[idx + 1]
            equipos = buscar_equipo_id(nombre)
            if equipos:
                print(f"\nEquipos encontrados para '{nombre}':")
                for e in equipos:
                    print(f"  id={e['id']:<6} nombre='{e['name']}'")
            else:
                print(f"No se encontraron equipos con '{nombre}'")
        else:
            print("Uso: --buscar-equipo <nombre>")
        sys.exit(0)

    if "--tune-pesos" in sys.argv:
        print("Ejecutando ajuste de pesos por Brier score... Puede tardar varios minutos.")
        try:
            # Usar temporada proporcionada o la última con FT
            if SEASON is None:
                liga_opts = buscar_liga_id(LEAGUE_NAME)
                if liga_opts:
                    season_to_tune = obtener_ultima_temporada(liga_opts[0]["id"])
                else:
                    season_to_tune = 2025
            else:
                season_to_tune = SEASON
            res = tune_pesos_por_brier(LEAGUE_NAME, season_to_tune, n_muestras=200)
            print(f"Resultado ajuste: {res}")
        except Exception as e:
            print(f"Error durante tuning: {e}")
        sys.exit(0)

    # Flujo principal de predicción
    if HOME_ID is None or AWAY_ID is None:
        print("ERROR: Debes configurar HOME_ID y AWAY_ID en la parte superior del archivo.")
        sys.exit(1)

    # Determinar temporada
    if SEASON is None:
        liga_opts = buscar_liga_id(LEAGUE_NAME)
        if liga_opts:
            season_to_use = obtener_ultima_temporada(liga_opts[0]["id"])
            print(f"Temporada automática seleccionada: {season_to_use}")
        else:
            season_to_use = 2025
            print("No se pudo determinar la temporada automáticamente, usando 2025")
    else:
        season_to_use = SEASON

    try:
        resultado = predecir_partido_ids(HOME_ID, AWAY_ID, LEAGUE_NAME, season_to_use)
    except Exception as e:
        print(f"Error en la predicción: {e}")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"Local (id={resultado['local_id']}) vs Visitante (id={resultado['visitante_id']})")
    print(f"{'='*50}")
    print(f"Goles esperados: {resultado['goles_esperados_local']} - {resultado['goles_esperados_visitante']}")
    print(f"\nProbabilidad victoria local: {resultado['prob_gana_local']}%")
    print(f"Probabilidad empate: {resultado['prob_empate']}%")
    print(f"Probabilidad victoria visitante: {resultado['prob_gana_visitante']}%")
    print(f"Marcador más probable: {resultado['marcador_mas_probable']}")

    print(f"\n--- Detalle de componentes ---")
    d = resultado["detalle"]
    print(f"Forma reciente -> local: {d['forma_reciente']['local']} (muestra local {d['forma_reciente']['muestra_local']}), "
          f"visitante: {d['forma_reciente']['visitante']} (muestra visita {d['forma_reciente']['muestra_visitante']})")
    print(f"Head-to-head -> local: {d['head_to_head']['local']}, visitante: {d['head_to_head']['visitante']} "
          f"(basado en {d['head_to_head']['partidos_directos']} partidos directos)")
    print(f"Standings -> local: {d['standings']['local']}, visitante: {d['standings']['visitante']}")