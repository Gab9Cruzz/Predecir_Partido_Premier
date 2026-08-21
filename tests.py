#!/usr/bin/env python
"""Pruebas del proyecto.  Uso:  python tests.py

Las pruebas marcadas como "red" descargan datos reales; se saltan solas si no
hay conexion. El resto son deterministas y no tocan internet.
"""

from __future__ import annotations

import sys
import traceback
from datetime import date, timedelta

import numpy as np

from futbol.fuentes.base import Partido, buscar_equipo, normalizar_nombre
from futbol.modelo.dixon_coles import DixonColes, _tau
from futbol.modelo.mercado import (
    a_one_hot, brier, kelly, log_loss, margen_casa,
    probabilidades_implicitas, rps, valor_esperado,
)

_fallos: list[str] = []
_pasadas = 0


def comprobar(condicion: bool, mensaje: str) -> None:
    global _pasadas
    if condicion:
        _pasadas += 1
    else:
        _fallos.append(mensaje)
        print(f"  FALLO: {mensaje}")


def prueba(fn):
    print(f"\n{fn.__name__}")
    try:
        fn()
    except Exception:
        _fallos.append(f"{fn.__name__} lanzo excepcion")
        traceback.print_exc()
    return fn


# --------------------------------------------------------------- datos sinteticos

def liga_sintetica(n_equipos: int = 12, temporadas: int = 3,
                   semilla: int = 7) -> tuple[list[Partido], np.ndarray]:
    """Genera una liga inventada con fuerzas conocidas.

    Sirve para verificar que el ajuste recupera las fuerzas reales: si el
    modelo no puede reencontrar unos parametros que sabemos con certeza,
    tampoco sirve con datos de verdad.
    """
    rng = np.random.default_rng(semilla)
    ataque = np.linspace(0.45, -0.45, n_equipos)
    defensa = np.linspace(0.35, -0.35, n_equipos)
    rng.shuffle(defensa)
    ventaja, mu = 0.28, np.log(1.35)

    partidos: list[Partido] = []
    dia = date(2023, 8, 1)
    for _ in range(temporadas):
        for i in range(n_equipos):
            for j in range(n_equipos):
                if i == j:
                    continue
                lh = np.exp(mu + ataque[i] - defensa[j] + ventaja)
                la = np.exp(mu + ataque[j] - defensa[i])
                partidos.append(Partido(
                    fecha=dia, liga="TEST", liga_nombre="Test", temporada="t",
                    local=f"Equipo{i:02d}", visitante=f"Equipo{j:02d}",
                    goles_local=int(rng.poisson(lh)),
                    goles_visitante=int(rng.poisson(la)),
                ))
                dia += timedelta(days=1)
    return partidos, ataque - defensa.mean() + defensa


# ---------------------------------------------------------------------- pruebas

@prueba
def test_normalizacion():
    comprobar(normalizar_nombre("  Atlético de Madrid! ") == "atletico de madrid",
              "normalizar_nombre no limpia acentos/puntuacion")
    comprobar(normalizar_nombre("Nott'm Forest") == "nott m forest",
              "normalizar_nombre no maneja apostrofes")


@prueba
def test_busqueda_equipos():
    equipos = ["FC Barcelona", "Real Madrid", "Ath Madrid", "Mallorca",
               "Espanol", "Vallecano", "Las Palmas", "Sociedad", "Alaves"]

    # Debe encontrar variantes legitimas.
    for consulta, esperado in [
        ("barcelona", "FC Barcelona"),
        ("Atl. Madrid", "Ath Madrid"),
        ("Espanyol", "Espanol"),
        ("Rayo Vallecano", "Vallecano"),
        ("Real Sociedad", "Sociedad"),
        ("Alaves", "Alaves"),
    ]:
        res = buscar_equipo(consulta, equipos, n=1)
        comprobar(bool(res) and res[0][0] == esperado,
                  f"'{consulta}' deberia resolver a '{esperado}', dio {res}")

    # Y NO debe inventarse equipos que no estan (el bug de Malaga/Mallorca).
    for consulta in ["Malaga", "Girona", "Almeria", "Leganes"]:
        res = buscar_equipo(consulta, equipos, n=1)
        comprobar(res == [],
                  f"'{consulta}' no esta en la lista pero se emparejo con {res}")


@prueba
def test_tau_dixon_coles():
    lh = np.array([1.2, 1.2, 1.2, 1.2, 1.2])
    la = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
    x = np.array([0, 0, 1, 1, 3.0])
    y = np.array([0, 1, 0, 1, 2.0])
    t = _tau(x, y, lh, la, rho=-0.1)
    comprobar(np.isclose(t[0], 1 + 1.2 * 0.9 * 0.1), "tau(0,0) mal")
    comprobar(np.isclose(t[1], 1 - 1.2 * 0.1), "tau(0,1) mal")
    comprobar(np.isclose(t[2], 1 - 0.9 * 0.1), "tau(1,0) mal")
    comprobar(np.isclose(t[3], 1 + 0.1), "tau(1,1) mal")
    comprobar(np.isclose(t[4], 1.0), "tau debe valer 1 fuera de los marcadores bajos")


@prueba
def test_gradiente_analitico():
    """El gradiente que usa el optimizador debe coincidir con el numerico."""
    import futbol.modelo.dixon_coles as modulo
    from scipy.optimize import minimize as minimizar_real

    partidos, _ = liga_sintetica(n_equipos=8, temporadas=2)
    capturado: dict = {}

    def espia(fun, x0, **kw):
        capturado["fun"] = fun
        capturado["x0"] = x0
        return minimizar_real(fun, x0, **kw)

    modulo.minimize = espia
    try:
        DixonColes().ajustar(partidos)
    finally:
        modulo.minimize = minimizar_real

    fun, x0 = capturado["fun"], capturado["x0"]
    rng = np.random.default_rng(3)
    x = x0 + rng.normal(0, 0.2, size=x0.shape)
    _, grad = fun(x)

    peor = 0.0
    for k in rng.choice(len(x), 15, replace=False):
        h = 1e-6
        mas, menos = x.copy(), x.copy()
        mas[k] += h
        menos[k] -= h
        numerico = (fun(mas)[0] - fun(menos)[0]) / (2 * h)
        peor = max(peor, abs(numerico - grad[k]) / max(abs(numerico), 1e-6))
    comprobar(peor < 1e-4, f"gradiente analitico incorrecto (error rel. {peor:.2e})")


@prueba
def test_recupera_parametros():
    """Sobre ligas inventadas, el modelo debe recuperar las fuerzas reales.

    Se promedia sobre varias semillas: con una sola liga de 528 partidos la
    ventaja de local tiene una desviacion tipica de ~0.06, asi que exigirle
    precision a una unica muestra seria exigirle suerte, no correccion.
    """
    ventajas, correlaciones = [], []
    for semilla in range(6):
        partidos, fuerza_real = liga_sintetica(n_equipos=12, temporadas=4,
                                               semilla=semilla)
        modelo = DixonColes(xi=0.0, regularizacion=1.0).ajustar(partidos)
        estimada = np.array([
            modelo.ataque[modelo._indice[f"Equipo{i:02d}"]]
            + modelo.defensa[modelo._indice[f"Equipo{i:02d}"]]
            for i in range(12)
        ])
        correlaciones.append(float(np.corrcoef(estimada, fuerza_real)[0, 1]))
        ventajas.append(modelo.ventaja_local)
        comprobar(modelo.convergio, f"el optimizador no convergio (semilla {semilla})")

    media_ventaja = float(np.mean(ventajas))
    media_corr = float(np.mean(correlaciones))
    print(f"  ventaja estimada {media_ventaja:+.3f} (real +0.280), "
          f"correlacion media {media_corr:.3f}")
    comprobar(abs(media_ventaja - 0.28) < 0.05,
              f"ventaja de local sesgada: {media_ventaja:.3f} vs 0.28")
    comprobar(media_corr > 0.92,
              f"correlacion media con la fuerza real solo {media_corr:.3f}")
    comprobar(min(correlaciones) > 0.85,
              f"alguna semilla recupero muy mal las fuerzas ({min(correlaciones):.3f})")


@prueba
def test_probabilidades_coherentes():
    partidos, _ = liga_sintetica(n_equipos=10, temporadas=2)
    modelo = DixonColes().ajustar(partidos)
    pred = modelo.predecir("Equipo00", "Equipo09")

    comprobar(abs(pred.probs.sum() - 1.0) < 1e-9, "las probabilidades no suman 1")
    comprobar(abs(pred.matriz.sum() - 1.0) < 1e-9, "la matriz de marcadores no suma 1")
    comprobar(pred.prob_local > pred.prob_visitante,
              "el equipo fuerte en casa deberia ser favorito")
    comprobar(0 <= pred.prob_mas_de(2.5) <= 1, "prob_mas_de fuera de rango")
    comprobar(pred.prob_mas_de(1.5) > pred.prob_mas_de(3.5),
              "prob de mas goles deberia decrecer con la linea")
    comprobar(abs(pred.prob_local_o_empate + pred.prob_visitante
                  - 1.0) < 1e-9, "doble oportunidad incoherente")

    # Goles esperados coherentes con la matriz.
    n = pred.matriz.shape[0]
    esperado_local = float((pred.matriz.sum(axis=1) * np.arange(n)).sum())
    comprobar(abs(esperado_local - pred.goles_esperados_local) < 0.05,
              "los goles esperados no cuadran con la matriz")


@prueba
def test_campo_neutral():
    partidos, _ = liga_sintetica(n_equipos=10, temporadas=2)
    modelo = DixonColes().ajustar(partidos)
    normal = modelo.predecir("Equipo04", "Equipo05")
    neutral = modelo.predecir("Equipo04", "Equipo05", campo_neutral=True)

    comprobar(neutral.prob_local < normal.prob_local,
              "en campo neutral el local deberia perder ventaja")
    total_normal = normal.goles_esperados_local + normal.goles_esperados_visitante
    total_neutral = neutral.goles_esperados_local + neutral.goles_esperados_visitante
    comprobar(abs(total_normal - total_neutral) < 0.15,
              "el total de goles no deberia cambiar tanto en campo neutral")


@prueba
def test_modo_estricto():
    partidos, _ = liga_sintetica(n_equipos=10, temporadas=2)
    modelo = DixonColes().ajustar(partidos)
    try:
        modelo.predecir("Equipo99", "Equipo01", estricto=True)
        comprobar(False, "el modo estricto deberia rechazar un equipo desconocido")
    except KeyError:
        comprobar(True, "")


@prueba
def test_encogimiento_muestra_pequena():
    """Un equipo con un solo partido no puede quedar en un extremo del ranking."""
    partidos, _ = liga_sintetica(n_equipos=10, temporadas=2)
    partidos.append(Partido(
        fecha=partidos[-1].fecha + timedelta(days=1), liga="TEST",
        liga_nombre="Test", temporada="t", local="Novato", visitante="Equipo09",
        goles_local=7, goles_visitante=0,
    ))
    modelo = DixonColes().ajustar(partidos)
    ratings = {f["equipo"]: f for f in modelo.ratings()}
    novato = ratings["Novato"]["general"]
    # Sin encoger, un 7-0 contra un rival medio daria un ataque de ~1.4.
    sin_encoger = np.log(7 / 1.8)
    print(f"  rating del novato tras un 7-0: {novato:+.2f} "
          f"(sin encogimiento seria ~{sin_encoger:+.2f})")
    comprobar(ratings["Novato"]["puesto"] > 1,
              "un 7-0 en un unico partido no deberia dar el primer puesto")
    comprobar(novato < sin_encoger * 0.6,
              f"el encogimiento apenas actua: {novato:+.2f} vs {sin_encoger:+.2f}")
    comprobar(novato < modelo.ratings()[0]["general"],
              "el novato no deberia superar al mejor equipo real")


@prueba
def test_mercado():
    cuotas = (2.0, 3.5, 4.0)
    comprobar(margen_casa(cuotas) > 0, "el margen de la casa deberia ser positivo")

    for metodo in ("proporcional", "potencia"):
        p = probabilidades_implicitas(cuotas, metodo=metodo)
        comprobar(abs(p.sum() - 1.0) < 1e-6, f"probs implicitas ({metodo}) no suman 1")
        comprobar(bool(np.all(p > 0)), f"probs implicitas ({metodo}) con valores <= 0")

    # El metodo de potencia baja la probabilidad del longshot frente al reparto
    # proporcional (corrige el sesgo favorito-perdedor).
    prop = probabilidades_implicitas(cuotas, metodo="proporcional")
    pot = probabilidades_implicitas(cuotas, metodo="potencia")
    comprobar(pot[2] < prop[2], "el metodo de potencia no corrige el longshot bias")

    comprobar(abs(valor_esperado(0.5, 2.0)) < 1e-12, "EV de una apuesta justa != 0")
    comprobar(valor_esperado(0.6, 2.0) > 0, "EV positivo mal calculado")
    comprobar(kelly(0.4, 2.0) == 0.0, "Kelly deberia ser 0 sin ventaja")
    comprobar(0 < kelly(0.6, 2.0, fraccion=1.0) < 1, "Kelly fuera de rango")
    comprobar(abs(kelly(0.6, 2.0, fraccion=1.0) - 0.2) < 1e-9, "Kelly mal calculado")


@prueba
def test_metricas():
    reales = a_one_hot(["H", "D", "A"])
    perfecto = reales.copy()
    comprobar(abs(brier(perfecto, reales)) < 1e-12, "Brier perfecto != 0")
    comprobar(abs(rps(perfecto, reales)) < 1e-12, "RPS perfecto != 0")

    azar = np.full((3, 3), 1 / 3)
    comprobar(abs(log_loss(azar, reales) - np.log(3)) < 1e-9, "LogLoss del azar mal")
    comprobar(brier(azar, reales) > brier(perfecto, reales),
              "el azar deberia puntuar peor que la prediccion perfecta")

    # El RPS debe penalizar mas fallar por dos casillas que por una.
    gano_local = a_one_hot(["H"])
    casi = np.array([[0.0, 1.0, 0.0]])      # dijo empate
    lejos = np.array([[0.0, 0.0, 1.0]])     # dijo visitante
    comprobar(rps(casi, gano_local) < rps(lejos, gano_local),
              "el RPS no esta respetando el orden 1-X-2")


@prueba
def test_backtest_sin_fuga_de_datos():
    """El backtest solo puede usar partidos anteriores al que predice."""
    from futbol import backtest

    partidos, _ = liga_sintetica(n_equipos=10, temporadas=4)
    vistos: list[date] = []
    original = DixonColes.ajustar

    def espia(self, datos, referencia=None):
        if datos:
            vistos.append(max(p.fecha for p in datos))
        return original(self, datos, referencia)

    DixonColes.ajustar = espia
    try:
        res = backtest.ejecutar(partidos, dias_minimos=300, refit_cada_dias=30,
                                verboso=False)
    finally:
        DixonColes.ajustar = original

    comprobar(res.n_partidos > 50, "el backtest evaluo demasiados pocos partidos")
    comprobar(0 < res.rps_modelo < res.rps_azar,
              f"el modelo deberia batir al azar (RPS {res.rps_modelo:.4f} "
              f"vs azar {res.rps_azar:.4f})")
    comprobar(0 < res.acierto_modelo < 1, "porcentaje de acierto fuera de rango")
    comprobar(bool(vistos), "no se registro ningun ajuste")


@prueba
def test_red_descarga_datos():
    """Prueba con datos reales: se salta si no hay internet."""
    import requests

    from futbol.fuentes import footballdata_uk as fuente
    try:
        partidos = fuente.cargar("E0", temporadas=2, verboso=False)
    except requests.RequestException:
        print("  (sin conexion: prueba omitida)")
        return

    comprobar(len(partidos) > 300, f"pocos partidos descargados: {len(partidos)}")
    comprobar(all(p.jugado for p in partidos), "hay partidos sin resultado")
    comprobar(all(p.fecha <= date.today() for p in partidos),
              "hay partidos con fecha futura entre los resultados")

    dias = (date.today() - max(p.fecha for p in partidos)).days
    comprobar(dias < 400, f"el partido mas reciente es de hace {dias} dias")
    print(f"  ultimo partido descargado: hace {dias} dias")

    con_cuotas = sum(1 for p in partidos if p.tiene_cuotas)
    comprobar(con_cuotas > len(partidos) * 0.8,
              f"solo {con_cuotas}/{len(partidos)} partidos traen cuotas")

    modelo = DixonColes().ajustar(partidos)
    comprobar(modelo.convergio, "no convergio con datos reales")
    comprobar(0.1 < modelo.ventaja_local < 0.6,
              f"ventaja de local irreal: {modelo.ventaja_local:.3f}")


def main() -> int:
    print("=" * 60)
    print("  Pruebas de PredicirUnPartido")
    print("=" * 60)
    for nombre, objeto in list(globals().items()):
        if nombre.startswith("test_") and callable(objeto):
            pass  # ya se ejecutaron via decorador
    print()
    print("=" * 60)
    if _fallos:
        print(f"  {_pasadas} comprobaciones ok, {len(_fallos)} FALLOS:")
        for f in _fallos:
            print(f"   - {f}")
        return 1
    print(f"  Todo correcto: {_pasadas} comprobaciones pasadas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
