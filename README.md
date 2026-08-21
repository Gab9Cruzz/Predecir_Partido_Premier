# PredicirUnPartido

Predice el resultado y la probabilidad de victoria de un partido de fútbol con un
modelo **Dixon–Coles** ajustado por máxima verosimilitud sobre datos **gratuitos,
sin API key y actualizados hasta hace uno o dos días**.

```
python predecir.py partido --liga SP1 --local Barcelona --visitante "Real Madrid"
```

```
====================================================================
  Barcelona  vs  Real Madrid
====================================================================

Goles esperados : 1.99 - 1.34
Marcador top    : 2-1 (9.5%)

Resultado                probabilidad   cuota justa
----------------------------------------------------
Gana Barcelona           52.7%      1.90  ##################
Empate                   21.5%      4.64  #######
Gana Real Madrid         25.8%      3.88  #########
```

---

## 1. El problema de la API anterior, y cuál se usa ahora

El proyecto usaba **API-Football (api-sports.io)**. En el plan gratuito esa API
solo da acceso a unas pocas temporadas antiguas, así que era imposible predecir
nada actual sin pagar.

La fuente nueva es **[football-data.co.uk](https://www.football-data.co.uk)**:

| | API-Football (plan gratis) | football-data.co.uk |
|---|---|---|
| API key | obligatoria | **ninguna** |
| Registro | sí | **no** |
| Límite de peticiones | 100/día | ninguno |
| Temporadas | solo antiguas | **histórico completo + temporada en curso** |
| Actualización | — | cada pocas horas |
| Cuotas del mercado | no en el plan gratis | **sí, de 10+ casas** |
| Tiros, córners, tarjetas | no en el plan gratis | sí (ligas europeas) |

Se comprobó al construir esto: los datos llegaban hasta **dos días antes**, tanto
en LaLiga como en la liga argentina, la brasileña o la MLS.

Son **38 ligas**. `python predecir.py ligas` las lista todas:

- **Europa** (con estadísticas de tiros, córners y tarjetas): Inglaterra (5
  divisiones), Escocia (4), Alemania (2), Italia (2), España (2), Francia (2),
  Holanda, Bélgica, Portugal, Turquía, Grecia.
- **Resto del mundo** (resultado + cuotas): Argentina, Brasil, México, EEUU,
  Japón, China, Austria, Dinamarca, Finlandia, Irlanda, Noruega, Polonia,
  Rumanía, Rusia, Suecia, Suiza.

El código antiguo quedó en [`legacy/`](legacy/) por si quieres consultarlo.

---

## 2. Instalación

```bash
pip install -r requirements.txt
```

Solo hacen falta `requests`, `numpy` y `scipy`. No hay que configurar nada más:
no hay claves, ni `.env`, ni cuentas.

Para comprobar que todo funciona:

```bash
python tests.py
```

---

## 3. Uso

### Ver las ligas y los equipos disponibles

```bash
python predecir.py ligas
python predecir.py equipos --liga SP1
```

### Predecir un partido

```bash
python predecir.py partido --liga SP1 --local Barcelona --visitante "Real Madrid"
```

Los nombres no tienen que ser exactos: `atl. madrid`, `espanyol` o
`rayo vallecano` se resuelven solos, y el programa te dice con qué los emparejó.
Si un nombre no existe en los datos **no se inventa uno parecido**: te avisa.

Opciones útiles:

| opción | para qué |
|---|---|
| `--cuotas 1.95 3.90 3.60` | compara tus probabilidades con las del mercado y calcula el EV |
| `--neutral` | campo neutral (finales a sede única): reparte la ventaja de local |
| `--temporadas 6` | cuánto histórico usar (por defecto 4) |
| `--json` | salida en JSON para usarla desde otro programa |
| `--actualizar` | ignora la caché y vuelve a descargar |

### Predecir todos los partidos de los próximos días

```bash
python predecir.py proximos --liga SP1,E0
python predecir.py proximos              # todas las ligas a la vez
```

Trae el calendario con las cuotas que ofrece el mercado ahora mismo, predice cada
partido y señala dónde más discrepan modelo y mercado.

### Ver la fuerza estimada de cada equipo

```bash
python predecir.py ratings --liga SP1
```

```
    # equipo                     ataque  defensa  general    pj
  ------------------------------------------------------------
    1 Barcelona                   +0.66    +0.28    +0.94   114
    2 Real Madrid                 +0.49    +0.36    +0.85   114
    3 Ath Madrid                  +0.31    +0.22    +0.53   114
```

No es la tabla de posiciones: es la fuerza real estimada, ya descontada la
calidad de los rivales a los que se ha enfrentado cada equipo.

### Validar el modelo

```bash
python predecir.py backtest --liga SP1
python predecir.py optimizar --liga SP1     # busca el mejor xi para esa liga
```

---

## 4. Cómo funciona el modelo

### Lo que hacía la versión anterior, y por qué fallaba

El script antiguo calculaba los goles esperados así:

```
lambda_local = (goles_a_favor_del_local_en_casa + goles_en_contra_del_visitante_fuera) / 2
```

Eso tiene un fallo grave: **no corrige por la calidad del rival**. Un equipo que
acaba de jugar contra los cinco últimos de la tabla parece buenísimo, y uno que
jugó contra los cinco primeros parece malísimo. El sesgo es enorme.

Además mezclaba tres componentes (forma, head-to-head, clasificación) con pesos
inventados a mano, y el head-to-head sobre 10 partidos repartidos en años es casi
puro ruido.

### Lo que hace ahora

Se estiman a la vez, por máxima verosimilitud, **un parámetro de ataque y uno de
defensa para cada equipo**:

```
goles_local  ~ Poisson(exp(mu + ataque_local  - defensa_visitante + ventaja_local))
goles_visita ~ Poisson(exp(mu + ataque_visitante - defensa_local))
```

Como todos los equipos aparecen en la misma ecuación, la fuerza del rival queda
descontada automáticamente: no hace falta ningún peso a mano.

Encima de eso hay tres cosas más:

1. **Corrección τ de Dixon–Coles.** Dos Poisson independientes subestiman los 0-0
   y 1-1 y sobreestiman los 1-0 y 0-1. El parámetro ρ corrige exactamente esas
   cuatro casillas, que es donde se juegan los empates.

2. **Ponderación temporal.** Cada partido pesa `exp(-xi · días_de_antigüedad)`.
   El valor por defecto (`xi = 0.0018`, semivida ≈ 385 días) no es inventado: se
   eligió **minimizando el RPS del backtest** en LaLiga, Premier, Bundesliga y
   Serie A. Puedes recalcularlo para tu liga con `predecir.py optimizar`.

3. **Regularización L2.** Encoge hacia la media a los equipos con pocos partidos.
   Un recién ascendido con tres jornadas no se lleva un rating extremo por una
   racha corta. El valor por defecto (2.0) también salió del backtest.

El ajuste usa **gradiente analítico** con L-BFGS-B: una liga entera se ajusta en
menos de una décima de segundo, lo que hace viable reajustar el modelo cientos de
veces durante un backtest.

Referencia: Dixon, M.J. y Coles, S.G. (1997), *Modelling Association Football
Scores and Inefficiencies in the Football Betting Market*, Applied Statistics
46(2), 265–280.

---

## 5. ¿Funciona? Resultados medidos

`backtest` recorre el histórico hacia adelante y predice cada partido usando
**solo** los partidos anteriores a él, reajustando el modelo cada semana. Se
compara contra dos rivales: el azar y las cuotas de cierre del mercado.

Resultado sobre 6 temporadas por liga, unos **18.000 partidos** en total:

| liga | partidos | RPS modelo | RPS mercado | diferencia | acierto modelo | acierto mercado |
|---|---|---|---|---|---|---|
| Premier League | 1454 | 0.2028 | 0.1949 | +0.0079 | 53.4% | 55.2% |
| LaLiga | 1469 | 0.1978 | 0.1916 | +0.0062 | 53.0% | 54.7% |
| Bundesliga | 1164 | 0.2016 | 0.1935 | +0.0080 | 52.1% | 55.5% |
| Serie A | 1446 | 0.1967 | 0.1907 | +0.0059 | 52.7% | 53.8% |
| Ligue 1 | 1235 | 0.2075 | 0.2013 | +0.0062 | 52.2% | 54.1% |
| Eredivisie | 1183 | 0.1921 | 0.1854 | +0.0066 | 54.3% | 56.2% |
| Primeira Liga | 1190 | 0.1823 | 0.1759 | +0.0064 | 56.5% | 58.6% |
| Argentina | 2591 | 0.2150 | 0.2109 | +0.0041 | 42.3% | 43.7% |
| Brasil | 1914 | 0.2086 | 0.2007 | +0.0079 | 49.0% | 51.3% |
| México | 1649 | 0.2108 | 0.2041 | +0.0067 | 49.7% | 50.9% |
| MLS | 2503 | 0.2204 | 0.2116 | +0.0088 | 47.2% | 48.5% |

*RPS = Ranked Probability Score, la métrica estándar en predicción de fútbol.
Más bajo es mejor. El azar puro está en ~0.235.*

Cómo leerlo:

- **El modelo bate al azar con holgura** en las 11 ligas.
- **Está bien calibrado**: cuando dice 30%, pasa aproximadamente el 30% de las
  veces (el backtest imprime la tabla de calibración).
- **Pierde contra el mercado por poco pero de forma sistemática**: entre 0.004 y
  0.009 de RPS, y entre 1 y 3 puntos de acierto. Esto es lo esperable y es
  información valiosa, no un fracaso: el mercado incorpora alineaciones,
  lesiones y rotaciones que este modelo no ve.

---

## 6. Limitaciones (léelas)

**No sirve para ganar dinero apostando.** El backtest incluye una simulación de
apuestas de valor con criterio de Kelly. En LaLiga, 968 apuestas dieron un
**ROI de −15.1%**, y un bankroll de 1.00 acabó en 0.028. Es el resultado esperado: el modelo pierde contra el mercado,
y encima hay que pagar el margen de la casa (~5%). Las discrepancias que marca
`proximos` son *puntos donde el modelo no coincide con el mercado*, y lo más
probable es que el equivocado sea el modelo.

**Copa Libertadores y otras competiciones continentales no están cubiertas.**
Ninguna fuente gratuita las ofrece con datos actuales: football-data.co.uk no las
incluye, y en football-data.org están en un plan de pago. Lo que sí se puede
hacer es cargar las ligas domésticas de los equipos implicados:

```bash
python predecir.py partido --liga ARG,BRA --local "Boca Juniors" --visitante Palmeiras --neutral
```

Aviso importante sobre esto: cuando dos ligas **nunca se enfrentan entre sí**
(argentina contra brasileña), sus niveles no son directamente comparables. El
modelo iguala el nivel goleador de ambas pero no sabe cuál es más fuerte, así
que ese resultado hay que tomárselo como orientativo. En cambio, con divisiones
del mismo país (`--liga SP1,SP2`) sí funciona bien: los ascensos y descensos de
las temporadas cargadas conectan las dos categorías, y el modelo llega a separar
LaLiga de LaLiga 2 por unos 0.41 log-goles sin que nadie se lo diga.

**Lo que el modelo no ve:** alineaciones, lesiones, sanciones, si el equipo tiene
partido de Champions entre semana, cambios de entrenador, si el partido no vale
nada porque ya está todo decidido, y el clima. Todo eso sí lo ve el mercado, y
por eso gana.

**Equipos con pocos partidos.** Un recién ascendido tiene el rating muy encogido
hacia la media; el programa lo marca con `*` y avisa. Ahí las predicciones son
flojas, y son también las que generan falsos "valores" espectaculares.

---

## 7. Estructura del proyecto

```
predecir.py                     interfaz de línea de comandos
tests.py                        pruebas (python tests.py)
futbol/
  fuentes/
    base.py                     Partido + emparejamiento de nombres de equipo
    footballdata_uk.py          descarga y parseo de football-data.co.uk
  modelo/
    dixon_coles.py              el modelo: ajuste, predicción y ratings
    mercado.py                  cuotas, probabilidades implícitas, Kelly, métricas
  backtest.py                   validación walk-forward y optimización de xi
datos/cache/                    CSV descargados (caché de 6 horas)
legacy/                         el código anterior, basado en API-Football
```

Como biblioteca:

```python
from futbol.fuentes import footballdata_uk as fuente
from futbol.modelo import DixonColes

partidos = fuente.cargar("SP1", temporadas=4)
modelo = DixonColes().ajustar(partidos)

pred = modelo.predecir("Barcelona", "Real Madrid")
print(pred.prob_local, pred.prob_empate, pred.prob_visitante)
print(pred.prob_mas_de(2.5), pred.prob_ambos_marcan)
print(pred.marcadores_probables[:3])
```

---

## 8. Resumen de qué cambió

| | antes | ahora |
|---|---|---|
| Datos | API-Football, temporadas viejas | football-data.co.uk, hasta hace 2 días |
| API key | obligatoria (y estaba escrita en el código) | ninguna |
| Ligas | dependía del plan | 38 |
| Modelo | promedios de goles con pesos a mano | Dixon–Coles por máxima verosimilitud |
| Calidad del rival | ignorada | descontada por el propio ajuste |
| Empates | Poisson simple | corrección τ de Dixon–Coles |
| Pasado lejano | mismo peso que ayer | decaimiento exponencial calibrado |
| Pocos partidos | rating extremo | encogido hacia la media |
| Validación | ninguna real | walk-forward sobre 18.000 partidos |
| Comparación con el mercado | ninguna | RPS, Brier, LogLoss, calibración y ROI |
| Salidas | 1X2 y marcador | 1X2, over/under, BTTS, hándicap, top-10 marcadores |
| Nombres de equipo | IDs numéricos a mano | búsqueda por nombre, sin falsos positivos |
| Pruebas | ninguna | 67 comprobaciones, incluida la del gradiente |

Sobre la API key: la del proyecto anterior estaba escrita directamente en
`EQUIPOS.PY` y `SIMULAR_UNPARTIDO.py`, y sigue ahí en `legacy/`. Si esa cuenta te
importa, **revócala y genera otra**; da igual que el plan sea gratuito.
