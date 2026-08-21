# Incorporar xG al modelo — análisis de arquitectura y datos

Estado: análisis únicamente, **nada implementado todavía**. Ver decisión pendiente al final.

---

## 1. Arquitectura actual (relevante para xG)

```
predecir.py                     CLI (argparse), 9 subcomandos
futbol/
  fuentes/
    base.py                     Partido (dataclass) + fuzzy-matching de nombres
    footballdata_uk.py          descarga + parseo de football-data.co.uk
  modelo/
    dixon_coles.py               el modelo: MLE con gradiente analitico
    mercado.py                   cuotas, Kelly, RPS/Brier/LogLoss
  backtest.py                   walk-forward: refit cada 7 dias, evalua RPS vs mercado
datos/cache/                    CSV crudos descargados (cache 6h)
```

`Partido` (`futbol/fuentes/base.py:12`) es el contrato central: **solo** trae
`goles_local`, `goles_visitante` y las tres cuotas de cierre. No hay tiros,
córners, ni ningún campo de calidad de la ocasión. Todo lo que fluye desde
`footballdata_uk.py` hacia el modelo y el backtest pasa por esta forma.

`DixonColes.ajustar()` (`futbol/modelo/dixon_coles.py:176`) ajusta por máxima
verosimilitud sobre `gl`/`gv` (goles reales, enteros) usando un **gradiente
analítico hecho a mano** (líneas 222-275). Esto es el punto más importante del
análisis, ver §4.

---

## 2. Qué hay realmente en los datos (y qué no)

Comprobé las columnas reales de los CSV en caché, no la documentación de la
web:

**Ligas "main" (formato europeo, 22 competiciones — Inglaterra, Escocia,
Alemania, Italia, España, Francia, Holanda, Bélgica, Portugal, Turquía,
Grecia):**

```
FTHG,FTAG,FTR, HS,AS,HST,AST, HF,AF, HC,AC, HY,AY,HR,AR, + ~90 columnas de cuotas
```

Hay **tiros** (HS/AS), **tiros a puerta** (HST/AST) y **córners** (HC/AC).
No hay xG en ninguna columna. Nunca lo ha habido en football-data.co.uk.

**Ligas "extra" (16 competiciones — Argentina, Brasil, México, MLS, Japón,
China, Austria, Dinamarca, Finlandia, Irlanda, Noruega, Polonia, Rumanía,
Rusia, Suecia, Suiza):**

```
Country,League,Season,Date,Time,Home,Away,HG,AG,Res, + cuotas
```

Nada de tiros ni córners siquiera. Solo resultado y cuotas.

**Conclusión dura: la fuente de datos actual del proyecto no tiene xG, y
tampoco tiene los ingredientes para aproximarlo en la mitad de las ligas que
cubre.** Esto no es un detalle de parseo que falte — es una limitación de la
fuente. `footballdata_uk.py` ni siquiera parsea HS/HST/HC hoy (solo lee
FTHG/FTAG y las cuotas), así que ahora mismo se tira información que sí
existe para las ligas "main".

---

## 3. De dónde saldría el xG, si se quiere xG real

| Fuente | Cobertura | Costo | Riesgo |
|---|---|---|---|
| **Proxy con tiros/córners ya en los CSV** | Solo ligas "main" (~22 comps) | Ninguno — cero nueva dependencia | Es una aproximación, no xG real (no pondera ángulo/distancia/tipo de jugada) |
| **Understat.com** | ~6 ligas top (EPL, LaLiga, Bundesliga, Serie A, Ligue 1, RPL), desde 2014/15 | Gratis | Sin API oficial: hay que scrapear JSON embebido en la página. Frágil a cambios del sitio, zona gris de ToS |
| **FBref / StatsBomb (vía paquete `soccerdata`)** | Top-5 ligas europeas, desde ~2017/18 | Gratis, paquete pip mantenido | Rate-limited, también scraping por debajo, ToS gris |
| **Opta / Stats Perform / Wyscout** | Amplia, en tiempo real | **De pago, requiere API key** | Contradice el principio fundacional del proyecto (README §1: "sin API key, sin registro") |

**Esto es la tensión central de la decisión.** El README dedica toda su
sección 1 a celebrar haber dejado atrás una fuente que exigía API key. Toda
fuente de xG *real* es o bien scraping frágil de un sitio no pensado para
eso, o bien un servicio de pago con API key — exactamente lo que el proyecto
decidió evitar. La única opción que respeta el principio "gratis, sin key,
sin registro" es el proxy con tiros/córners que **ya está en los datos que
se descargan hoy**, y solo para la mitad de las ligas.

---

## 4. El hallazgo de arquitectura que importa más: xG no encaja como target del Poisson tal cual

`DixonColes.ajustar()` no solo usa `gl`/`gv` como la media del Poisson — los
usa también para indexar la corrección τ de Dixon-Coles (`_tau`,
`dixon_coles.py:132`), que corrige **específicamente las cuatro celdas de
marcador bajo discretas**: 0-0, 0-1, 1-0, 1-1 (`gl==0 & gv==0`, etc., líneas
136-143 y otra vez en el gradiente, líneas 243-254).

Esa corrección existe porque dos Poisson independientes subestiman 0-0/1-1 y
sobreestiman 1-0/0-1 — es un fenómeno de **marcadores enteros bajos**. El xG
es continuo (p. ej. 1.34, 2.07). Si se sustituyera `gl`/`gv` por xG
directamente como variable dependiente del Poisson, esas máscaras de
igualdad exacta (`gl == 0`) dejarían de tener sentido, y con ellas el
gradiente analítico completo que hace que el ajuste tarde <0.1s por liga.

En corto: **no es un cambio de una columna.** Sustituir goles por xG como
target rompe la pieza más cuidada del modelo actual (la corrección τ y su
gradiente cerrado). Cualquier plan serio tiene que tratar esto como
rediseño del objetivo de optimización, no como sustituir una entrada.

---

## 5. Tres estrategias de integración (de menor a mayor invasividad)

### A. Proxy de tiros como covariable de fuerza, no como target del Poisson
Se sigue ajustando el modelo sobre goles reales (τ intacta, gradiente
intacto). El proxy de tiros/córners se usa como una **señal adicional para
regularizar o inicializar** ataque/defensa — por ejemplo, un rating de
"calidad de ocasiones creadas" que se mezcla con el rating basado en goles
mediante un blend `α·goles + (1-α)·proxy_tiros`, con `α` optimizado por
backtest igual que se hace hoy con `xi`.

- Cobertura: ~22 ligas "main" (las otras 16 se quedan sin la señal extra,
  caen al comportamiento actual — no es un breaking change).
- Nueva dependencia: ninguna. Los datos ya se descargan, solo falta parsearlos.
- Riesgo: el gradiente analítico del ajuste no se toca — es la opción que
  menos amenaza la pieza más frágil del código.

### B. xG real (fuente externa) como blend con goles reales
Se descarga xG real de Understat/FBref, se empareja por fecha+equipos con
los partidos de football-data.co.uk (nombres de equipo distintos entre
fuentes → hay que extender `buscar_equipo` de `base.py` o mapear a mano), y
se usa como un componente adicional del target: técnica estándar en la
literatura ("blended xG models") donde el Poisson se ajusta sobre una mezcla
`goles_reales·(1-β) + xG·β` con `β` pequeño y calibrado por backtest, en vez
de sustituir goles por completo. Igual de intacta la corrección τ si se
sigue redondeando/discretizando el target sensatamente, pero la lógica de
mezcla y el emparejamiento de partidos entre dos fuentes es trabajo real.

- Cobertura: 5-6 ligas top únicamente.
- Nueva dependencia: scraper propio o `soccerdata`.
- Riesgo: mantenimiento del scraper, ToS, fuente no cae bajo el control del
  proyecto (football-data.co.uk es estable desde hace años; Understat/FBref
  cambian de estructura sin aviso).

### C. Modelo de dos etapas (rating por xG, calibración por goles)
Ajustar un primer Dixon-Coles simplificado sobre xG (sin τ, porque xG es
continuo — usaría verosimilitud gaussiana o gamma en vez de Poisson) para
obtener un rating de "calidad de juego" menos ruidoso que el basado en
goles, y usar ese rating como prior/regularizador del ajuste final sobre
goles reales (que sigue usando τ y Poisson tal cual). Es la opción
académicamente más "correcta" — separa señal de suerte (goles) de señal de
proceso (xG) — pero es la de más superficie nueva: dos ajustes optimizados
en cascada, dos verosimilitudes distintas, más superficie de bugs.

- Cobertura: igual que B (limitada a donde hay xG real).
- Riesgo: el mayor de los tres. Justificable solo si A y B ya se probaron y
  el backtest muestra que el rating basado en xG realmente reduce RPS
  respecto al basado en goles — cosa que hay que medir, no asumir.

---

## 6. Cambios de archivo necesarios (para cualquiera de las tres)

| Archivo | Cambio |
|---|---|
| `futbol/fuentes/base.py` | `Partido` necesita campos nuevos opcionales (`tiros_local`, `tiros_local_puerta`, `corners_local`, ... y/o `xg_local`/`xg_visitante` si se usa fuente externa). Es un `frozen dataclass` — añadir campos con default `None` es retrocompatible. |
| `futbol/fuentes/footballdata_uk.py` | `_parsear_main` ya lee la fila completa del CSV pero descarta HS/HST/HC — hay que empezar a mapearlas. `_parsear_extra` no tiene de dónde sacarlas (fuente no las trae). |
| `futbol/fuentes/xg_externo.py` (nuevo, solo si se elige B/C) | Cliente para Understat o wrapper de `soccerdata`, con caché en `datos/cache/` igual que el resto, y lógica de emparejamiento fecha+equipos con `buscar_equipo`. |
| `futbol/modelo/dixon_coles.py` | Estrategia A: nuevo término opcional en `objetivo_y_gradiente` (covariable, no cambia τ). Estrategia B/C: rediseño real del target y posiblemente de `_tau`/gradiente. |
| `futbol/backtest.py` | Extender `ejecutar()` para poder correr en modo "con proxy/xG" vs "sin" y comparar RPS — el proyecto ya tiene la infraestructura de A/B (es literalmente lo que hace `optimizar_xi`), solo hay que generalizarla a un segundo hiperparámetro. |
| `predecir.py` | Flag nuevo (`--usar-tiros` o similar) en los subcomandos `partido`/`proximos`/`backtest`/`optimizar`. |
| `tests.py` | Casos nuevos: parseo de las columnas de tiros, comportamiento cuando faltan (ligas "extra"), y que el gradiente analítico siga verificando numéricamente si se toca `objetivo_y_gradiente` (el proyecto ya prueba esto — README menciona "67 comprobaciones, incluida la del gradiente"). |
| `requirements.txt` | Sin cambios para A. `beautifulsoup4`/`lxml` o `soccerdata` para B/C. |

---

## 7. Riesgos a vigilar

1. **Cobertura desigual.** Cualquier estrategia dejará ~16 ligas ("extra")
   sin la señal nueva. El modelo tiene que degradar limpio a su
   comportamiento actual ahí, no fallar ni mentir con un proxy inventado.
2. **Fuga de datos histórica en xG externo.** Si se usa Understat/FBref, hay
   que confirmar que el xG de un partido se publica *después* del partido
   (es así, xG post-partido es estándar) y no reconstruir con datos que
   incluyan información posterior — el mismo cuidado que ya tiene
   `backtest.py` con el walk-forward de goles.
3. **Nombres de equipo entre fuentes.** `buscar_equipo` en `base.py` ya
   resuelve esto para football-data.co.uk; una segunda fuente externa
   duplica el problema con su propio vocabulario de nombres.
4. **Overfitting con poco histórico de xG.** Understat/FBref solo llegan a
   ~2014-2017 en adelante; el backtest actual usa 6 temporadas — está bien,
   pero no hay margen para retroceder más si se quiere xG real.
5. **El gradiente analítico es la joya del código.** Cualquier estrategia
   que lo toque (B parcialmente, C directamente) necesita el mismo cuidado
   que tuvo el original: verificación numérica del gradiente antes de
   confiar en la velocidad de ajuste.

---

## 8. Cómo validar antes de comprometerse

El proyecto ya tiene exactamente la herramienta necesaria:
`backtest.py:ejecutar()` + `optimizar_xi()`. El patrón a seguir es idéntico
al que ya usaron para calibrar `xi` y la regularización: correr walk-forward
con y sin la señal nueva, comparar RPS en las mismas ligas/temporadas, y
solo quedarse con la señal si baja el RPS de forma consistente (no en una
liga suelta). Dado que el modelo actual ya pierde contra el mercado por
0.004-0.009 de RPS (README §5), el bar a superar es: **reducir esa brecha,
no solo "no empeorar".**

---

## 9. Recomendación

**Empezar por la Estrategia A** (proxy de tiros/córners, ya presente en los
datos, cero dependencias nuevas, cero riesgo para el gradiente analítico) y
medirla con el backtest existente antes de tocar nada más. Es la única
opción de las tres que:

- No contradice el principio "gratis, sin API key, sin registro" del proyecto.
- No introduce una fuente externa frágil o de pago.
- No toca la corrección τ ni el gradiente analítico.
- Se puede medir con la infraestructura de backtest que ya existe, en un
  fin de semana de trabajo real, no en un rediseño.

Si A mide una mejora real de RPS en las ligas "main", ahí sí vale la pena
evaluar si el esfuerzo de B (xG real de Understat/FBref, con su
mantenimiento y riesgo de ToS) se justifica para las 5-6 ligas top donde
existe. C solo tendría sentido si B ya demuestra la mejora y se quiere
exprimir más — no antes.

---

## Decisión pendiente (para vos, no auto-decidida)

Esto es justo el tipo de llamada que un plan review normal marcaría como
"user challenge" — implica escoger entre respetar el principio fundacional
del proyecto (gratis/sin key) o ampliar el alcance a scraping/pago por xG
real:

1. **¿Arrancamos con la Estrategia A** (tiros/córners, ya disponible, sin
   nueva fuente) **y medimos con el backtest antes de seguir?** — es lo que
   recomiendo.
2. **¿O querés ir directo a xG real** (Understat/FBref) asumiendo el
   scraping y la cobertura reducida a 5-6 ligas, aunque contradiga el
   "sin API key" del README?

Decime cuál preferís (o si querés que arme el plan detallado de
implementación de A primero) y desde ahí seguimos — sin tocar código todavía,
tal como pediste.
