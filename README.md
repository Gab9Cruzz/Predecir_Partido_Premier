# PredicirUnPartido

Predice el resultado y la probabilidad de victoria de un partido de **Premier
League** con un modelo **Dixon–Coles** ajustado por máxima verosimilitud,
sobre una base de datos local en SQLite que vos mismo controlás y
actualizás.

```
python predecir.py partido --local Arsenal --visitante Chelsea
```

```
====================================================================
  Arsenal  vs  Chelsea
====================================================================

Goles esperados : 1.92 - 0.87
Marcador top    : 2-0 (11.3%)

Resultado                probabilidad   cuota justa
----------------------------------------------------
Gana Arsenal              61.5%      1.63  #####################
Empate                    22.5%      4.44  ########
Gana Chelsea               16.0%      6.26  #####
```

---

## 1. Alcance: solo Premier League

El proyecto ya no cubre 38 ligas. Predice **una sola competición, Premier
League**, sobre una base de datos local en vez de descargar CSV en cada
corrida.

- La base (`BD_SQLITE/futbol_predicciones.db`) tiene Premier League y
  Championship, 2000-2026: 23.950 partidos jugados, resultado completo.
- El modelo se entrena con **ambas** divisiones combinadas, pero solo
  predice y se valida sobre Premier League. Por qué Championship: sin su
  historial, un equipo recién ascendido llega cada temporada con cero
  partidos previos y el modelo no tiene con qué estimarlo. Con Championship
  cargado, el grafo de partidos queda conectado y el modelo separa el nivel
  de cada división solo, sin configuración manual.
- Validado contra el backtest walk-forward (ver §6): el puente a la base
  local funciona igual o mejor que el pipeline CSV que reemplazó.

---

## 2. La base de datos local

Vive en `BD_SQLITE/futbol_predicciones.db`, SQLite (cero dependencias
nuevas, es un archivo, no un servicio que mantener). Cuatro tablas:

```
equipos               catálogo: nombre crudo del CSV, nombre corto oficial,
                       código de 3 letras, liga actual
partidos               fecha, temporada, liga (E0/E1), goles, equipos, fuente
estadisticas_partido   tiros, tiros a puerta, corners, faltas, tarjetas
                       (posesión: columna lista, sin dato disponible todavía)
cuotas_cierre          mercados 1x2 / over_under / hándicap asiático,
                       formato largo (una fila por partido y mercado)
```

### Scripts de mantenimiento (`BD_SQLITE/`)

| script | para qué |
|---|---|
| `init_db.py` | crea el esquema (idempotente, `CREATE TABLE IF NOT EXISTS`) |
| `cargar_datos.py` | descarga el histórico completo de football-data.co.uk (E0 + E1) y lo carga a la base. Reintentable: una fila con error no tira la temporada entera |
| `actualizar_resultados.py` | trae la jornada más reciente y hace UPSERT — completa resultados de partidos que estaban pendientes, sin duplicar nada |
| `equipos_premier.py` | catálogo de 44 equipos (nombre oficial, nombre corto, código) usado para normalizar nombres, tanto en la ingesta como al resolver lo que escribís en `predecir.py partido` |

Los scripts de descarga manejan el caso de temporada todavía no publicada
(football-data.co.uk responde HTTP 300 con una página de error en vez de
404): lo detectan y avisan, no crashean.

```bash
cd BD_SQLITE
python init_db.py              # una vez, crea el esquema
python cargar_datos.py         # carga histórico completo (tarda: 26 temporadas)
python actualizar_resultados.py   # corré esto seguido: resultados Y próximos partidos
```

`actualizar_resultados.py` hace dos cosas cada vez que corre: trae los
resultados de la jornada más reciente (completa partidos que estaban
pendientes) y trae el calendario de próximos partidos con las cuotas
actuales del mercado. Es la única parte del proyecto que todavía habla con
football-data.co.uk — todo lo demás (`predecir.py` completo) solo lee la
base local.

---

## 3. Instalación

```bash
pip install -r requirements.txt
```

`requests`, `numpy`, `scipy` para el modelo; `pandas` para los scripts de
`BD_SQLITE/`. Sin API keys, sin `.env`, sin cuentas.

---

## 4. Uso del predictor

### Predecir un partido

```bash
python predecir.py partido --local Arsenal --visitante Chelsea
```

Los nombres no tienen que ser exactos. Dos caminos, en este orden:
1. El catálogo de `equipos_premier.py` — reconoce nombre oficial, nombre
   corto y el código de 3 letras (`"Manchester City"`, `"Man City"`, `"MCI"`
   resuelven igual).
2. Si no matchea ahí, cae al emparejamiento aproximado de `base.py` —
   típos como `"Arsnal"` o abreviaciones como `"spurs"` también resuelven.

Si un nombre no existe en los datos, no se inventa uno parecido: avisa.

Opciones útiles:

| opción | para qué |
|---|---|
| `--cuotas 1.95 3.90 3.60` | compara tus probabilidades con las del mercado y calcula el EV |
| `--neutral` | campo neutral: reparte la ventaja de local |
| `--json` | salida en JSON para usarla desde otro programa |

### Ver los equipos y la fuerza estimada

```bash
python predecir.py equipos
python predecir.py ratings
```

`ratings` no es la tabla de posiciones: es la fuerza real estimada, ya
descontada la calidad de los rivales a los que se enfrentó cada equipo.

### Validar el modelo

```bash
python predecir.py backtest
python predecir.py optimizar     # busca el mejor xi
```

### Próximos partidos

```bash
python predecir.py proximos
```

Lee de la base local igual que todo lo demás — cero descargas en vivo desde
`predecir.py`. Los partidos que todavía no se jugaron quedan en `partidos`
con `goles_local`/`goles_visitante` en `NULL`; cuando se juegan,
`actualizar_resultados.py` completa esa misma fila con el resultado real
(mismo `UPSERT`, no se duplica nada). Si no corriste
`actualizar_resultados.py` últimamente, este comando avisa y no muestra
nada — no se inventa partidos.

---

## 5. Cómo funciona el modelo

Se estiman a la vez, por máxima verosimilitud, **un parámetro de ataque y
uno de defensa para cada equipo**:

```
goles_local  ~ Poisson(exp(mu + ataque_local  - defensa_visitante + ventaja_local))
goles_visita ~ Poisson(exp(mu + ataque_visitante - defensa_local))
```

Como todos los equipos aparecen en la misma ecuación, la fuerza del rival
queda descontada automáticamente. Encima de eso:

1. **Corrección τ de Dixon–Coles.** Dos Poisson independientes subestiman
   los 0-0 y 1-1 y sobreestiman los 1-0 y 0-1. El parámetro ρ corrige
   exactamente esas cuatro casillas — donde más se juegan los empates.

2. **Ponderación temporal.** Cada partido pesa
   `exp(-xi · días_de_antigüedad)`. El valor por defecto (`xi = 0.0018`,
   semivida ≈ 385 días) salió de minimizar el RPS del backtest.
   Recalculable con `predecir.py optimizar`.

3. **Regularización L2.** Encoge hacia la media a los equipos con pocos
   partidos — un recién ascendido no se lleva un rating extremo por una
   racha corta de 3 jornadas.

El ajuste usa gradiente analítico con L-BFGS-B: se ajusta en menos de una
décima de segundo, lo que hace viable reajustar el modelo cientos de veces
durante un backtest.

Referencia: Dixon, M.J. y Coles, S.G. (1997), *Modelling Association
Football Scores and Inefficiencies in the Football Betting Market*, Applied
Statistics 46(2), 265–280.

---

## 6. ¿Funciona? Resultados medidos

Backtest walk-forward vía `bd_local.py` (Premier League + Championship
2000-2026 para entrenar, solo Premier League puntuada: entrenar con las dos
divisiones le da conectividad al grafo de partidos, pero el RPS solo tiene
sentido medido sobre lo que efectivamente se predice — mezclarlas en la
métrica final habría dado un número no comparable con nada):

| partidos evaluados | RPS modelo | RPS mercado | diferencia | acierto modelo | acierto mercado |
|---|---|---|---|---|---|
| 9.368 | 0.1986 | 0.1942 | +0.0044 | 53.4% | 54.3% |

*RPS = Ranked Probability Score. Más bajo es mejor. El azar puro está en
~0.235.*

- **El modelo bate al azar con holgura.**
- **Está bien calibrado**: cuando dice 30%, pasa aproximadamente el 30% de
  las veces.
- **Está a la altura del mercado, algo por detrás** (+0.0044 de RPS, -0.9
  puntos de acierto). Es lo esperable: el mercado incorpora alineaciones,
  lesiones y rotaciones que este modelo no ve. Es mejor que el resultado
  del pipeline CSV anterior (+0.0079 de RPS sobre 6 temporadas) — esperable
  también, porque ahora entrena con 26 temporadas y con la conectividad que
  da Championship.

---

## 7. Limitaciones (léelas)

**No sirve para ganar dinero apostando.** El modelo pierde contra el
mercado de forma sistemática (ver §6), y encima hay que pagar el margen de
la casa. Cualquier discrepancia grande entre el modelo y las cuotas es
mucho más probable que sea un error del modelo que una oportunidad real.

**Lo que el modelo no ve:** alineaciones, lesiones, sanciones, si el equipo
tiene partido de Champions o de copa entre semana, cambios de entrenador,
si el partido no vale nada porque ya está todo decidido, y el clima. Todo
eso sí lo ve el mercado, y por eso gana. La idea a futuro es capturar esto
como *feature* de rotación/fatiga en una capa de ML (no cubierta todavía),
no meterlo como partidos de entrenamiento adicionales.

**Equipos con pocos partidos.** Un recién ascendido tiene el rating muy
encogido hacia la media (la regularización lo hace a propósito); el
programa lo marca y avisa. Ahí las predicciones son flojas.

---

## 8. Estructura del proyecto

```
predecir.py                     interfaz de línea de comandos -- solo lee la base local
BD_SQLITE/
  init_db.py                    crea el esquema
  cargar_datos.py               ingesta histórica completa (E0 + E1)
  actualizar_resultados.py      sincroniza resultados Y próximos partidos (UPSERT)
  equipos_premier.py            catálogo de equipos: nombre oficial/corto/código
  futbol_predicciones.db        la base en sí (no versionada, ver .gitignore)
futbol/
  fuentes/
    base.py                     Partido + emparejamiento aproximado de nombres
    bd_local.py                  lee BD_SQLITE/ y arma la lista de Partido para el modelo
  modelo/
    dixon_coles.py               el modelo: ajuste, predicción y ratings
    mercado.py                   cuotas, probabilidades implícitas, Kelly, métricas
  backtest.py                   validación walk-forward y optimización de xi
```

No queda ningún módulo que descargue CSV en vivo dentro de `predecir.py`/
`futbol/` -- eso es trabajo exclusivo de los scripts de `BD_SQLITE/`, que se
corren aparte y por separado del predictor.

Como biblioteca:

```python
from futbol.fuentes import bd_local
from futbol.modelo import DixonColes

partidos = bd_local.cargar_para_ajuste()   # Premier League + Championship
modelo = DixonColes().ajustar(partidos)

pred = modelo.predecir("Arsenal", "Chelsea", liga="E0")
print(pred.prob_local, pred.prob_empate, pred.prob_visitante)
print(pred.prob_mas_de(2.5), pred.prob_ambos_marcan)
print(pred.marcadores_probables[:3])
```
