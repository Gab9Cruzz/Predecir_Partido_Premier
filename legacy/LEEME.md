# Código anterior (no se usa)

Estos dos archivos son la version original del proyecto, basada en
**API-Football (api-sports.io)**. Se conservan solo como referencia.

Por que se sustituyeron:

1. **La API no servia.** En el plan gratuito solo devuelve temporadas antiguas,
   asi que era imposible predecir partidos actuales sin pagar.

2. **El modelo tenia un fallo de fondo.** Calculaba los goles esperados como el
   promedio de goles a favor del local y en contra del visitante. Eso no corrige
   por la calidad del rival: un equipo que jugo contra los ultimos de la tabla
   parece mucho mejor de lo que es.

3. **`tune_pesos_por_brier()` usaba datos del futuro.** Para "predecir" partidos
   ya jugados consultaba la clasificacion ACTUAL de esa misma temporada, que ya
   incorpora el resultado del partido a predecir. Cualquier metrica salida de
   ahi estaba inflada. Ademas hacia una llamada a la API por cada partido y por
   cada iteracion del optimizador: con 200 partidos y 200 iteraciones son 40.000
   peticiones, muy por encima del limite diario de 100 del plan gratuito.

4. **La API key estaba escrita en el codigo.** Sigue visible en estos archivos.
   Si esa cuenta te importa, revocala y genera otra.

El proyecto actual esta en la raiz del repositorio. Ver `README.md`.
