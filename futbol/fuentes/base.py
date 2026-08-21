"""Estructuras comunes a todas las fuentes de datos."""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class Partido:
    """Un partido con resultado (o pendiente, si los goles son None)."""

    fecha: date
    liga: str                      # codigo interno de liga (ej. "SP1", "ARG")
    liga_nombre: str               # nombre legible (ej. "LaLiga")
    temporada: str
    local: str
    visitante: str
    goles_local: int | None = None
    goles_visitante: int | None = None
    # Cuotas de cierre del mercado (promedio de casas). Sirven de referencia
    # para medir si el modelo aporta algo por encima del mercado.
    cuota_local: float | None = None
    cuota_empate: float | None = None
    cuota_visitante: float | None = None

    @property
    def jugado(self) -> bool:
        return self.goles_local is not None and self.goles_visitante is not None

    @property
    def resultado(self) -> str | None:
        """'H' local gana, 'D' empate, 'A' visitante gana."""
        if not self.jugado:
            return None
        if self.goles_local > self.goles_visitante:
            return "H"
        if self.goles_local == self.goles_visitante:
            return "D"
        return "A"

    @property
    def tiene_cuotas(self) -> bool:
        return None not in (self.cuota_local, self.cuota_empate, self.cuota_visitante)

    def __str__(self) -> str:
        marcador = f"{self.goles_local}-{self.goles_visitante}" if self.jugado else "vs"
        return f"{self.fecha} [{self.liga}] {self.local} {marcador} {self.visitante}"


def normalizar_nombre(nombre: str) -> str:
    """Minusculas, sin acentos ni puntuacion: para comparar nombres de equipos."""
    txt = unicodedata.normalize("NFKD", nombre.strip().lower())
    txt = "".join(c for c in txt if not unicodedata.combining(c))
    txt = re.sub(r"[^a-z0-9 ]+", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()


# Palabras genericas que casi todos los equipos comparten: si dos nombres solo
# coinciden en una de estas, no es evidencia de que sean el mismo equipo.
_GENERICAS = {
    "fc", "cf", "sc", "ac", "afc", "cd", "ca", "sd", "ud", "rc", "club",
    "de", "del", "la", "el", "los", "las", "real", "deportivo", "sporting",
    "athletic", "atletico", "united", "city", "town", "county", "fk", "sk",
}


def _tokens_utiles(normalizado: str) -> set[str]:
    tokens = set(normalizado.split())
    utiles = tokens - _GENERICAS
    return utiles or tokens


def buscar_equipo(consulta: str, equipos: list[str],
                  n: int = 5) -> list[tuple[str, float]]:
    """Busca un equipo por nombre aproximado. Devuelve [(nombre, confianza)].

    El emparejamiento por parecido bruto es peligroso: "Malaga" y "Mallorca"
    se parecen mucho como cadenas y son equipos distintos. Por eso no basta con
    un umbral de similitud; se exige ademas una de estas evidencias:

      * coincidencia exacta tras normalizar,
      * que un nombre contenga al otro ("Rayo Vallecano" / "Vallecano"),
      * que compartan una palabra que no sea generica ("Ath Madrid" /
        "Atl. Madrid" comparten "madrid"),
      * o un parecido altisimo, >= 0.88 ("Espanol" / "Espanyol").

    Sin ninguna de esas, el candidato se descarta aunque se parezca.
    """
    if not equipos:
        return []

    objetivo = normalizar_nombre(consulta)
    if not objetivo:
        return []

    catalogo: dict[str, str] = {}
    for e in equipos:
        catalogo.setdefault(normalizar_nombre(e), e)

    if objetivo in catalogo:
        return [(catalogo[objetivo], 1.0)]

    tokens_obj = _tokens_utiles(objetivo)
    candidatos: list[tuple[str, float]] = []

    for norm, original in catalogo.items():
        ratio = difflib.SequenceMatcher(None, objetivo, norm).ratio()
        tokens_cand = _tokens_utiles(norm)
        comunes = tokens_obj & tokens_cand

        contenido = objetivo in norm or norm in objetivo
        # Contener al otro solo vale si lo contenido es sustancial: "as" dentro
        # de "las palmas" no dice nada.
        if contenido and min(len(objetivo), len(norm)) < 4:
            contenido = False

        if not (contenido or comunes or ratio >= 0.88):
            continue

        confianza = ratio
        if contenido:
            confianza = max(confianza, 0.80)
        if comunes:
            confianza = max(
                confianza,
                0.70 + 0.25 * len(comunes) / max(len(tokens_obj), 1),
            )
        candidatos.append((original, min(confianza, 0.99)))

    candidatos.sort(key=lambda x: (-x[1], x[0]))
    return candidatos[:n]


def equipos_de(partidos: list[Partido]) -> list[str]:
    """Nombres unicos de equipo (local + visitante) en una lista de partidos."""
    return sorted({p.local for p in partidos} | {p.visitante for p in partidos})
