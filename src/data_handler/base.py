"""
src/data_handler/base.py

Classe base astratta dei DataHandler (`BaseDataHandler`).

Soddisfa `DataHandlerProtocol` di src/core/engine.py (`has_next`, `update_bars`)
e aggiunge `get_latest_bars`, usato da Strategy e RiskManager (es. ATR) per
leggere lo storico.

Garanzia anti look-ahead (strutturale, non solo "per convenzione"):
    - Le barre entrano nello storico interno SOLO nel momento in cui
      `update_bars()` le emette verso l'engine.
    - `get_latest_bars()` legge esclusivamente da quello storico interno,
      quindi non puo' restituire barre non ancora emesse, indipendentemente
      da quanti dati futuri la sottoclasse tenga in memoria o su database.
    - `update_bars()` e `get_latest_bars()` sono `@final`: le sottoclassi
      implementano solo la sorgente dati (`_next_batch`, `has_next`).

Allineamento con config/settings.yaml:
    - symbols   <-> universe.symbols
    - timeframe <-> universe.timeframe
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from typing import Final, TypeAlias, final

import pandas as pd

from src.core.events import MarketEvent

__all__ = ["BAR_COLUMNS", "INDEX_NAME", "BaseDataHandler"]

#: Colonne del DataFrame restituito da `get_latest_bars` (tutte float64).
BAR_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close", "volume")

#: Nome dell'indice (timestamp della barra) del DataFrame restituito.
INDEX_NAME: Final[str] = "timestamp"

# (timestamp, open, high, low, close, volume)
_BarRow: TypeAlias = tuple[datetime, float, float, float, float, float]


class BaseDataHandler(ABC):
    """
    Base astratta per tutte le sorgenti dati (storiche o live).

    Le sottoclassi implementano solo:
        - `has_next()`: esistono altri timestamp da emettere?
        - `_next_batch()`: prossimo timestamp, con le barre di tutti i simboli
          disponibili in quel timestamp.

    Tutto il resto (validazione, storico, accesso ai dati passati) e' gestito
    qui, in modo identico per backtest e live.

    Args:
        symbols: Universo di ticker (da `universe.symbols`).
        timeframe: Granularita' delle barre (da `universe.timeframe`).
    """

    def __init__(self, symbols: Sequence[str], timeframe: str) -> None:
        if not symbols:
            raise ValueError("L'universo di simboli non puo' essere vuoto.")
        if len(set(symbols)) != len(symbols):
            raise ValueError(f"Simboli duplicati nell'universo: {list(symbols)}.")
        if not timeframe.strip():
            raise ValueError("'timeframe' non puo' essere vuoto.")

        self._symbols: tuple[str, ...] = tuple(symbols)
        self._timeframe: str = timeframe
        self._history: dict[str, list[_BarRow]] = {s: [] for s in self._symbols}
        self._current_ts: datetime | None = None

    # ------------------------------------------------------------------ #
    # Proprieta' di sola lettura
    # ------------------------------------------------------------------ #
    @property
    def symbols(self) -> tuple[str, ...]:
        """Universo di simboli gestito dall'handler."""
        return self._symbols

    @property
    def timeframe(self) -> str:
        """Granularita' delle barre (es. '1d')."""
        return self._timeframe

    @property
    def current_timestamp(self) -> datetime | None:
        """Timestamp dell'ultima barra emessa (None prima della prima emissione)."""
        return self._current_ts

    # ------------------------------------------------------------------ #
    # Da implementare nelle sottoclassi
    # ------------------------------------------------------------------ #
    @abstractmethod
    def has_next(self) -> bool:
        """True se esiste almeno un altro timestamp da emettere."""

    @abstractmethod
    def _next_batch(self) -> list[MarketEvent]:
        """
        Restituisce il prossimo timestamp: un MarketEvent per ogni simbolo
        con una barra CHIUSA in quel timestamp.

        Contratto per le sottoclassi:
            - tutti gli eventi hanno lo stesso timestamp;
            - i timestamp sono strettamente crescenti tra una chiamata e l'altra;
            - solo simboli dell'universo e `timeframe` coerente;
            - lista vuota consentita (nessun dato in questo passo).
        """

    # ------------------------------------------------------------------ #
    # Protocollo dell'engine
    # ------------------------------------------------------------------ #
    @final
    def update_bars(self) -> list[MarketEvent]:
        """
        Avanza di UN timestamp: valida il batch, lo registra nello storico e
        lo restituisce all'engine.

        Returns:
            Barre chiuse di tutti i simboli disponibili in quel timestamp
            (lista vuota se non ci sono dati).

        Raises:
            ValueError: se il batch viola il contratto di `_next_batch`.
        """
        batch = self._next_batch()
        if not batch:
            return []

        self._validate_batch(batch)

        for event in batch:
            self._history[event.symbol].append(
                (event.timestamp, event.open, event.high, event.low, event.close, event.volume)
            )
        self._current_ts = batch[0].timestamp
        return batch

    # ------------------------------------------------------------------ #
    # Accesso allo storico (zero look-ahead)
    # ------------------------------------------------------------------ #
    @final
    def get_latest_bars(self, symbol: str, n: int) -> pd.DataFrame:
        """
        Ultime `n` barre di `symbol` gia' emesse, inclusa quella corrente.

        Non puo' mai includere barre successive al `current_timestamp`: lo
        storico contiene solo cio' che `update_bars()` ha gia' consegnato.

        Args:
            symbol: Ticker appartenente all'universo.
            n: Numero di barre richieste (>= 1).

        Returns:
            DataFrame in ordine cronologico crescente, indicizzato per
            timestamp (`INDEX_NAME`), colonne `BAR_COLUMNS` in float64.
            Se sono disponibili meno di `n` barre, restituisce quelle
            disponibili (anche zero righe): il chiamante deve verificare
            `len(df) == n` prima di calcolare indicatori a finestra fissa.
            E' sempre una copia: modificarla non altera lo storico.

        Raises:
            ValueError: se `n < 1`.
            KeyError: se `symbol` non appartiene all'universo.
        """
        if n < 1:
            raise ValueError(f"'n' deve essere >= 1 (ricevuto {n}).")
        try:
            rows = self._history[symbol]
        except KeyError:
            raise KeyError(f"Simbolo '{symbol}' non appartiene all'universo {self._symbols}.") from None

        frame = pd.DataFrame(rows[-n:], columns=[INDEX_NAME, *BAR_COLUMNS])
        frame = frame.set_index(INDEX_NAME)
        return frame.astype("float64")

    # ------------------------------------------------------------------ #
    # Validazione interna
    # ------------------------------------------------------------------ #
    def _validate_batch(self, batch: list[MarketEvent]) -> None:
        """Verifica coerenza del batch: tempo, universo, timeframe, unicita'."""
        timestamp = batch[0].timestamp

        if any(event.timestamp != timestamp for event in batch):
            raise ValueError("Il batch contiene timestamp diversi.")
        if self._current_ts is not None and timestamp <= self._current_ts:
            raise ValueError(
                f"Timestamp non strettamente crescenti: {timestamp} dopo {self._current_ts}."
            )

        seen: set[str] = set()
        for event in batch:
            if event.symbol not in self._history:
                raise ValueError(f"Simbolo '{event.symbol}' fuori universo {self._symbols}.")
            if event.timeframe != self._timeframe:
                raise ValueError(
                    f"[{event.symbol}] timeframe '{event.timeframe}' diverso da "
                    f"quello atteso '{self._timeframe}'."
                )
            if event.symbol in seen:
                raise ValueError(f"Simbolo '{event.symbol}' duplicato nel batch {timestamp}.")
            seen.add(event.symbol)