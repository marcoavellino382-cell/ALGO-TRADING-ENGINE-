"""
src/strategy/base.py

Classe base astratta delle strategie (`BaseStrategy`).

Soddisfa `StrategyProtocol` di src/core/engine.py:
    - on_market_event(event) -> list[SignalEvent]
    - on_fill(fill) -> None

Responsabilita' della Strategy (e solo queste):
    - leggere lo storico tramite `BaseDataHandler.get_latest_bars` (mai dati futuri);
    - esprimere intenti direzionali (`SignalEvent`: LONG / SHORT / EXIT).
    NON decide quantita', stop, esposizione o leva: e' compito del RiskManager.

Garanzie anti look-ahead:
    - `on_market_event` e' un template method `@final`: la sottoclasse implementa
      solo `generate_signals`, quindi ogni segnale passa dalla validazione.
    - `get_history` verifica che l'ultima barra restituita non sia successiva
      all'istante `as_of` richiesto (difesa in profondita' oltre a quella del
      DataHandler) e solleva `LookAheadError` altrimenti.
    - Ogni segnale deve avere `timestamp == event.timestamp`: un segnale a t
      viene eseguito dall'engine al massimo a t+1.

Convenzione sui segnali (per tutte le strategie):
    - LONG/SHORT solo quando la strategia e' flat sul simbolo (ingresso);
    - EXIT solo quando e' in posizione. Lo stato di posizione e' ricostruito dai
      FillEvent, quindi riflette solo cio' che e' stato davvero eseguito.

Allineamento con config/settings.yaml:
    - `strategy_id` coincide con `strategy.name` (es. "mean_reversion");
    - `ensure_warmup()` verifica `backtest.warmup_bars >= required_history`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import datetime
from typing import Final, final

import pandas as pd

from src.core.events import (
    FillEvent,
    MarketEvent,
    SignalDirection,
    SignalEvent,
)
from src.data_handler.base import BaseDataHandler

__all__ = ["LookAheadError", "BaseStrategy"]

#: Soglia sotto la quale una quantita' netta e' considerata zero (azioni frazionarie).
_EPS: Final[float] = 1e-9


class LookAheadError(RuntimeError):
    """Una strategia ha ottenuto o prodotto informazioni dal futuro."""


class BaseStrategy(ABC):
    """
    Base astratta per tutte le strategie.

    Le sottoclassi implementano solo:
        - `required_history`: barre di storico necessarie (barra corrente inclusa);
        - `generate_signals(event)`: logica del segnale sulla barra chiusa `event`.

    Args:
        strategy_id: Identificativo della strategia (`strategy.name` in YAML).
        data_handler: DataHandler da cui leggere lo storico (sola lettura).
    """

    def __init__(self, strategy_id: str, data_handler: BaseDataHandler) -> None:
        if not strategy_id.strip():
            raise ValueError("'strategy_id' non puo' essere vuoto.")
        self._strategy_id: str = strategy_id
        self._data_handler: BaseDataHandler = data_handler
        self._positions: dict[str, float] = {}   # quantita' netta per simbolo (da fill)
        self._bars_held: dict[str, int] = {}     # barre dall'ingresso, per simbolo

    # ------------------------------------------------------------------ #
    # Da implementare nelle sottoclassi
    # ------------------------------------------------------------------ #
    @property
    @abstractmethod
    def required_history(self) -> int:
        """
        Numero minimo di barre (barra corrente inclusa) necessarie per calcolare
        tutti gli indicatori della strategia (es. max tra finestre e filtri).
        """

    @abstractmethod
    def generate_signals(self, event: MarketEvent) -> list[SignalEvent]:
        """
        Produce zero o piu' segnali sulla barra chiusa `event`.

        Usa `get_history(event.symbol, n, event.timestamp)` per i dati storici e
        `make_signal(event, ...)` per costruire i segnali.

        Returns:
            Lista di SignalEvent (anche vuota). Mai quantita' o sizing.
        """

    # ------------------------------------------------------------------ #
    # Protocollo dell'engine
    # ------------------------------------------------------------------ #
    @final
    def on_market_event(self, event: MarketEvent) -> list[SignalEvent]:
        """
        Elabora una barra chiusa: aggiorna il conteggio di holding, delega a
        `generate_signals` e valida i segnali prodotti.

        Raises:
            LookAheadError: se un segnale non e' allineato alla barra corrente.
            ValueError: se un segnale ha simbolo o strategy_id incoerenti.
        """
        symbol = event.symbol
        if symbol in self._positions:
            self._bars_held[symbol] += 1

        signals = self.generate_signals(event)
        for signal in signals:
            self._validate_signal(event, signal)
        return list(signals)

    def on_fill(self, fill: FillEvent) -> None:
        """
        Aggiorna posizione netta e holding period a seguito di un fill.

        Ignora i fill di altre strategie. Le sottoclassi possono estenderlo
        chiamando `super().on_fill(fill)`.
        """
        if fill.strategy_id != self._strategy_id:
            return

        symbol = fill.symbol
        previous = self._positions.get(symbol, 0.0)
        updated = previous + fill.signed_quantity

        if abs(updated) < _EPS:
            self._positions.pop(symbol, None)
            self._bars_held.pop(symbol, None)
            return

        self._positions[symbol] = updated
        if abs(previous) < _EPS:  # nuovo ingresso
            self._bars_held[symbol] = 0

    # ------------------------------------------------------------------ #
    # Accesso sicuro ai dati
    # ------------------------------------------------------------------ #
    @final
    def get_history(self, symbol: str, n: int, as_of: datetime) -> pd.DataFrame | None:
        """
        Ultime `n` barre di `symbol` note all'istante `as_of`.

        Args:
            symbol: Ticker dell'universo.
            n: Numero di barre richieste (>= 1).
            as_of: Istante di riferimento, normalmente `event.timestamp`.

        Returns:
            DataFrame (indice timestamp crescente; colonne open, high, low, close,
            volume) con ESATTAMENTE `n` righe, oppure `None` se lo storico
            disponibile e' insufficiente: in quel caso la strategia deve
            astenersi dal produrre segnali.

        Raises:
            LookAheadError: se l'ultima barra e' successiva ad `as_of`.
        """
        bars = self._data_handler.get_latest_bars(symbol, n)
        if bars.empty:
            return None
        if bars.index[-1] > as_of:
            raise LookAheadError(
                f"[{symbol}] barra del {bars.index[-1]} successiva a as_of={as_of}."
            )
        if len(bars) < n:
            return None
        return bars

    # ------------------------------------------------------------------ #
    # Costruzione segnali
    # ------------------------------------------------------------------ #
    def make_signal(
        self,
        event: MarketEvent,
        direction: SignalDirection,
        strength: float = 1.0,
        metadata: Mapping[str, float] | None = None,
    ) -> SignalEvent:
        """
        Costruisce un SignalEvent coerente con la barra `event`.

        Imposta automaticamente timestamp, simbolo, strategy_id e
        `reference_price` (close della barra), evitando errori di allineamento.

        Args:
            event: Barra chiusa che ha generato il segnale.
            direction: LONG, SHORT o EXIT.
            strength: Convinzione in [0, 1].
            metadata: Diagnostica numerica (es. {"zscore": -2.3}).
        """
        return SignalEvent(
            timestamp=event.timestamp,
            symbol=event.symbol,
            direction=direction,
            strategy_id=self._strategy_id,
            strength=strength,
            reference_price=event.close,
            metadata=dict(metadata) if metadata else {},
        )

    # ------------------------------------------------------------------ #
    # Stato di posizione (ricostruito dai fill)
    # ------------------------------------------------------------------ #
    @property
    def strategy_id(self) -> str:
        """Identificativo della strategia (`strategy.name`)."""
        return self._strategy_id

    @property
    def positions(self) -> dict[str, float]:
        """Copia delle posizioni nette per simbolo (solo simboli non flat)."""
        return dict(self._positions)

    def position(self, symbol: str) -> float:
        """Quantita' netta del simbolo (0.0 se flat; negativa se short)."""
        return self._positions.get(symbol, 0.0)

    def is_flat(self, symbol: str) -> bool:
        """True se la strategia non ha posizione sul simbolo."""
        return symbol not in self._positions

    def is_long(self, symbol: str) -> bool:
        """True se la posizione netta sul simbolo e' lunga."""
        return self.position(symbol) > 0.0

    def is_short(self, symbol: str) -> bool:
        """True se la posizione netta sul simbolo e' corta."""
        return self.position(symbol) < 0.0

    def bars_held(self, symbol: str) -> int:
        """
        Barre trascorse dall'ingresso, inclusa la barra di ingresso (0 se flat).

        Il fill di ingresso avviene all'inizio della barra t+1: alla chiusura di
        quella barra il valore e' 1.
        """
        return self._bars_held.get(symbol, 0)

    # ------------------------------------------------------------------ #
    # Verifica di coerenza con la configurazione
    # ------------------------------------------------------------------ #
    def ensure_warmup(self, warmup_bars: int) -> None:
        """
        Verifica che `backtest.warmup_bars` copra lo storico richiesto.

        Da chiamare in `main.py` prima di avviare l'engine.

        Raises:
            ValueError: se `warmup_bars < required_history`.
        """
        if warmup_bars < self.required_history:
            raise ValueError(
                f"warmup_bars ({warmup_bars}) inferiore allo storico richiesto dalla "
                f"strategia '{self._strategy_id}' ({self.required_history}): "
                "aumenta backtest.warmup_bars in settings.yaml."
            )

    # ------------------------------------------------------------------ #
    # Validazione interna
    # ------------------------------------------------------------------ #
    def _validate_signal(self, event: MarketEvent, signal: SignalEvent) -> None:
        """Controlla che il segnale sia allineato alla barra e a questa strategia."""
        if signal.timestamp != event.timestamp:
            raise LookAheadError(
                f"Segnale con timestamp {signal.timestamp} diverso dalla barra "
                f"corrente {event.timestamp}."
            )
        if signal.symbol != event.symbol:
            raise ValueError(
                f"Segnale su '{signal.symbol}' emesso durante la barra di '{event.symbol}'."
            )
        if signal.strategy_id != self._strategy_id:
            raise ValueError(
                f"strategy_id '{signal.strategy_id}' diverso da '{self._strategy_id}'."
            )