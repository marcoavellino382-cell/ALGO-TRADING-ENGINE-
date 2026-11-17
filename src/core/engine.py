"""
src/core/engine.py

Event loop principale (`TradingEngine`) del backtest.

Per ogni timestamp t (un "batch" di MarketEvent, uno per simbolo) l'engine
esegue SEMPRE le stesse tre fasi, in quest'ordine:

    Fase 1 - Settlement:  gli ordini creati in t-1 vengono eseguiti sulla barra t
                          (fill_timing = next_open | next_close) -> FillEvent.
    Fase 2 - Mark-to-market: il RiskManager valorizza il portafoglio ai prezzi
                          di chiusura di t, registra lo snapshot e valuta il
                          kill-switch sul drawdown.
    Fase 3 - Segnali:     la Strategy vede la barra t CHIUSA -> SignalEvent ->
                          RiskManager -> OrderEvent -> ExecutionHandler (pending).
                          Gli ordini verranno eseguiti SOLO dalla barra t+1.

Il look-ahead bias e' quindi escluso per costruzione: un ordine non puo' mai
essere eseguito nella stessa barra che ha generato il segnale.

SIGNAL -> ORDER -> FILL transitano nella coda eventi (`deque`). I MarketEvent
sono invece consumati in batch per timestamp: l'ordine tra simboli conta
(es. i fill di tutti i simboli devono essere registrati PRIMA che il
RiskManager valuti `max_open_positions` o `max_gross_exposure`).

Allineamento con config/settings.yaml (sezione `backtest`):
    - warmup_bars  -> nessun segnale finche' un simbolo non ha accumulato
                      almeno `warmup_bars` barre.
    - fill_timing  -> validato qui; il prezzo di fill (open/close di t+1) e'
                      applicato dall'ExecutionHandler.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

from src.core.events import (
    AnyEvent,
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
)

__all__ = [
    "FillTiming",
    "EngineConfig",
    "PortfolioSnapshot",
    "EngineResult",
    "DataHandlerProtocol",
    "StrategyProtocol",
    "RiskManagerProtocol",
    "ExecutionHandlerProtocol",
    "TradingEngine",
]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configurazione
# --------------------------------------------------------------------------- #
class FillTiming(StrEnum):
    """Momento di esecuzione degli ordini. Coincide con `backtest.fill_timing`."""

    NEXT_OPEN = "next_open"    # open della barra t+1
    NEXT_CLOSE = "next_close"  # close della barra t+1


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """
    Parametri dell'engine, letti dalla sezione `backtest` di settings.yaml.

    Attributes:
        warmup_bars: Barre minime per simbolo prima di accettare segnali.
        fill_timing: Quando vengono eseguiti gli ordini (sempre dalla barra t+1).
    """

    warmup_bars: int
    fill_timing: FillTiming = FillTiming.NEXT_OPEN

    def __post_init__(self) -> None:
        if self.warmup_bars < 0:
            raise ValueError(f"'warmup_bars' deve essere >= 0 (ricevuto {self.warmup_bars}).")

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> EngineConfig:
        """
        Costruisce la configurazione dal dizionario completo di settings.yaml.

        Args:
            settings: Contenuto di settings.yaml gia' caricato (`yaml.safe_load`).
        """
        backtest = settings["backtest"]
        return cls(
            warmup_bars=int(backtest["warmup_bars"]),
            fill_timing=FillTiming(backtest["fill_timing"]),
        )


# --------------------------------------------------------------------------- #
# Strutture dati di output
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """
    Fotografia del portafoglio alla chiusura di una barra.

    Prodotta dal RiskManager (che possiede lo stato di portafoglio) e raccolta
    dall'engine per costruire l'equity curve.

    Attributes:
        timestamp: Timestamp della barra.
        cash: Liquidita' in valuta base.
        equity: Valore totale = cash + valore di mercato delle posizioni.
        gross_exposure: Valore assoluto delle posizioni / equity (1.0 = 100%).
        drawdown: Drawdown corrente dal massimo storico dell'equity, frazione >= 0
            (0.15 = -15%).
        positions: Quantita' per simbolo (copia, non una vista mutabile).
    """

    timestamp: datetime
    cash: float
    equity: float
    gross_exposure: float
    drawdown: float
    positions: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class EngineResult:
    """
    Risultato di un `TradingEngine.run()`, consumato da analytics e dashboard.

    Attributes:
        snapshots: Equity curve, uno snapshot per timestamp elaborato.
        fills: Tutte le esecuzioni (con commissioni e slippage).
        orders: Tutti gli ordini inviati all'ExecutionHandler.
        signals: Tutti i segnali accettati (post-warmup).
        bars_processed: Numero di timestamp elaborati.
        halted: True se il kill-switch ha fermato il backtest.
        halt_timestamp: Timestamp in cui il kill-switch e' scattato, se scattato.
    """

    snapshots: tuple[PortfolioSnapshot, ...]
    fills: tuple[FillEvent, ...]
    orders: tuple[OrderEvent, ...]
    signals: tuple[SignalEvent, ...]
    bars_processed: int
    halted: bool
    halt_timestamp: datetime | None = None


# --------------------------------------------------------------------------- #
# Contratti dei componenti (typing.Protocol)
# --------------------------------------------------------------------------- #
class DataHandlerProtocol(Protocol):
    """Contratto minimo del DataHandler usato dall'engine."""

    def has_next(self) -> bool:
        """True se esiste almeno un altro timestamp da emettere."""
        ...

    def update_bars(self) -> list[MarketEvent]:
        """
        Avanza di UN timestamp e restituisce le barre chiuse di tutti i simboli
        disponibili in quel timestamp. Dopo la chiamata, il DataHandler non deve
        mai esporre dati con timestamp successivo a quello restituito.
        """
        ...


class StrategyProtocol(Protocol):
    """Contratto della Strategy: produce SOLO intenti direzionali."""

    def on_market_event(self, event: MarketEvent) -> list[SignalEvent]:
        """Elabora una barra chiusa e restituisce zero o piu' segnali."""
        ...

    def on_fill(self, fill: FillEvent) -> None:
        """Notifica di un fill (per tracciare holding period / stato interno)."""
        ...


class RiskManagerProtocol(Protocol):
    """Contratto del RiskManager: possiede capitale, posizioni e rischio."""

    @property
    def kill_switch_triggered(self) -> bool:
        """True se il drawdown ha raggiunto `risk.max_drawdown_pct`."""
        ...

    def on_market_event(self, event: MarketEvent) -> None:
        """Aggiorna i prezzi di mercato (mark-to-market) del simbolo."""
        ...

    def on_fill(self, fill: FillEvent) -> None:
        """Aggiorna cassa e posizioni a seguito di un'esecuzione."""
        ...

    def on_signal(self, signal: SignalEvent) -> list[OrderEvent]:
        """Trasforma un segnale in ordini dimensionati (o in nessun ordine)."""
        ...

    def snapshot(self, timestamp: datetime) -> PortfolioSnapshot:
        """Registra e restituisce lo stato del portafoglio; aggiorna il kill-switch."""
        ...

    def liquidation_orders(self, timestamp: datetime) -> list[OrderEvent]:
        """Ordini di chiusura totale (vuota se `kill_switch_action = halt_only`)."""
        ...


class ExecutionHandlerProtocol(Protocol):
    """Contratto dell'ExecutionHandler (simulato o broker reale)."""

    def submit_order(self, order: OrderEvent) -> None:
        """Accoda l'ordine. NON deve eseguirlo nella barra corrente."""
        ...

    def on_market_event(self, event: MarketEvent) -> list[FillEvent]:
        """
        Prova a eseguire gli ordini pendenti sulla nuova barra. Esegue solo
        ordini con `signal_timestamp < event.timestamp`, applicando
        slippage e commissioni.
        """
        ...

    def has_pending_orders(self) -> bool:
        """True se esistono ordini non ancora eseguiti o cancellati."""
        ...


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class TradingEngine:
    """
    Orchestratore event-driven: collega DataHandler, Strategy, RiskManager ed
    ExecutionHandler rispettando `warmup_bars` e `fill_timing`.

    Esempio:
        >>> engine = TradingEngine(config, data_handler, strategy, risk, execution)
        >>> result = engine.run()
    """

    def __init__(
        self,
        config: EngineConfig,
        data_handler: DataHandlerProtocol,
        strategy: StrategyProtocol,
        risk_manager: RiskManagerProtocol,
        execution_handler: ExecutionHandlerProtocol,
    ) -> None:
        self._config = config
        self._data = data_handler
        self._strategy = strategy
        self._risk = risk_manager
        self._execution = execution_handler

        self._queue: deque[AnyEvent] = deque()
        self._handlers: dict[EventType, Callable[[Any], None]] = {
            EventType.SIGNAL: self._handle_signal,
            EventType.ORDER: self._handle_order,
            EventType.FILL: self._handle_fill,
        }

        self._bars_seen: defaultdict[str, int] = defaultdict(int)
        self._current_ts: datetime | None = None
        self._bars_processed = 0
        self._has_run = False

        self._halted = False
        self._halt_ts: datetime | None = None

        self._snapshots: list[PortfolioSnapshot] = []
        self._signals: list[SignalEvent] = []
        self._orders: list[OrderEvent] = []
        self._fills: list[FillEvent] = []

    # ------------------------------------------------------------------ #
    # API pubblica
    # ------------------------------------------------------------------ #
    def run(self) -> EngineResult:
        """
        Esegue l'intero backtest e restituisce il risultato.

        Raises:
            RuntimeError: se l'engine e' gia' stato eseguito (usa una nuova istanza).
        """
        if self._has_run:
            raise RuntimeError("TradingEngine gia' eseguito: crea una nuova istanza.")
        self._has_run = True

        logger.info(
            "Avvio backtest (warmup_bars=%d, fill_timing=%s)",
            self._config.warmup_bars,
            self._config.fill_timing.value,
        )

        while self._data.has_next():
            batch = self._data.update_bars()
            if not batch:
                continue
            self._process_timestamp(batch)

            # Dopo il kill-switch si prosegue solo finche' le liquidazioni
            # pendenti non sono state eseguite.
            if self._halted and not self._execution.has_pending_orders():
                break

        if self._execution.has_pending_orders():
            logger.warning("Fine dati con ordini pendenti non eseguiti: vengono scartati.")

        logger.info(
            "Backtest concluso: %d barre, %d segnali, %d ordini, %d fill, halted=%s",
            self._bars_processed,
            len(self._signals),
            len(self._orders),
            len(self._fills),
            self._halted,
        )
        return EngineResult(
            snapshots=tuple(self._snapshots),
            fills=tuple(self._fills),
            orders=tuple(self._orders),
            signals=tuple(self._signals),
            bars_processed=self._bars_processed,
            halted=self._halted,
            halt_timestamp=self._halt_ts,
        )

    # ------------------------------------------------------------------ #
    # Elaborazione di un timestamp
    # ------------------------------------------------------------------ #
    def _process_timestamp(self, batch: list[MarketEvent]) -> None:
        """Esegue le 3 fasi (settlement, mark-to-market, segnali) per un timestamp."""
        timestamp = self._validate_batch(batch)
        self._current_ts = timestamp
        self._bars_processed += 1

        # Fase 1 - Settlement: ordini di t-1 eseguiti sulla barra t.
        for market_event in batch:
            self._queue.extend(self._execution.on_market_event(market_event))
        self._drain_queue()

        # Fase 2 - Mark-to-market ai close di t, snapshot e kill-switch.
        for market_event in batch:
            self._bars_seen[market_event.symbol] += 1
            self._risk.on_market_event(market_event)
        self._snapshots.append(self._risk.snapshot(timestamp))

        if not self._halted and self._risk.kill_switch_triggered:
            self._trigger_kill_switch(timestamp)

        # Fase 3 - Segnali sulla barra t chiusa (eseguiti al massimo da t+1).
        if self._halted:
            return
        for market_event in batch:
            if self._bars_seen[market_event.symbol] < self._config.warmup_bars:
                continue  # warmup: la Strategy non viene nemmeno interpellata
            self._queue.extend(self._strategy.on_market_event(market_event))
        self._drain_queue()

    def _validate_batch(self, batch: list[MarketEvent]) -> datetime:
        """Verifica che il batch sia omogeneo e che il tempo avanzi sempre."""
        timestamp = batch[0].timestamp
        if any(event.timestamp != timestamp for event in batch):
            raise ValueError("Il batch del DataHandler contiene timestamp diversi.")
        if self._current_ts is not None and timestamp <= self._current_ts:
            raise ValueError(
                f"Timestamp non strettamente crescenti: {timestamp} dopo {self._current_ts}."
            )
        return timestamp

    def _trigger_kill_switch(self, timestamp: datetime) -> None:
        """Ferma le nuove entrate e accoda le eventuali liquidazioni (fill da t+1)."""
        self._halted = True
        self._halt_ts = timestamp
        logger.warning("KILL-SWITCH attivato il %s: trading sospeso.", timestamp)
        self._queue.extend(self._risk.liquidation_orders(timestamp))
        self._drain_queue()

    # ------------------------------------------------------------------ #
    # Coda eventi e handler
    # ------------------------------------------------------------------ #
    def _drain_queue(self) -> None:
        """Svuota la coda instradando ogni evento al proprio handler."""
        while self._queue:
            event = self._queue.popleft()
            handler = self._handlers.get(event.event_type)
            if handler is None:
                raise TypeError(
                    f"Evento {event.event_type.value} non ammesso nella coda "
                    "(i MarketEvent sono gestiti in batch)."
                )
            handler(event)

    def _handle_signal(self, signal: SignalEvent) -> None:
        """SIGNAL -> RiskManager -> OrderEvent (in coda)."""
        if signal.timestamp != self._now:
            raise ValueError(
                f"Look-ahead: segnale con timestamp {signal.timestamp} "
                f"diverso dalla barra corrente {self._now}."
            )
        self._signals.append(signal)
        self._queue.extend(self._risk.on_signal(signal))

    def _handle_order(self, order: OrderEvent) -> None:
        """ORDER -> ExecutionHandler (pendente fino alla barra successiva)."""
        if order.signal_timestamp != self._now:
            raise ValueError(
                f"Ordine {order.order_id} basato su un segnale stantio "
                f"({order.signal_timestamp}, barra corrente {self._now})."
            )
        self._orders.append(order)
        self._execution.submit_order(order)

    def _handle_fill(self, fill: FillEvent) -> None:
        """FILL -> RiskManager e Strategy (aggiornamento stato)."""
        if fill.timestamp != self._now:
            raise ValueError(
                f"Fill {fill.order_id} con timestamp {fill.timestamp} "
                f"diverso dalla barra corrente {self._now}."
            )
        self._fills.append(fill)
        self._risk.on_fill(fill)
        self._strategy.on_fill(fill)

    @property
    def _now(self) -> datetime:
        """Timestamp della barra in elaborazione."""
        if self._current_ts is None:
            raise RuntimeError("Nessuna barra in elaborazione.")
        return self._current_ts