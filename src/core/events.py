"""
src/core/events.py

Definizione degli eventi che circolano nella coda dell'engine event-driven.

Flusso canonico (nessun look-ahead bias):

    MarketEvent (barra t chiusa)
        -> Strategy      -> SignalEvent (timestamp = t)
        -> RiskManager   -> OrderEvent  (timestamp = t, signal_timestamp = t)
        -> Execution     -> FillEvent   (timestamp >= t+1, tipicamente open di t+1)

Tutti gli eventi sono immutabili (frozen) e validati in fase di costruzione,
cosi' un dato incoerente (prezzo negativo, fill nel passato, ecc.) fa fallire
subito l'esecuzione invece di corrompere silenziosamente il backtest.

Allineamento con config/settings.yaml:
    - OrderType            <-> execution.order_type (market | limit)
    - SignalDirection.SHORT <-> consentito solo se execution.allow_short = true
                               (il controllo e' responsabilita' del RiskManager)
    - FillEvent.timestamp  > signal_timestamp <-> backtest.fill_timing = next_open
    - strategy_id          <-> strategy.name
    - MarketEvent.timeframe <-> universe.timeframe
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import ClassVar, TypeAlias

__all__ = [
    "EventType",
    "SignalDirection",
    "OrderSide",
    "OrderType",
    "Event",
    "MarketEvent",
    "SignalEvent",
    "OrderEvent",
    "FillEvent",
    "AnyEvent",
]


# --------------------------------------------------------------------------- #
# Enumerazioni
# --------------------------------------------------------------------------- #
class EventType(StrEnum):
    """Tipo di evento, usato dall'engine per instradare i messaggi."""

    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"


class SignalDirection(StrEnum):
    """Intento direzionale espresso da una Strategy (non una quantita')."""

    LONG = "LONG"    # apri / mantieni posizione lunga
    SHORT = "SHORT"  # apri / mantieni posizione corta (se allow_short)
    EXIT = "EXIT"    # chiudi la posizione corrente sul simbolo


class OrderSide(StrEnum):
    """Lato dell'ordine."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    """Tipo di ordine. I valori coincidono con `execution.order_type` in YAML."""

    MARKET = "market"
    LIMIT = "limit"


# --------------------------------------------------------------------------- #
# Helper di validazione (funzioni di modulo: compatibili con slots=True)
# --------------------------------------------------------------------------- #
def _require_non_empty(name: str, value: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"'{name}' non puo' essere vuoto.")


def _require_positive(name: str, value: float) -> None:
    if not value > 0:  # intercetta anche NaN
        raise ValueError(f"'{name}' deve essere > 0 (ricevuto {value!r}).")


def _require_non_negative(name: str, value: float) -> None:
    if not value >= 0:  # intercetta anche NaN
        raise ValueError(f"'{name}' deve essere >= 0 (ricevuto {value!r}).")


# --------------------------------------------------------------------------- #
# Evento base
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True, kw_only=True)
class Event:
    """
    Classe base di tutti gli eventi.

    Attributes:
        timestamp: Istante a cui l'evento si riferisce. Per un MarketEvent e'
            il timestamp della barra; per gli altri e' il momento (simulato)
            in cui l'evento viene generato. Usare timestamp coerenti nel
            timezone (tutti naive oppure tutti tz-aware).
    """

    event_type: ClassVar[EventType]

    timestamp: datetime


# --------------------------------------------------------------------------- #
# MarketEvent
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True, kw_only=True)
class MarketEvent(Event):
    """
    Barra OHLCV completata per un simbolo.

    Viene emessa dal DataHandler solo quando la barra e' CHIUSA: la Strategy
    non vede mai dati parziali o futuri.

    Attributes:
        symbol: Ticker (es. "SPY").
        timeframe: Granularita' della barra (es. "1d"), da `universe.timeframe`.
        open: Prezzo di apertura.
        high: Massimo della barra.
        low: Minimo della barra.
        close: Prezzo di chiusura.
        volume: Volume scambiato (>= 0).
    """

    event_type: ClassVar[EventType] = EventType.MARKET

    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        _require_non_empty("symbol", self.symbol)
        _require_non_empty("timeframe", self.timeframe)
        for name in ("open", "high", "low", "close"):
            _require_positive(name, getattr(self, name))
        _require_non_negative("volume", self.volume)
        if self.high < max(self.open, self.close, self.low):
            raise ValueError(
                f"[{self.symbol} {self.timestamp}] high ({self.high}) inferiore "
                "a open/close/low."
            )
        if self.low > min(self.open, self.close, self.high):
            raise ValueError(
                f"[{self.symbol} {self.timestamp}] low ({self.low}) superiore "
                "a open/close/high."
            )


# --------------------------------------------------------------------------- #
# SignalEvent
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True, kw_only=True)
class SignalEvent(Event):
    """
    Segnale generato da una Strategy alla chiusura della barra `timestamp`.

    Esprime SOLO l'intento direzionale: niente quantita', niente stop, niente
    sizing. Capitale e rischio sono competenza esclusiva del RiskManager.

    Attributes:
        symbol: Ticker a cui si riferisce il segnale.
        direction: LONG, SHORT oppure EXIT.
        strategy_id: Identificativo della strategia (es. "mean_reversion",
            coincide con `strategy.name`).
        strength: Convinzione del segnale in [0, 1] (default 1.0).
        reference_price: Close della barra che ha generato il segnale. E'
            puramente informativo: NON e' il prezzo di esecuzione.
        metadata: Diagnostica numerica libera (es. {"zscore": -2.3}).
    """

    event_type: ClassVar[EventType] = EventType.SIGNAL

    symbol: str
    direction: SignalDirection
    strategy_id: str
    strength: float = 1.0
    reference_price: float | None = None
    metadata: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty("symbol", self.symbol)
        _require_non_empty("strategy_id", self.strategy_id)
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"'strength' deve essere in [0, 1] (ricevuto {self.strength!r}).")
        if self.reference_price is not None:
            _require_positive("reference_price", self.reference_price)


# --------------------------------------------------------------------------- #
# OrderEvent
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True, kw_only=True)
class OrderEvent(Event):
    """
    Ordine prodotto dal RiskManager a partire da un SignalEvent.

    `timestamp` e' il momento di creazione (barra t); `signal_timestamp` e'
    la barra del segnale originario. L'esecuzione puo' avvenire solo DOPO
    `signal_timestamp` (vedi FillEvent).

    Attributes:
        symbol: Ticker.
        side: BUY o SELL.
        quantity: Quantita' (> 0). L'arrotondamento a numero intero in base a
            `execution.allow_fractional_shares` e' fatto dal RiskManager.
        signal_timestamp: Timestamp del segnale che ha originato l'ordine.
        strategy_id: Strategia di provenienza (tracciabilita' P&L).
        order_type: MARKET o LIMIT (da `execution.order_type`).
        limit_price: Obbligatorio se order_type == LIMIT, vietato altrimenti.
        stop_price: Livello di stop calcolato dal RiskManager
            (es. entry - `atr_stop_multiple` x ATR); informativo per l'uscita.
        order_id: Identificativo univoco (UUID4 esadecimale).
    """

    event_type: ClassVar[EventType] = EventType.ORDER

    symbol: str
    side: OrderSide
    quantity: float
    signal_timestamp: datetime
    strategy_id: str
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    order_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        _require_non_empty("symbol", self.symbol)
        _require_non_empty("strategy_id", self.strategy_id)
        _require_positive("quantity", self.quantity)
        if self.timestamp < self.signal_timestamp:
            raise ValueError("L'ordine non puo' precedere il segnale che lo ha generato.")
        if self.order_type is OrderType.LIMIT:
            if self.limit_price is None:
                raise ValueError("Un ordine LIMIT richiede 'limit_price'.")
            _require_positive("limit_price", self.limit_price)
        elif self.limit_price is not None:
            raise ValueError("'limit_price' ammesso solo per ordini LIMIT.")
        if self.stop_price is not None:
            _require_positive("stop_price", self.stop_price)


# --------------------------------------------------------------------------- #
# FillEvent
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True, kw_only=True)
class FillEvent(Event):
    """
    Esecuzione (anche parziale) di un ordine, con costi gia' inclusi.

    Garanzia anti look-ahead: `timestamp` deve essere STRETTAMENTE successivo a
    `signal_timestamp` (segnale a t -> fill a t+1 o dopo).

    Garanzia sui costi: lo slippage e' sempre sfavorevole (BUY: fill_price >=
    reference_price; SELL: fill_price <= reference_price) e la commissione e'
    sempre registrata esplicitamente.

    Attributes:
        order_id: Ordine di origine.
        symbol: Ticker.
        side: BUY o SELL.
        quantity: Quantita' effettivamente eseguita (> 0).
        reference_price: Prezzo di mercato "pulito" (es. open di t+1) PRIMA dello
            slippage.
        fill_price: Prezzo di esecuzione DOPO lo slippage
            (`execution.slippage`).
        commission: Commissione totale in valuta base (`execution.commission`).
        signal_timestamp: Timestamp del segnale originario.
        strategy_id: Strategia di provenienza.
    """

    event_type: ClassVar[EventType] = EventType.FILL

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    reference_price: float
    fill_price: float
    commission: float
    signal_timestamp: datetime
    strategy_id: str

    def __post_init__(self) -> None:
        _require_non_empty("order_id", self.order_id)
        _require_non_empty("symbol", self.symbol)
        _require_non_empty("strategy_id", self.strategy_id)
        _require_positive("quantity", self.quantity)
        _require_positive("reference_price", self.reference_price)
        _require_positive("fill_price", self.fill_price)
        _require_non_negative("commission", self.commission)
        if self.timestamp <= self.signal_timestamp:
            raise ValueError(
                "Look-ahead bias: il fill deve avvenire strettamente dopo il "
                f"segnale (signal={self.signal_timestamp}, fill={self.timestamp})."
            )
        adverse = (
            self.fill_price >= self.reference_price
            if self.side is OrderSide.BUY
            else self.fill_price <= self.reference_price
        )
        if not adverse:
            raise ValueError("Lo slippage deve essere sfavorevole all'esecuzione.")

    # ---- Metriche derivate (sempre in valuta base) ------------------------ #
    @property
    def gross_value(self) -> float:
        """Controvalore lordo dell'esecuzione: quantity x fill_price."""
        return self.quantity * self.fill_price

    @property
    def slippage_cost(self) -> float:
        """Costo di slippage in valuta base (sempre >= 0)."""
        return self.quantity * abs(self.fill_price - self.reference_price)

    @property
    def slippage_bps(self) -> float:
        """Slippage realizzato in basis points rispetto al prezzo di riferimento."""
        return abs(self.fill_price - self.reference_price) / self.reference_price * 10_000

    @property
    def total_cost(self) -> float:
        """Costi di transazione totali: commissione + slippage."""
        return self.commission + self.slippage_cost

    @property
    def signed_quantity(self) -> float:
        """Variazione di posizione: +quantity per BUY, -quantity per SELL."""
        return self.quantity if self.side is OrderSide.BUY else -self.quantity

    @property
    def cash_flow(self) -> float:
        """
        Variazione di cassa dovuta al fill, commissioni incluse.

        BUY  -> -(gross_value + commission)
        SELL -> +(gross_value - commission)
        """
        if self.side is OrderSide.BUY:
            return -(self.gross_value + self.commission)
        return self.gross_value - self.commission


# --------------------------------------------------------------------------- #
# Type alias per la coda eventi dell'engine
# --------------------------------------------------------------------------- #
AnyEvent: TypeAlias = MarketEvent | SignalEvent | OrderEvent | FillEvent