"""
src/risk/risk_manager.py

`RiskManager`: unico proprietario di capitale, posizioni e rischio.

Soddisfa `RiskManagerProtocol` di src/core/engine.py:
    on_market_event, on_fill, on_signal, snapshot, liquidation_orders,
    property kill_switch_triggered.

Responsabilita':
    1. Stato di portafoglio: liquidita', posizioni, prezzo medio di carico,
       PnL realizzato (netto commissioni) e non realizzato.
    2. Da SignalEvent a OrderEvent con position sizing dinamico (ATR, frazione
       fissa o azioni fisse) entro i limiti `max_position_pct`,
       `max_gross_exposure`, `max_open_positions` e liquidita' disponibile,
       tenendo conto degli ordini pendenti (emessi ma non ancora eseguiti).
    3. Kill-switch: `snapshot()` calcola equity e drawdown dal massimo storico;
       al raggiungimento di `max_drawdown_pct` il kill-switch scatta (una sola
       volta, in modo irreversibile) e blocca ogni nuovo ordine.

Contratti verso l'ExecutionHandler (da rispettare in `simulated.py`):
    - al massimo UN ordine pendente per simbolo;
    - un ordine e' valido solo per la barra successiva a quella del segnale: se
      non e' eseguito integralmente li', va scartato. Il RiskManager lo toglie
      dai pendenti quando arriva la barra successiva del simbolo;
    - le uscite e le liquidazioni sono SEMPRE ordini a mercato, cosi' la
      riduzione del rischio non puo' restare inevasa.

Contabilita' (identita' verificabile):
    equity = cash + somma(quantita x ultimo prezzo)
    equity - initial_capital = PnL realizzato (netto commissioni) + PnL non realizzato
    Lo slippage e' gia' incluso nel `fill_price`; le commissioni sono in `cash_flow`.

Sizing e liquidita':
    Il capitale e' impegnato al prezzo di riferimento maggiorato di
    `price_buffer_pct` (gap tra close del segnale e prezzo di esecuzione,
    slippage e commissioni): la liquidita' non diventa negativa. Non e' prevista
    leva: gli acquisti sono finanziati dalla cassa disponibile.

Allineamento con config/settings.yaml:
    general.initial_capital
    risk.max_risk_per_trade, max_position_pct, max_gross_exposure,
         max_open_positions, max_drawdown_pct, kill_switch_action
         (opzionale: risk.price_buffer_pct, default 0.02)
    risk.position_sizing.method, atr_window, atr_stop_multiple,
         fixed_fraction, fixed_shares
    execution.order_type, allow_short, allow_fractional_shares
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from typing import Any, Final

import pandas as pd

from src.core.engine import PortfolioSnapshot
from src.core.events import (
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderSide,
    OrderType,
    SignalDirection,
    SignalEvent,
)
from src.data_handler.base import BaseDataHandler
from src.strategy.base import LookAheadError

__all__ = [
    "KillSwitchAction",
    "SizingMethod",
    "RiskParams",
    "Position",
    "RiskManager",
]

logger = logging.getLogger(__name__)

#: Soglia sotto la quale una quantita' e' considerata zero.
_EPS: Final[float] = 1e-9

#: Granularita' delle quantita' frazionarie (4 decimali) se `allow_fractional_shares`.
_FRACTIONAL_STEP: Final[float] = 10_000.0


# --------------------------------------------------------------------------- #
# Enumerazioni e parametri
# --------------------------------------------------------------------------- #
class KillSwitchAction(StrEnum):
    """Azione del kill-switch. Coincide con `risk.kill_switch_action`."""

    LIQUIDATE_AND_HALT = "liquidate_and_halt"
    HALT_ONLY = "halt_only"


class SizingMethod(StrEnum):
    """Metodo di position sizing. Coincide con `risk.position_sizing.method`."""

    ATR = "atr"
    FIXED_FRACTION = "fixed_fraction"
    FIXED_SHARES = "fixed_shares"


@dataclass(frozen=True, slots=True)
class RiskParams:
    """
    Parametri di rischio, validati in fase di costruzione.

    Attributes:
        initial_capital: Capitale iniziale (`general.initial_capital`).
        max_risk_per_trade: Frazione di equity a rischio per trade (0.01 = 1%).
        max_position_pct: Valore massimo di una posizione / equity.
        max_gross_exposure: Valore assoluto totale delle posizioni / equity.
        max_open_positions: Numero massimo di simboli con posizione (o ingresso pendente).
        max_drawdown_pct: Drawdown dal massimo storico che attiva il kill-switch.
        kill_switch_action: Liquidare e fermarsi, oppure solo fermarsi.
        sizing_method: Metodo di sizing.
        atr_window: Finestra dell'ATR.
        atr_stop_multiple: Distanza dello stop in multipli di ATR.
        fixed_fraction: Frazione di equity per `fixed_fraction`.
        fixed_shares: Quantita' per `fixed_shares`.
        allow_short: Se False i segnali SHORT sono rifiutati.
        allow_fractional_shares: Se False le quantita' sono arrotondate per difetto.
        order_type: Tipo di ordine per gli INGRESSI (le uscite sono sempre a mercato).
        price_buffer_pct: Maggiorazione del prezzo nel calcolo di cassa ed
            esposizione impegnate (gap, slippage, commissioni).
    """

    initial_capital: float
    max_risk_per_trade: float
    max_position_pct: float
    max_gross_exposure: float
    max_open_positions: int
    max_drawdown_pct: float
    kill_switch_action: KillSwitchAction
    sizing_method: SizingMethod
    atr_window: int
    atr_stop_multiple: float
    fixed_fraction: float
    fixed_shares: int
    allow_short: bool
    allow_fractional_shares: bool
    order_type: OrderType
    price_buffer_pct: float = 0.02

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("'initial_capital' deve essere > 0.")
        if not 0 < self.max_risk_per_trade <= 1:
            raise ValueError("'max_risk_per_trade' deve essere in (0, 1].")
        if not 0 < self.max_position_pct <= 1:
            raise ValueError("'max_position_pct' deve essere in (0, 1].")
        if self.max_gross_exposure <= 0:
            raise ValueError("'max_gross_exposure' deve essere > 0.")
        if self.max_open_positions < 1:
            raise ValueError("'max_open_positions' deve essere >= 1.")
        if not 0 < self.max_drawdown_pct < 1:
            raise ValueError("'max_drawdown_pct' deve essere in (0, 1).")
        if self.atr_window < 1:
            raise ValueError("'atr_window' deve essere >= 1.")
        if self.atr_stop_multiple <= 0:
            raise ValueError("'atr_stop_multiple' deve essere > 0.")
        if not 0 < self.fixed_fraction <= 1:
            raise ValueError("'fixed_fraction' deve essere in (0, 1].")
        if self.fixed_shares < 1:
            raise ValueError("'fixed_shares' deve essere >= 1.")
        if self.price_buffer_pct < 0:
            raise ValueError("'price_buffer_pct' deve essere >= 0.")

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> RiskParams:
        """
        Legge i parametri dal dizionario completo di settings.yaml.

        Args:
            settings: Contenuto di settings.yaml gia' caricato (`yaml.safe_load`).
        """
        risk = settings["risk"]
        sizing = risk["position_sizing"]
        execution = settings["execution"]
        return cls(
            initial_capital=float(settings["general"]["initial_capital"]),
            max_risk_per_trade=float(risk["max_risk_per_trade"]),
            max_position_pct=float(risk["max_position_pct"]),
            max_gross_exposure=float(risk["max_gross_exposure"]),
            max_open_positions=int(risk["max_open_positions"]),
            max_drawdown_pct=float(risk["max_drawdown_pct"]),
            kill_switch_action=KillSwitchAction(risk["kill_switch_action"]),
            sizing_method=SizingMethod(sizing["method"]),
            atr_window=int(sizing["atr_window"]),
            atr_stop_multiple=float(sizing["atr_stop_multiple"]),
            fixed_fraction=float(sizing["fixed_fraction"]),
            fixed_shares=int(sizing["fixed_shares"]),
            allow_short=bool(execution["allow_short"]),
            allow_fractional_shares=bool(execution["allow_fractional_shares"]),
            order_type=OrderType(execution["order_type"]),
            price_buffer_pct=float(risk.get("price_buffer_pct", 0.02)),
        )


# --------------------------------------------------------------------------- #
# Strutture di stato
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Position:
    """
    Posizione aperta su un simbolo.

    Attributes:
        symbol: Ticker.
        strategy_id: Strategia che ha aperto la posizione.
        quantity: Quantita' netta con segno (> 0 long, < 0 short).
        avg_price: Prezzo medio di carico (include lo slippage, esclude le commissioni).
        last_price: Ultimo prezzo di mercato noto (mark-to-market).
        opened_at: Timestamp del fill di apertura.
    """

    symbol: str
    strategy_id: str
    quantity: float = 0.0
    avg_price: float = 0.0
    last_price: float = 0.0
    opened_at: datetime | None = None

    @property
    def market_value(self) -> float:
        """Valore di mercato con segno: quantity x last_price."""
        return self.quantity * self.last_price

    @property
    def unrealized_pnl(self) -> float:
        """PnL non realizzato: quantity x (last_price - avg_price)."""
        return self.quantity * (self.last_price - self.avg_price)

    def apply_fill(
        self, signed_quantity: float, price: float, commission: float, timestamp: datetime
    ) -> float:
        """
        Applica un'esecuzione e restituisce il PnL realizzato da quel fill,
        al netto della commissione.

        Gestisce aumento, riduzione, chiusura e inversione della posizione.
        """
        realized = -commission
        flat = abs(self.quantity) < _EPS

        if flat or self.quantity * signed_quantity > 0:  # apre o aumenta
            new_quantity = self.quantity + signed_quantity
            self.avg_price = (
                abs(self.quantity) * self.avg_price + abs(signed_quantity) * price
            ) / abs(new_quantity)
            if flat:
                self.opened_at = timestamp
        else:  # riduce, chiude o inverte
            sign = 1.0 if self.quantity > 0 else -1.0
            closed = min(abs(signed_quantity), abs(self.quantity))
            realized += closed * (price - self.avg_price) * sign
            new_quantity = self.quantity + signed_quantity
            if abs(new_quantity) < _EPS:
                new_quantity = 0.0
                self.avg_price = 0.0
                self.opened_at = None
            elif new_quantity * sign < 0:  # inversione: il residuo parte da `price`
                self.avg_price = price
                self.opened_at = timestamp

        self.quantity = new_quantity
        self.last_price = price
        return realized


@dataclass(slots=True)
class _PendingOrder:
    """Ordine emesso e non ancora (del tutto) eseguito."""

    order: OrderEvent
    remaining: float          # quantita' ancora da eseguire
    reserved_value: float     # valore impegnato (solo ingressi; 0 per le uscite)
    is_entry: bool


# --------------------------------------------------------------------------- #
# RiskManager
# --------------------------------------------------------------------------- #
class RiskManager:
    """
    Gestore di capitale e rischio: portafoglio, sizing, limiti e kill-switch.

    Args:
        params: Parametri di rischio validati.
        data_handler: DataHandler da cui leggere lo storico per l'ATR (sola lettura).
    """

    def __init__(self, params: RiskParams, data_handler: BaseDataHandler) -> None:
        self._params = params
        self._data_handler = data_handler

        self._cash: float = params.initial_capital
        self._positions: dict[str, Position] = {}
        self._last_prices: dict[str, float] = {}
        self._pending: dict[str, _PendingOrder] = {}  # order_id -> ordine pendente

        self._realized_by_symbol: dict[str, float] = {}
        self._total_commissions: float = 0.0

        self._peak_equity: float = params.initial_capital
        self._drawdown: float = 0.0
        self._kill_switch_triggered: bool = False
        self._last_snapshot: PortfolioSnapshot | None = None
        self._rejections: Counter[str] = Counter()

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any], data_handler: BaseDataHandler) -> RiskManager:
        """Costruisce il RiskManager dal dizionario completo di settings.yaml."""
        return cls(RiskParams.from_settings(settings), data_handler)

    # ------------------------------------------------------------------ #
    # Proprieta' di sola lettura (analytics, dashboard, log)
    # ------------------------------------------------------------------ #
    @property
    def params(self) -> RiskParams:
        """Parametri di rischio correnti (immutabili)."""
        return self._params

    @property
    def kill_switch_triggered(self) -> bool:
        """True se il drawdown ha raggiunto `max_drawdown_pct` (irreversibile)."""
        return self._kill_switch_triggered

    @property
    def cash(self) -> float:
        """Liquidita' corrente in valuta base."""
        return self._cash

    @property
    def equity(self) -> float:
        """Equity corrente: cash + valore di mercato delle posizioni."""
        return self._cash + sum(p.market_value for p in self._positions.values())

    @property
    def peak_equity(self) -> float:
        """Massimo storico dell'equity osservato negli snapshot."""
        return self._peak_equity

    @property
    def drawdown(self) -> float:
        """Drawdown dall'ultimo snapshot, come frazione >= 0."""
        return self._drawdown

    @property
    def realized_pnl(self) -> float:
        """PnL realizzato totale, al netto delle commissioni."""
        return sum(self._realized_by_symbol.values())

    @property
    def unrealized_pnl(self) -> float:
        """PnL non realizzato totale ai prezzi correnti."""
        return sum(p.unrealized_pnl for p in self._positions.values())

    @property
    def total_commissions(self) -> float:
        """Commissioni totali pagate."""
        return self._total_commissions

    @property
    def positions(self) -> dict[str, Position]:
        """Copia delle posizioni aperte per simbolo."""
        return {symbol: replace(position) for symbol, position in self._positions.items()}

    @property
    def pending_orders(self) -> tuple[OrderEvent, ...]:
        """Ordini emessi e non ancora eseguiti."""
        return tuple(p.order for p in self._pending.values())

    @property
    def rejections(self) -> dict[str, int]:
        """Conteggio dei segnali rifiutati per motivo (diagnostica)."""
        return dict(self._rejections)

    # ------------------------------------------------------------------ #
    # Protocollo dell'engine
    # ------------------------------------------------------------------ #
    def on_market_event(self, event: MarketEvent) -> None:
        """
        Mark-to-market del simbolo ai prezzi di chiusura della barra e scadenza
        degli ordini pendenti creati prima di questa barra (valgono una sola barra).
        """
        symbol = event.symbol
        self._last_prices[symbol] = event.close
        position = self._positions.get(symbol)
        if position is not None:
            position.last_price = event.close

        expired = [
            order_id
            for order_id, pending in self._pending.items()
            if pending.order.symbol == symbol and pending.order.timestamp < event.timestamp
        ]
        for order_id in expired:
            logger.warning(
                "[%s] ordine %s scaduto senza esecuzione completa.", symbol, order_id[:8]
            )
            del self._pending[order_id]
            self._rejections["order_expired"] += 1

    def on_fill(self, fill: FillEvent) -> None:
        """Aggiorna cassa, posizione, PnL realizzato e ordini pendenti."""
        self._cash += fill.cash_flow
        self._total_commissions += fill.commission

        position = self._positions.get(fill.symbol)
        if position is None:
            position = Position(symbol=fill.symbol, strategy_id=fill.strategy_id)

        realized = position.apply_fill(
            fill.signed_quantity, fill.fill_price, fill.commission, fill.timestamp
        )
        self._realized_by_symbol[fill.symbol] = (
            self._realized_by_symbol.get(fill.symbol, 0.0) + realized
        )

        if abs(position.quantity) < _EPS:
            self._positions.pop(fill.symbol, None)
        else:
            self._positions[fill.symbol] = position
        self._last_prices[fill.symbol] = fill.fill_price

        pending = self._pending.get(fill.order_id)
        if pending is not None:
            previous = pending.remaining
            pending.remaining -= fill.quantity
            if pending.remaining < _EPS:
                del self._pending[fill.order_id]
            else:
                pending.reserved_value *= pending.remaining / previous

    def on_signal(self, signal: SignalEvent) -> list[OrderEvent]:
        """
        Trasforma un segnale in zero o un ordine, applicando limiti e sizing.

        I segnali rifiutati vengono contati in `rejections` con il motivo.
        """
        if self._kill_switch_triggered:
            return self._reject("kill_switch_active", signal)
        if signal.symbol in {p.order.symbol for p in self._pending.values()}:
            return self._reject("pending_order_exists", signal)

        if signal.direction is SignalDirection.EXIT:
            return self._exit_orders(signal)
        return self._entry_orders(signal)

    def snapshot(self, timestamp: datetime) -> PortfolioSnapshot:
        """
        Fotografa il portafoglio, aggiorna massimo storico e drawdown e valuta il
        kill-switch.

        Returns:
            PortfolioSnapshot per l'equity curve registrata dall'engine.
        """
        equity = self.equity
        self._peak_equity = max(self._peak_equity, equity)
        self._drawdown = (
            max(0.0, (self._peak_equity - equity) / self._peak_equity)
            if self._peak_equity > 0
            else 1.0
        )

        if not self._kill_switch_triggered and self._drawdown >= self._params.max_drawdown_pct:
            self._kill_switch_triggered = True
            logger.warning(
                "Kill-switch: drawdown %.2f%% >= soglia %.2f%% (equity %.2f, picco %.2f).",
                self._drawdown * 100,
                self._params.max_drawdown_pct * 100,
                equity,
                self._peak_equity,
            )

        gross_value = sum(abs(p.market_value) for p in self._positions.values())
        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            cash=self._cash,
            equity=equity,
            gross_exposure=gross_value / equity if equity > 0 else 0.0,
            drawdown=self._drawdown,
            positions={s: p.quantity for s, p in self._positions.items()},
        )
        self._last_snapshot = snapshot
        return snapshot

    def liquidation_orders(self, timestamp: datetime) -> list[OrderEvent]:
        """
        Ordini a mercato per chiudere tutte le posizioni (fill dalla barra successiva).

        Lista vuota se `kill_switch_action = halt_only`.
        """
        if self._params.kill_switch_action is KillSwitchAction.HALT_ONLY:
            return []

        orders: list[OrderEvent] = []
        for symbol, position in self._positions.items():
            order = OrderEvent(
                timestamp=timestamp,
                symbol=symbol,
                side=OrderSide.SELL if position.quantity > 0 else OrderSide.BUY,
                quantity=abs(position.quantity),
                signal_timestamp=timestamp,
                strategy_id=position.strategy_id,
                order_type=OrderType.MARKET,
            )
            self._register_pending(order, reserved_value=0.0, is_entry=False)
            orders.append(order)
        if orders:
            logger.warning("Kill-switch: liquidazione di %d posizioni.", len(orders))
        return orders

    # ------------------------------------------------------------------ #
    # Uscite
    # ------------------------------------------------------------------ #
    def _exit_orders(self, signal: SignalEvent) -> list[OrderEvent]:
        """Ordine a mercato che chiude l'intera posizione sul simbolo."""
        position = self._positions.get(signal.symbol)
        if position is None:
            return self._reject("exit_without_position", signal)

        order = OrderEvent(
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            side=OrderSide.SELL if position.quantity > 0 else OrderSide.BUY,
            quantity=abs(position.quantity),
            signal_timestamp=signal.timestamp,
            strategy_id=signal.strategy_id,
            order_type=OrderType.MARKET,
        )
        self._register_pending(order, reserved_value=0.0, is_entry=False)
        return [order]

    # ------------------------------------------------------------------ #
    # Ingressi
    # ------------------------------------------------------------------ #
    def _entry_orders(self, signal: SignalEvent) -> list[OrderEvent]:
        """Valida i limiti, calcola la quantita' e costruisce l'ordine di ingresso."""
        p = self._params
        symbol = signal.symbol
        is_long = signal.direction is SignalDirection.LONG

        if not is_long and not p.allow_short:
            return self._reject("short_not_allowed", signal)
        if symbol in self._positions:
            return self._reject("already_in_position", signal)

        entry_symbols = {
            pending.order.symbol for pending in self._pending.values() if pending.is_entry
        }
        if len(self._positions) + len(entry_symbols) >= p.max_open_positions:
            return self._reject("max_open_positions", signal)

        price = signal.reference_price or self._last_prices.get(symbol)
        if price is None or price <= 0:
            return self._reject("no_reference_price", signal)

        equity = self.equity
        if equity <= 0:
            return self._reject("no_equity", signal)

        atr = self._atr(symbol, signal.timestamp) if p.sizing_method is SizingMethod.ATR else None
        if p.sizing_method is SizingMethod.ATR and (atr is None or atr <= 0):
            return self._reject("atr_unavailable", signal)

        quantity, binding = self._size(equity, price, atr, is_long)
        if quantity <= 0:
            return self._reject(f"size_below_minimum:{binding}", signal)

        stop_price: float | None = None
        if atr is not None:
            distance = p.atr_stop_multiple * atr
            raw_stop = price - distance if is_long else price + distance
            stop_price = raw_stop if raw_stop > 0 else None

        limit_price = round(price, 4) if p.order_type is OrderType.LIMIT else None
        order = OrderEvent(
            timestamp=signal.timestamp,
            symbol=symbol,
            side=OrderSide.BUY if is_long else OrderSide.SELL,
            quantity=quantity,
            signal_timestamp=signal.timestamp,
            strategy_id=signal.strategy_id,
            order_type=p.order_type,
            limit_price=limit_price,
            stop_price=stop_price,
        )
        reserved = quantity * price * (1.0 + p.price_buffer_pct)
        self._register_pending(order, reserved_value=reserved, is_entry=True)
        logger.debug(
            "[%s %s] ordine %s qty=%s (limite: %s)", symbol, signal.timestamp, order.side, quantity, binding
        )
        return [order]

    def _size(
        self, equity: float, price: float, atr: float | None, is_long: bool
    ) -> tuple[float, str]:
        """
        Quantita' da negoziare: minimo tra target del metodo di sizing e i limiti.

        Returns:
            (quantita' arrotondata, nome del vincolo piu' stringente).
        """
        p = self._params
        price_eff = price * (1.0 + p.price_buffer_pct)

        if p.sizing_method is SizingMethod.ATR:
            if atr is None:  # garantito dal chiamante; guardia per il type checker
                return 0.0, "atr_unavailable"
            target = equity * p.max_risk_per_trade / (p.atr_stop_multiple * atr)
        elif p.sizing_method is SizingMethod.FIXED_FRACTION:
            target = equity * p.fixed_fraction / price_eff
        else:
            target = float(p.fixed_shares)

        reserved_gross = sum(o.reserved_value for o in self._pending.values() if o.is_entry)
        gross_value = sum(abs(pos.market_value) for pos in self._positions.values())
        limits: dict[str, float] = {
            "target": target,
            "max_position_pct": equity * p.max_position_pct / price_eff,
            "max_gross_exposure": max(
                0.0, equity * p.max_gross_exposure - gross_value - reserved_gross
            )
            / price_eff,
        }
        if is_long:
            reserved_cash = sum(
                o.reserved_value
                for o in self._pending.values()
                if o.is_entry and o.order.side is OrderSide.BUY
            )
            limits["cash"] = max(0.0, self._cash - reserved_cash) / price_eff

        binding = min(limits, key=limits.__getitem__)
        quantity = limits[binding]

        if p.allow_fractional_shares:
            quantity = math.floor(quantity * _FRACTIONAL_STEP) / _FRACTIONAL_STEP
        else:
            quantity = float(math.floor(quantity))
        return quantity, binding

    # ------------------------------------------------------------------ #
    # ATR (stessa definizione di MeanReversionStrategy)
    # ------------------------------------------------------------------ #
    def _atr(self, symbol: str, as_of: datetime) -> float | None:
        """
        ATR = media semplice del True Range sulle ultime `atr_window` barre, con
        TR = max(high - low, |high - close_prec|, |low - close_prec|).

        Raises:
            LookAheadError: se lo storico contiene barre successive ad `as_of`.
        """
        window = self._params.atr_window
        bars = self._data_handler.get_latest_bars(symbol, window + 1)
        if len(bars) < window + 1:
            return None
        if bars.index[-1] > as_of:
            raise LookAheadError(f"[{symbol}] barra del {bars.index[-1]} successiva a {as_of}.")

        prev_close = bars["close"].shift(1)
        true_range = pd.concat(
            [
                bars["high"] - bars["low"],
                (bars["high"] - prev_close).abs(),
                (bars["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = float(true_range.iloc[1:].mean())
        return atr if math.isfinite(atr) else None

    # ------------------------------------------------------------------ #
    # Utilita' interne
    # ------------------------------------------------------------------ #
    def _register_pending(self, order: OrderEvent, reserved_value: float, is_entry: bool) -> None:
        """Registra un ordine emesso tra i pendenti."""
        self._pending[order.order_id] = _PendingOrder(
            order=order,
            remaining=order.quantity,
            reserved_value=reserved_value,
            is_entry=is_entry,
        )

    def _reject(self, reason: str, signal: SignalEvent) -> list[OrderEvent]:
        """Conta e registra un segnale rifiutato; restituisce sempre lista vuota."""
        self._rejections[reason.split(":")[0]] += 1
        logger.debug(
            "[%s %s] segnale %s rifiutato: %s", signal.symbol, signal.timestamp, signal.direction, reason
        )
        return []