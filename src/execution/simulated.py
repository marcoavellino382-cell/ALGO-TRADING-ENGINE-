"""
src/execution/simulated.py

`SimulatedExecutionHandler`: esecuzione simulata per il backtest.

Soddisfa `ExecutionHandlerProtocol` di src/core/engine.py:
    submit_order, on_market_event, has_pending_orders.

Regole di esecuzione (rispettano i contratti del RiskManager):
    - Un ordine creato alla barra t (signal_timestamp = t) viene eseguito SOLO
      sulla barra successiva: `on_market_event` ignora gli ordini con
      `signal_timestamp >= event.timestamp`. Look-ahead impossibile.
    - Prezzo di riferimento della barra t+1, da `backtest.fill_timing`:
        next_open  -> open di t+1
        next_close -> close di t+1
    - Lo slippage e' sempre sfavorevole: BUY a prezzo piu' alto, SELL a prezzo
      piu' basso del riferimento.
    - Un ordine vale una sola barra: se non e' eseguito (limite non raggiunto,
      nessuna liquidita'), viene scartato; un'esecuzione parziale scarta il resto.
    - Un solo ordine pendente per simbolo (contratto del RiskManager, verificato).

Ordini LIMIT (solo ingressi; le uscite sono sempre a mercato):
    BUY  eseguito se riferimento <= limite, prezzo = min(riferimento + slippage, limite)
    SELL eseguito se riferimento >= limite, prezzo = max(riferimento - slippage, limite)
    Non si assume mai che il prezzo "tocchi" il limite durante la barra.

Modelli di slippage (`execution.slippage.model`):
    fixed_bps      slippage = fixed_bps
    volume_impact  slippage = fixed_bps + 10.000 x volume_impact_coeff x partecipazione
                   dove partecipazione = quantita' eseguita / volume della barra;
                   la quantita' eseguibile e' limitata a `max_participation_rate`
                   x volume (esecuzione parziale). Con volume nullo non si esegue.
                   `fixed_bps` rappresenta il costo base (mezzo spread) e resta
                   sempre applicato.

Modelli di commissione (`execution.commission.model`):
    per_share   per_share x quantita'
    percentage  percentage_bps sul controvalore
    fixed       fixed_per_order
    Per per_share e percentage: commissione = max(importo, min_per_order), poi
    limitata a max_pct_of_trade_value x controvalore (come i broker retail: il
    tetto prevale sul minimo per ordini molto piccoli).

Allineamento con config/settings.yaml:
    backtest.fill_timing
    execution.allow_fractional_shares
    execution.commission.* e execution.slippage.*
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from src.core.engine import FillTiming
from src.core.events import FillEvent, MarketEvent, OrderEvent, OrderSide, OrderType

__all__ = [
    "CommissionModel",
    "SlippageModel",
    "CommissionParams",
    "SlippageParams",
    "SimulatedExecutionParams",
    "SimulatedExecutionHandler",
]

logger = logging.getLogger(__name__)

#: Tetto di sicurezza allo slippage (50%): evita prezzi di vendita non positivi.
_MAX_SLIPPAGE_BPS: Final[float] = 5_000.0

#: Granularita' delle quantita' frazionarie (4 decimali).
_FRACTIONAL_STEP: Final[float] = 10_000.0


# --------------------------------------------------------------------------- #
# Enumerazioni e parametri
# --------------------------------------------------------------------------- #
class CommissionModel(StrEnum):
    """Modello di commissione. Coincide con `execution.commission.model`."""

    PER_SHARE = "per_share"
    PERCENTAGE = "percentage"
    FIXED = "fixed"


class SlippageModel(StrEnum):
    """Modello di slippage. Coincide con `execution.slippage.model`."""

    FIXED_BPS = "fixed_bps"
    VOLUME_IMPACT = "volume_impact"


@dataclass(frozen=True, slots=True)
class CommissionParams:
    """
    Parametri di commissione (valuta base).

    Attributes:
        model: Modello attivo.
        per_share: Importo per azione (model = per_share).
        min_per_order: Commissione minima per ordine (per_share e percentage).
        max_pct_of_trade_value: Tetto in frazione del controvalore (0.01 = 1%).
        percentage_bps: Commissione in bps sul controvalore (model = percentage).
        fixed_per_order: Importo fisso per ordine (model = fixed).
    """

    model: CommissionModel
    per_share: float
    min_per_order: float
    max_pct_of_trade_value: float
    percentage_bps: float
    fixed_per_order: float

    def __post_init__(self) -> None:
        for name in ("per_share", "min_per_order", "percentage_bps", "fixed_per_order"):
            if getattr(self, name) < 0:
                raise ValueError(f"'{name}' deve essere >= 0.")
        if not 0 < self.max_pct_of_trade_value <= 1:
            raise ValueError("'max_pct_of_trade_value' deve essere in (0, 1].")


@dataclass(frozen=True, slots=True)
class SlippageParams:
    """
    Parametri di slippage.

    Attributes:
        model: Modello attivo.
        fixed_bps: Slippage fisso in bps (costo base anche per volume_impact).
        volume_impact_coeff: Coefficiente di impatto sulla partecipazione (>= 0).
        max_participation_rate: Quota massima del volume della barra eseguibile (0, 1].
    """

    model: SlippageModel
    fixed_bps: float
    volume_impact_coeff: float
    max_participation_rate: float

    def __post_init__(self) -> None:
        if self.fixed_bps < 0:
            raise ValueError("'fixed_bps' deve essere >= 0.")
        if self.volume_impact_coeff < 0:
            raise ValueError("'volume_impact_coeff' deve essere >= 0.")
        if not 0 < self.max_participation_rate <= 1:
            raise ValueError("'max_participation_rate' deve essere in (0, 1].")


@dataclass(frozen=True, slots=True)
class SimulatedExecutionParams:
    """
    Parametri completi dell'esecuzione simulata.

    Attributes:
        commission: Parametri di commissione.
        slippage: Parametri di slippage.
        fill_timing: Prezzo di esecuzione sulla barra t+1 (open o close).
        allow_fractional_shares: Se False le quantita' parziali sono arrotondate per difetto.
    """

    commission: CommissionParams
    slippage: SlippageParams
    fill_timing: FillTiming
    allow_fractional_shares: bool

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> SimulatedExecutionParams:
        """
        Legge i parametri dal dizionario completo di settings.yaml.

        Args:
            settings: Contenuto di settings.yaml gia' caricato (`yaml.safe_load`).
        """
        execution = settings["execution"]
        commission = execution["commission"]
        slippage = execution["slippage"]
        return cls(
            commission=CommissionParams(
                model=CommissionModel(commission["model"]),
                per_share=float(commission["per_share"]),
                min_per_order=float(commission["min_per_order"]),
                max_pct_of_trade_value=float(commission["max_pct_of_trade_value"]),
                percentage_bps=float(commission["percentage_bps"]),
                fixed_per_order=float(commission["fixed_per_order"]),
            ),
            slippage=SlippageParams(
                model=SlippageModel(slippage["model"]),
                fixed_bps=float(slippage["fixed_bps"]),
                volume_impact_coeff=float(slippage["volume_impact_coeff"]),
                max_participation_rate=float(slippage["max_participation_rate"]),
            ),
            fill_timing=FillTiming(settings["backtest"]["fill_timing"]),
            allow_fractional_shares=bool(execution["allow_fractional_shares"]),
        )


# --------------------------------------------------------------------------- #
# Handler
# --------------------------------------------------------------------------- #
class SimulatedExecutionHandler:
    """
    Simula l'esecuzione degli ordini sulla barra successiva al segnale, con
    slippage sfavorevole e commissioni realistiche.

    Args:
        params: Parametri di esecuzione validati.
    """

    def __init__(self, params: SimulatedExecutionParams) -> None:
        self._params = params
        self._pending: dict[str, OrderEvent] = {}  # order_id -> ordine
        self._discarded: Counter[str] = Counter()

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> SimulatedExecutionHandler:
        """Costruisce l'handler dal dizionario completo di settings.yaml."""
        return cls(SimulatedExecutionParams.from_settings(settings))

    # ------------------------------------------------------------------ #
    # Proprieta' di sola lettura
    # ------------------------------------------------------------------ #
    @property
    def params(self) -> SimulatedExecutionParams:
        """Parametri di esecuzione correnti (immutabili)."""
        return self._params

    @property
    def pending_orders(self) -> tuple[OrderEvent, ...]:
        """Ordini accodati e non ancora valutati."""
        return tuple(self._pending.values())

    @property
    def discarded(self) -> dict[str, int]:
        """Conteggio degli ordini scartati per motivo (diagnostica)."""
        return dict(self._discarded)

    # ------------------------------------------------------------------ #
    # Protocollo dell'engine
    # ------------------------------------------------------------------ #
    def submit_order(self, order: OrderEvent) -> None:
        """
        Accoda l'ordine: NON viene mai eseguito nella barra corrente.

        Raises:
            ValueError: se l'ordine e' duplicato o il simbolo ha gia' un ordine pendente.
        """
        if order.order_id in self._pending:
            raise ValueError(f"Ordine {order.order_id} gia' presente.")
        if any(pending.symbol == order.symbol for pending in self._pending.values()):
            raise ValueError(f"[{order.symbol}] esiste gia' un ordine pendente.")
        self._pending[order.order_id] = order

    def has_pending_orders(self) -> bool:
        """True se esistono ordini accodati non ancora valutati."""
        return bool(self._pending)

    def on_market_event(self, event: MarketEvent) -> list[FillEvent]:
        """
        Valuta gli ordini pendenti del simbolo sulla nuova barra.

        Considera solo ordini con `signal_timestamp < event.timestamp`. Ogni ordine
        valutato esce dai pendenti, eseguito o no (validita': una barra).

        Returns:
            FillEvent per gli ordini eseguiti (anche parzialmente).
        """
        fills: list[FillEvent] = []
        for order_id, order in list(self._pending.items()):
            if order.symbol != event.symbol:
                continue
            if order.signal_timestamp >= event.timestamp:
                continue  # mai nella stessa barra del segnale
            del self._pending[order_id]
            fill = self._try_fill(order, event)
            if fill is not None:
                fills.append(fill)
        return fills

    # ------------------------------------------------------------------ #
    # Modelli di costo (pubblici per test e analisi)
    # ------------------------------------------------------------------ #
    def calculate_slippage_bps(self, quantity: float, bar_volume: float) -> float:
        """
        Slippage in basis points per un'esecuzione di `quantity` azioni.

        Args:
            quantity: Quantita' eseguita (> 0).
            bar_volume: Volume della barra (> 0 per il modello volume_impact).

        Raises:
            ValueError: se il modello e' volume_impact e `bar_volume <= 0`.
        """
        slippage = self._params.slippage
        if slippage.model is SlippageModel.FIXED_BPS:
            return slippage.fixed_bps
        if bar_volume <= 0:
            raise ValueError("Il modello volume_impact richiede un volume di barra > 0.")
        participation = quantity / bar_volume
        bps = slippage.fixed_bps + 10_000.0 * slippage.volume_impact_coeff * participation
        return min(bps, _MAX_SLIPPAGE_BPS)

    def calculate_commission(self, quantity: float, price: float) -> float:
        """
        Commissione totale dell'ordine in valuta base.

        Args:
            quantity: Quantita' eseguita (> 0).
            price: Prezzo di esecuzione (> 0).
        """
        commission = self._params.commission
        if commission.model is CommissionModel.FIXED:
            return commission.fixed_per_order

        trade_value = quantity * price
        if commission.model is CommissionModel.PER_SHARE:
            raw = commission.per_share * quantity
        else:
            raw = trade_value * commission.percentage_bps / 10_000.0
        return min(max(raw, commission.min_per_order), commission.max_pct_of_trade_value * trade_value)

    # ------------------------------------------------------------------ #
    # Logica di esecuzione
    # ------------------------------------------------------------------ #
    def _reference_price(self, event: MarketEvent) -> float:
        """Prezzo di mercato 'pulito' della barra t+1 secondo `fill_timing`."""
        if self._params.fill_timing is FillTiming.NEXT_OPEN:
            return event.open
        return event.close

    def _try_fill(self, order: OrderEvent, event: MarketEvent) -> FillEvent | None:
        """Prova a eseguire l'ordine sulla barra `event`; None se scartato."""
        reference = self._reference_price(event)
        is_buy = order.side is OrderSide.BUY
        limit = order.limit_price if order.order_type is OrderType.LIMIT else None

        if limit is not None and ((is_buy and reference > limit) or (not is_buy and reference < limit)):
            return self._discard("limit_not_reached", order)

        quantity = self._executable_quantity(order, event.volume)
        if quantity <= 0:
            return self._discard("no_liquidity", order)
        if quantity < order.quantity:
            logger.info(
                "[%s] esecuzione parziale: %s su %s (volume barra %s).",
                order.symbol,
                quantity,
                order.quantity,
                event.volume,
            )

        slippage = self.calculate_slippage_bps(quantity, event.volume) / 10_000.0
        fill_price = reference * (1.0 + slippage) if is_buy else reference * (1.0 - slippage)
        if limit is not None:  # il limite non si supera mai
            fill_price = min(fill_price, limit) if is_buy else max(fill_price, limit)

        return FillEvent(
            timestamp=event.timestamp,
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            reference_price=reference,
            fill_price=fill_price,
            commission=self.calculate_commission(quantity, fill_price),
            signal_timestamp=order.signal_timestamp,
            strategy_id=order.strategy_id,
        )

    def _executable_quantity(self, order: OrderEvent, bar_volume: float) -> float:
        """
        Quantita' eseguibile: tutta l'ordine, salvo il tetto di partecipazione
        del modello volume_impact (esecuzione parziale, resto scartato).
        """
        slippage = self._params.slippage
        if slippage.model is not SlippageModel.VOLUME_IMPACT:
            return order.quantity
        if bar_volume <= 0:
            return 0.0

        quantity = min(order.quantity, slippage.max_participation_rate * bar_volume)
        if self._params.allow_fractional_shares:
            return math.floor(quantity * _FRACTIONAL_STEP) / _FRACTIONAL_STEP
        return float(math.floor(quantity))

    def _discard(self, reason: str, order: OrderEvent) -> None:
        """Registra un ordine scartato e restituisce None."""
        self._discarded[reason] += 1
        logger.debug("[%s] ordine %s scartato: %s", order.symbol, order.order_id[:8], reason)
        return None