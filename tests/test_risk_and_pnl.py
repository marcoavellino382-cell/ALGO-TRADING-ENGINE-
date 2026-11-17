"""
tests/test_risk_and_pnl.py

Suite pytest su rischio, PnL, costi di esecuzione e protezioni anti look-ahead.

Esecuzione (dalla root del progetto):
    pip install pytest
    pytest -q

Cosa verifica:
    1. PnL e contabilita': `Position` (costo medio, realizzato netto commissioni,
       non realizzato, inversioni) e identita' contabile del `RiskManager`.
    2. Commissioni: modelli per_share / percentage / fixed con minimo e tetto.
    3. Slippage: modelli fixed_bps / volume_impact, sempre sfavorevole.
    4. Anti look-ahead: `FillEvent`, `SimulatedExecutionHandler`,
       `BaseDataHandler`, `BaseStrategy` e ATR del `RiskManager`.
    5. Position sizing ATR e limiti (posizione, esposizione, cassa, posizioni
       aperte, ordini pendenti).
    6. Kill-switch: attivazione, irreversibilita', liquidazione.
    7. Test end-to-end con l'engine (segnale a t, fill a t+1, warmup, kill-switch)
       e coerenza con `PerformanceAnalyzer` e con `config/settings.yaml`.

Tutti i numeri attesi nei test end-to-end sono ricavati a mano (vedi commenti).
"""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.performance import PerformanceAnalyzer  # noqa: E402
from src.core.engine import EngineConfig, EngineResult, FillTiming, TradingEngine  # noqa: E402
from src.core.events import (  # noqa: E402
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderSide,
    OrderType,
    SignalDirection,
    SignalEvent,
)
from src.data_handler.base import BaseDataHandler  # noqa: E402
from src.execution.simulated import (  # noqa: E402
    CommissionModel,
    CommissionParams,
    SimulatedExecutionHandler,
    SimulatedExecutionParams,
    SlippageModel,
    SlippageParams,
)
from src.risk.risk_manager import (  # noqa: E402
    KillSwitchAction,
    Position,
    RiskManager,
    RiskParams,
    SizingMethod,
)
from src.strategy.base import BaseStrategy, LookAheadError  # noqa: E402
from src.strategy.mean_reversion import MeanReversionParams, MeanReversionStrategy  # noqa: E402

# --------------------------------------------------------------------------- #
# Configurazione di riferimento (specchio di config/settings.yaml)
# --------------------------------------------------------------------------- #
SETTINGS: dict[str, Any] = {
    "general": {"initial_capital": 100_000.0, "base_currency": "USD"},
    "backtest": {"warmup_bars": 200, "fill_timing": "next_open"},
    "execution": {
        "order_type": "market",
        "allow_short": False,
        "allow_fractional_shares": False,
        "commission": {
            "model": "per_share",
            "per_share": 0.005,
            "min_per_order": 1.0,
            "max_pct_of_trade_value": 0.01,
            "percentage_bps": 1.0,
            "fixed_per_order": 1.0,
        },
        "slippage": {
            "model": "fixed_bps",
            "fixed_bps": 5.0,
            "volume_impact_coeff": 0.1,
            "max_participation_rate": 0.10,
        },
    },
    "risk": {
        "max_risk_per_trade": 0.01,
        "max_position_pct": 0.20,
        "max_gross_exposure": 1.0,
        "max_open_positions": 4,
        "max_drawdown_pct": 0.15,
        "kill_switch_action": "liquidate_and_halt",
        "position_sizing": {
            "method": "atr",
            "atr_window": 14,
            "atr_stop_multiple": 2.0,
            "fixed_fraction": 0.10,
            "fixed_shares": 100,
        },
    },
}

DEFAULT_RISK_PARAMS = RiskParams(
    initial_capital=100_000.0,
    max_risk_per_trade=0.01,
    max_position_pct=0.20,
    max_gross_exposure=1.0,
    max_open_positions=4,
    max_drawdown_pct=0.15,
    kill_switch_action=KillSwitchAction.LIQUIDATE_AND_HALT,
    sizing_method=SizingMethod.ATR,
    atr_window=14,
    atr_stop_multiple=2.0,
    fixed_fraction=0.10,
    fixed_shares=100,
    allow_short=False,
    allow_fractional_shares=False,
    order_type=OrderType.MARKET,
    price_buffer_pct=0.02,
)

DEFAULT_EXEC_PARAMS = SimulatedExecutionParams(
    commission=CommissionParams(
        model=CommissionModel.PER_SHARE,
        per_share=0.005,
        min_per_order=1.0,
        max_pct_of_trade_value=0.01,
        percentage_bps=1.0,
        fixed_per_order=1.0,
    ),
    slippage=SlippageParams(
        model=SlippageModel.FIXED_BPS,
        fixed_bps=5.0,
        volume_impact_coeff=0.1,
        max_participation_rate=0.10,
    ),
    fill_timing=FillTiming.NEXT_OPEN,
    allow_fractional_shares=False,
)

T0 = datetime(2024, 1, 1)


# --------------------------------------------------------------------------- #
# Helper e doppioni di test
# --------------------------------------------------------------------------- #
def day(n: int) -> datetime:
    """Timestamp giornaliero: giorno `n` a partire da T0."""
    return T0 + timedelta(days=n)


def make_bar(
    symbol: str,
    n: int,
    *,
    open_: float = 100.0,
    close: float | None = None,
    high: float | None = None,
    low: float | None = None,
    volume: float = 1_000_000.0,
) -> MarketEvent:
    """Barra giornaliera valida; high/low di default avvolgono open e close di 1."""
    close = open_ if close is None else close
    high = max(open_, close) + 1.0 if high is None else high
    low = min(open_, close) - 1.0 if low is None else low
    return MarketEvent(
        timestamp=day(n),
        symbol=symbol,
        timeframe="1d",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


class ListDataHandler(BaseDataHandler):
    """DataHandler di test: emette una lista predefinita di batch, un timestamp alla volta."""

    def __init__(self, symbols: list[str], batches: list[list[MarketEvent]]) -> None:
        super().__init__(symbols, "1d")
        self._batches = list(batches)
        self._cursor = 0

    def has_next(self) -> bool:
        return self._cursor < len(self._batches)

    def _next_batch(self) -> list[MarketEvent]:
        batch = self._batches[self._cursor]
        self._cursor += 1
        return batch


def feed(handler: BaseDataHandler, n: int) -> None:
    """Emette `n` timestamp dal DataHandler (le barre entrano nello storico)."""
    for _ in range(n):
        handler.update_bars()


def atr_handler(symbols: list[str], n_bars: int, half_range: float) -> ListDataHandler:
    """
    Storico con close sempre 100 e high/low a +/- `half_range`: il True Range di
    ogni barra e' `2 x half_range`, quindi anche l'ATR. Le barre sono gia' emesse.
    """
    batches = [
        [make_bar(s, i, high=100.0 + half_range, low=100.0 - half_range) for s in symbols]
        for i in range(n_bars)
    ]
    handler = ListDataHandler(symbols, batches)
    feed(handler, n_bars)
    return handler


def make_signal(
    symbol: str, n: int, direction: SignalDirection = SignalDirection.LONG, price: float = 100.0
) -> SignalEvent:
    """Segnale alla chiusura del giorno `n`."""
    return SignalEvent(
        timestamp=day(n),
        symbol=symbol,
        direction=direction,
        strategy_id="mean_reversion",
        reference_price=price,
    )


def make_fill(
    symbol: str = "SPY",
    side: OrderSide = OrderSide.BUY,
    quantity: float = 100.0,
    price: float = 100.0,
    *,
    reference: float | None = None,
    commission: float = 0.0,
    fill_day: int = 1,
    signal_day: int = 0,
    order_id: str = "order-1",
) -> FillEvent:
    """FillEvent valido (slippage nullo se `reference` non e' indicato)."""
    return FillEvent(
        timestamp=day(fill_day),
        order_id=order_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        reference_price=price if reference is None else reference,
        fill_price=price,
        commission=commission,
        signal_timestamp=day(signal_day),
        strategy_id="mean_reversion",
    )


def make_order(
    symbol: str = "SPY",
    side: OrderSide = OrderSide.BUY,
    quantity: float = 100.0,
    n: int = 0,
    **kwargs: Any,
) -> OrderEvent:
    """Ordine creato alla chiusura del giorno `n`."""
    return OrderEvent(
        timestamp=day(n),
        symbol=symbol,
        side=side,
        quantity=quantity,
        signal_timestamp=day(n),
        strategy_id="mean_reversion",
        **kwargs,
    )


def make_risk(handler: BaseDataHandler, **overrides: Any) -> RiskManager:
    """RiskManager con i parametri di riferimento, eventualmente modificati."""
    return RiskManager(replace(DEFAULT_RISK_PARAMS, **overrides), handler)


def make_execution(**overrides: Any) -> SimulatedExecutionHandler:
    """Handler simulato; `commission`/`slippage` sono sostituibili con replace()."""
    return SimulatedExecutionHandler(replace(DEFAULT_EXEC_PARAMS, **overrides))


class AlwaysLongStrategy(BaseStrategy):
    """Strategia di test: apre un long a ogni barra in cui e' flat."""

    @property
    def required_history(self) -> int:
        return 1

    def generate_signals(self, event: MarketEvent) -> list[SignalEvent]:
        if self.is_flat(event.symbol):
            return [self.make_signal(event, SignalDirection.LONG)]
        return []


class FutureSignalStrategy(BaseStrategy):
    """Strategia scorretta: data il segnale al giorno successivo (look-ahead)."""

    @property
    def required_history(self) -> int:
        return 1

    def generate_signals(self, event: MarketEvent) -> list[SignalEvent]:
        return [
            SignalEvent(
                timestamp=event.timestamp + timedelta(days=1),
                symbol=event.symbol,
                direction=SignalDirection.LONG,
                strategy_id=self.strategy_id,
            )
        ]


def run_engine(
    prices: list[float], *, warmup_bars: int, **risk_overrides: Any
) -> tuple[EngineResult, RiskManager, SimulatedExecutionHandler]:
    """
    Backtest completo su SPY con barre open = close = prezzo, sizing a azioni fisse.
    """
    handler = ListDataHandler(["SPY"], [[make_bar("SPY", i, open_=p)] for i, p in enumerate(prices)])
    strategy = AlwaysLongStrategy("test_strategy", handler)
    risk = make_risk(handler, sizing_method=SizingMethod.FIXED_SHARES, **risk_overrides)
    execution = make_execution()
    engine = TradingEngine(EngineConfig(warmup_bars=warmup_bars), handler, strategy, risk, execution)
    return engine.run(), risk, execution


# =========================================================================== #
# 1. PnL e contabilita'
# =========================================================================== #
def test_position_round_trip_realized_pnl_is_net_of_commission() -> None:
    position = Position("SPY", "s")

    opening = position.apply_fill(100, 100.0, 1.0, day(1))
    assert opening == pytest.approx(-1.0)  # solo la commissione d'ingresso
    assert position.quantity == pytest.approx(100.0)
    assert position.avg_price == pytest.approx(100.0)

    closing = position.apply_fill(-100, 110.0, 1.0, day(2))
    assert closing == pytest.approx(100 * 10.0 - 1.0)
    assert position.quantity == 0.0
    assert position.avg_price == 0.0
    assert position.opened_at is None


def test_position_scale_in_uses_weighted_average_price() -> None:
    position = Position("SPY", "s")
    position.apply_fill(100, 100.0, 0.0, day(1))
    position.apply_fill(100, 110.0, 0.0, day(2))

    assert position.quantity == pytest.approx(200.0)
    assert position.avg_price == pytest.approx(105.0)


def test_position_partial_close_keeps_average_price() -> None:
    position = Position("SPY", "s")
    position.apply_fill(100, 100.0, 0.0, day(1))
    position.apply_fill(100, 110.0, 0.0, day(2))

    realized = position.apply_fill(-50, 120.0, 0.5, day(3))

    assert realized == pytest.approx(50 * (120.0 - 105.0) - 0.5)
    assert position.quantity == pytest.approx(150.0)
    assert position.avg_price == pytest.approx(105.0)


def test_position_unrealized_pnl_and_market_value() -> None:
    position = Position("SPY", "s")
    position.apply_fill(100, 100.0, 0.0, day(1))
    position.last_price = 105.0

    assert position.unrealized_pnl == pytest.approx(500.0)
    assert position.market_value == pytest.approx(10_500.0)


def test_position_short_flip_realizes_pnl_and_resets_average() -> None:
    position = Position("SPY", "s")
    position.apply_fill(-100, 100.0, 0.0, day(1))
    assert position.quantity == pytest.approx(-100.0)

    realized = position.apply_fill(150, 90.0, 0.0, day(2))

    assert realized == pytest.approx(1_000.0)  # 100 azioni short chiuse con +10 ciascuna
    assert position.quantity == pytest.approx(50.0)  # residuo: ora long
    assert position.avg_price == pytest.approx(90.0)


def test_fill_event_cost_and_cash_flow_accounting() -> None:
    buy = make_fill(side=OrderSide.BUY, quantity=100, price=100.05, reference=100.0, commission=1.0)
    assert buy.gross_value == pytest.approx(10_005.0)
    assert buy.slippage_cost == pytest.approx(5.0)
    assert buy.slippage_bps == pytest.approx(5.0)
    assert buy.total_cost == pytest.approx(6.0)
    assert buy.cash_flow == pytest.approx(-10_006.0)
    assert buy.signed_quantity == pytest.approx(100.0)

    sell = make_fill(
        side=OrderSide.SELL, quantity=100, price=99.95, reference=100.0, commission=1.0,
        fill_day=2, signal_day=1,
    )
    assert sell.cash_flow == pytest.approx(9_995.0 - 1.0)
    assert sell.signed_quantity == pytest.approx(-100.0)


def test_risk_manager_fills_update_cash_positions_and_pnl_identity() -> None:
    risk = make_risk(ListDataHandler(["SPY"], []))

    risk.on_fill(make_fill(side=OrderSide.BUY, quantity=100, price=100.0, commission=1.0))
    assert risk.cash == pytest.approx(100_000.0 - 10_000.0 - 1.0)
    assert risk.positions["SPY"].quantity == pytest.approx(100.0)
    assert risk.positions["SPY"].avg_price == pytest.approx(100.0)

    risk.on_market_event(make_bar("SPY", 1, open_=110.0))
    assert risk.equity == pytest.approx(100_999.0)
    assert risk.realized_pnl == pytest.approx(-1.0)
    assert risk.unrealized_pnl == pytest.approx(1_000.0)
    assert risk.equity - 100_000.0 == pytest.approx(risk.realized_pnl + risk.unrealized_pnl)

    risk.on_fill(
        make_fill(
            side=OrderSide.SELL, quantity=100, price=110.0, commission=1.0,
            fill_day=2, signal_day=1, order_id="order-2",
        )
    )
    assert "SPY" not in risk.positions
    assert risk.cash == pytest.approx(100_998.0)
    assert risk.realized_pnl == pytest.approx(998.0)
    assert risk.unrealized_pnl == pytest.approx(0.0)
    assert risk.total_commissions == pytest.approx(2.0)
    assert risk.equity - 100_000.0 == pytest.approx(risk.realized_pnl)


# =========================================================================== #
# 2. Commissioni
# =========================================================================== #
@pytest.mark.parametrize(
    ("model", "quantity", "price", "expected"),
    [
        pytest.param(CommissionModel.PER_SHARE, 1_000, 50.0, 5.0, id="per_share"),
        pytest.param(CommissionModel.PER_SHARE, 10, 50.0, 1.0, id="per_share_minimo"),
        pytest.param(CommissionModel.PER_SHARE, 1, 10.0, 0.1, id="tetto_batte_minimo"),
        pytest.param(CommissionModel.PERCENTAGE, 100, 100.0, 10.0, id="percentuale"),
        pytest.param(CommissionModel.FIXED, 500, 20.0, 2.5, id="fissa"),
    ],
)
def test_commission_models(model: CommissionModel, quantity: float, price: float, expected: float) -> None:
    commission = replace(
        DEFAULT_EXEC_PARAMS.commission, model=model, percentage_bps=10.0, fixed_per_order=2.5
    )
    handler = make_execution(commission=commission)

    assert handler.calculate_commission(quantity, price) == pytest.approx(expected)


# =========================================================================== #
# 3. Slippage
# =========================================================================== #
def test_fixed_slippage_is_constant() -> None:
    handler = make_execution()
    assert handler.calculate_slippage_bps(100, 1_000_000) == pytest.approx(5.0)
    assert handler.calculate_slippage_bps(50_000, 1_000_000) == pytest.approx(5.0)


def test_volume_impact_slippage_adds_participation_cost() -> None:
    handler = make_execution(slippage=replace(DEFAULT_EXEC_PARAMS.slippage, model=SlippageModel.VOLUME_IMPACT))

    # partecipazione 1% -> 10.000 x 0,1 x 0,01 = 10 bps, piu' 5 bps di costo base
    assert handler.calculate_slippage_bps(1_000, 100_000) == pytest.approx(15.0)


def test_volume_impact_slippage_requires_positive_volume() -> None:
    handler = make_execution(slippage=replace(DEFAULT_EXEC_PARAMS.slippage, model=SlippageModel.VOLUME_IMPACT))

    with pytest.raises(ValueError):
        handler.calculate_slippage_bps(100, 0.0)


def test_buy_fill_has_adverse_slippage_and_commission_at_next_open() -> None:
    handler = make_execution()
    handler.submit_order(make_order(side=OrderSide.BUY, quantity=100, n=0))

    fills = handler.on_market_event(make_bar("SPY", 1, open_=100.0, close=101.0))

    assert len(fills) == 1
    fill = fills[0]
    assert fill.timestamp == day(1)
    assert fill.reference_price == pytest.approx(100.0)  # open della barra t+1
    assert fill.fill_price == pytest.approx(100.05)  # +5 bps
    assert fill.slippage_bps == pytest.approx(5.0)
    assert fill.slippage_cost == pytest.approx(5.0)
    assert fill.commission == pytest.approx(1.0)  # 100 x 0,005 = 0,5 -> minimo 1,00
    assert not handler.has_pending_orders()


def test_sell_fill_has_adverse_slippage() -> None:
    handler = make_execution()
    handler.submit_order(make_order(side=OrderSide.SELL, quantity=100, n=0))

    fill = handler.on_market_event(make_bar("SPY", 1, open_=100.0, close=101.0))[0]

    assert fill.fill_price == pytest.approx(99.95)  # -5 bps
    assert fill.fill_price < fill.reference_price


def test_next_close_timing_fills_at_close_of_next_bar() -> None:
    handler = make_execution(fill_timing=FillTiming.NEXT_CLOSE)
    handler.submit_order(make_order(n=0))

    fill = handler.on_market_event(make_bar("SPY", 1, open_=100.0, close=102.0))[0]

    assert fill.reference_price == pytest.approx(102.0)
    assert fill.fill_price == pytest.approx(102.051)


def test_volume_impact_caps_participation_and_fills_partially() -> None:
    slippage = replace(DEFAULT_EXEC_PARAMS.slippage, model=SlippageModel.VOLUME_IMPACT)
    handler = make_execution(slippage=slippage)
    handler.submit_order(make_order(quantity=500, n=0))

    fill = handler.on_market_event(make_bar("SPY", 1, open_=100.0, volume=1_000.0))[0]

    assert fill.quantity == pytest.approx(100.0)  # 10% del volume della barra
    assert fill.slippage_bps == pytest.approx(105.0)  # 5 + 10.000 x 0,1 x (100/1.000)
    assert fill.fill_price == pytest.approx(101.05)
    assert not handler.has_pending_orders()  # il resto viene scartato


def test_volume_impact_with_zero_volume_does_not_fill() -> None:
    slippage = replace(DEFAULT_EXEC_PARAMS.slippage, model=SlippageModel.VOLUME_IMPACT)
    handler = make_execution(slippage=slippage)
    handler.submit_order(make_order(n=0))

    assert handler.on_market_event(make_bar("SPY", 1, volume=0.0)) == []
    assert handler.discarded == {"no_liquidity": 1}


def test_limit_buy_not_reached_is_discarded() -> None:
    handler = make_execution()
    handler.submit_order(make_order(n=0, order_type=OrderType.LIMIT, limit_price=99.0))

    assert handler.on_market_event(make_bar("SPY", 1, open_=100.0)) == []
    assert handler.discarded == {"limit_not_reached": 1}
    assert not handler.has_pending_orders()


def test_limit_buy_never_pays_more_than_limit() -> None:
    handler = make_execution()
    handler.submit_order(make_order(n=0, order_type=OrderType.LIMIT, limit_price=100.02))

    fill = handler.on_market_event(make_bar("SPY", 1, open_=100.0))[0]

    assert fill.fill_price == pytest.approx(100.02)  # 100,05 limitato al limite
    assert fill.fill_price >= fill.reference_price


# =========================================================================== #
# 4. Anti look-ahead bias
# =========================================================================== #
@pytest.mark.parametrize("fill_day", [0, -1], ids=["stessa_barra", "barra_precedente"])
def test_fill_event_rejects_fill_not_after_signal(fill_day: int) -> None:
    with pytest.raises(ValueError, match="Look-ahead"):
        make_fill(fill_day=fill_day, signal_day=0)


def test_fill_event_rejects_favorable_slippage() -> None:
    with pytest.raises(ValueError, match="slippage"):
        make_fill(side=OrderSide.BUY, price=99.9, reference=100.0)


def test_execution_never_fills_an_order_on_the_signal_bar() -> None:
    handler = make_execution()
    handler.submit_order(make_order(n=0))

    assert handler.on_market_event(make_bar("SPY", 0)) == []  # stessa barra del segnale
    assert handler.has_pending_orders()

    fills = handler.on_market_event(make_bar("SPY", 1))  # barra successiva
    assert len(fills) == 1
    assert fills[0].timestamp > fills[0].signal_timestamp


def test_execution_rejects_second_pending_order_on_same_symbol() -> None:
    handler = make_execution()
    handler.submit_order(make_order(n=0))

    with pytest.raises(ValueError, match="pendente"):
        handler.submit_order(make_order(n=0))


def test_data_handler_exposes_only_already_emitted_bars() -> None:
    batches = [[make_bar("SPY", i)] for i in range(5)]
    handler = ListDataHandler(["SPY"], batches)
    feed(handler, 3)

    bars = handler.get_latest_bars("SPY", 10)

    assert len(bars) == 3
    assert bars.index[-1] == day(2)
    assert handler.current_timestamp == day(2)


def test_data_handler_rejects_non_increasing_timestamps() -> None:
    handler = ListDataHandler(["SPY"], [[make_bar("SPY", 1)], [make_bar("SPY", 1)]])
    handler.update_bars()

    with pytest.raises(ValueError, match="strettamente crescenti"):
        handler.update_bars()


def test_strategy_history_is_none_when_too_short() -> None:
    handler = atr_handler(["SPY"], 3, 1.0)
    strategy = AlwaysLongStrategy("test_strategy", handler)

    assert strategy.get_history("SPY", 5, day(2)) is None
    history = strategy.get_history("SPY", 3, day(2))
    assert history is not None and len(history) == 3


def test_strategy_history_blocks_future_bars() -> None:
    handler = atr_handler(["SPY"], 3, 1.0)
    strategy = AlwaysLongStrategy("test_strategy", handler)

    with pytest.raises(LookAheadError):
        strategy.get_history("SPY", 3, as_of=day(1))  # i dati arrivano al giorno 2


def test_strategy_rejects_signal_stamped_in_the_future() -> None:
    handler = atr_handler(["SPY"], 3, 1.0)
    strategy = FutureSignalStrategy("test_strategy", handler)

    with pytest.raises(LookAheadError):
        strategy.on_market_event(make_bar("SPY", 2))


def test_risk_atr_uses_only_bars_up_to_signal_timestamp() -> None:
    handler = atr_handler(["SPY"], 15, 5.0)  # storico fino al giorno 14
    risk = make_risk(handler)

    with pytest.raises(LookAheadError):
        risk.on_signal(make_signal("SPY", 10))


# =========================================================================== #
# 5. Position sizing ATR e limiti
# =========================================================================== #
def test_atr_sizing_risks_one_percent_of_equity() -> None:
    handler = atr_handler(["SPY"], 15, half_range=5.0)  # ATR = 10
    risk = make_risk(handler)

    orders = risk.on_signal(make_signal("SPY", 14))

    assert len(orders) == 1
    order = orders[0]
    assert order.side is OrderSide.BUY
    assert order.order_type is OrderType.MARKET
    assert order.signal_timestamp == day(14)
    # 100.000 x 1% / (2,0 x ATR 10) = 50 azioni
    assert order.quantity == pytest.approx(50.0)
    assert order.stop_price == pytest.approx(80.0)  # 100 - 2 x 10
    assert order.quantity * (100.0 - order.stop_price) == pytest.approx(1_000.0)
    assert len(risk.pending_orders) == 1


def test_atr_sizing_scales_with_equity() -> None:
    handler = atr_handler(["SPY"], 15, half_range=5.0)
    risk = make_risk(handler, initial_capital=200_000.0)

    order = risk.on_signal(make_signal("SPY", 14))[0]

    assert order.quantity == pytest.approx(100.0)  # 200.000 x 1% / 20


def test_position_cap_limits_quantity_when_atr_is_small() -> None:
    handler = atr_handler(["SPY"], 15, half_range=1.0)  # ATR = 2 -> target 250 azioni
    risk = make_risk(handler)

    order = risk.on_signal(make_signal("SPY", 14))[0]

    # tetto del 20%: 20.000 / (100 x 1,02) = 196,08 -> 196 azioni
    assert order.quantity == pytest.approx(196.0)


@pytest.mark.parametrize(
    ("allow_fractional", "expected"),
    [(False, 83.0), (True, 83.3333)],
    ids=["intere", "frazionarie"],
)
def test_quantity_rounding_follows_fractional_setting(allow_fractional: bool, expected: float) -> None:
    handler = atr_handler(["SPY"], 15, half_range=3.0)  # ATR = 6 -> 1.000 / 12 = 83,33
    risk = make_risk(handler, allow_fractional_shares=allow_fractional)

    order = risk.on_signal(make_signal("SPY", 14))[0]

    assert order.quantity == pytest.approx(expected, abs=1e-4)


def test_signal_is_rejected_when_size_rounds_to_zero() -> None:
    handler = atr_handler(["SPY"], 15, half_range=5.0)
    risk = make_risk(handler, initial_capital=500.0)  # 500 x 1% / 20 = 0,25 azioni

    assert risk.on_signal(make_signal("SPY", 14)) == []
    assert risk.rejections == {"size_below_minimum": 1}


def test_signal_is_rejected_when_atr_is_unavailable() -> None:
    handler = atr_handler(["SPY"], 5, half_range=5.0)  # servono 15 barre
    risk = make_risk(handler)

    assert risk.on_signal(make_signal("SPY", 4)) == []
    assert risk.rejections == {"atr_unavailable": 1}


def test_short_signal_is_rejected_when_not_allowed() -> None:
    risk = make_risk(atr_handler(["SPY"], 15, 5.0))

    assert risk.on_signal(make_signal("SPY", 14, SignalDirection.SHORT)) == []
    assert risk.rejections == {"short_not_allowed": 1}


def test_max_open_positions_counts_pending_entries() -> None:
    risk = make_risk(atr_handler(["SPY", "QQQ"], 15, 5.0), max_open_positions=1)

    assert len(risk.on_signal(make_signal("SPY", 14))) == 1  # ordine ancora pendente
    assert risk.on_signal(make_signal("QQQ", 14)) == []
    assert risk.rejections == {"max_open_positions": 1}


def test_pending_orders_reserve_exposure_and_cash() -> None:
    handler = atr_handler(["SPY", "QQQ"], 15, 5.0)
    risk = make_risk(
        handler,
        sizing_method=SizingMethod.FIXED_SHARES,
        fixed_shares=1_000,
        max_position_pct=0.6,
    )

    first = risk.on_signal(make_signal("SPY", 14))[0]
    second = risk.on_signal(make_signal("QQQ", 14))[0]

    assert first.quantity == pytest.approx(588.0)  # 60.000 / 102
    # restano 100.000 - 588 x 100 x 1,02 = 40.024 impegnabili -> 392 azioni
    assert second.quantity == pytest.approx(392.0)


def test_second_signal_on_same_symbol_is_blocked_while_order_is_pending() -> None:
    risk = make_risk(atr_handler(["SPY"], 15, 5.0))

    assert len(risk.on_signal(make_signal("SPY", 14))) == 1
    assert risk.on_signal(make_signal("SPY", 14)) == []
    assert risk.rejections == {"pending_order_exists": 1}


def test_pending_order_expires_on_the_next_bar_of_the_symbol() -> None:
    risk = make_risk(atr_handler(["SPY"], 15, 5.0))
    risk.on_signal(make_signal("SPY", 14))

    risk.on_market_event(make_bar("SPY", 15))

    assert risk.pending_orders == ()
    assert risk.rejections == {"order_expired": 1}


def test_fill_removes_pending_order_and_updates_cash() -> None:
    risk = make_risk(atr_handler(["SPY"], 15, 5.0))
    order = risk.on_signal(make_signal("SPY", 14))[0]

    risk.on_fill(
        make_fill(
            quantity=order.quantity, price=100.0, commission=1.0,
            fill_day=15, signal_day=14, order_id=order.order_id,
        )
    )

    assert risk.pending_orders == ()
    assert risk.cash == pytest.approx(100_000.0 - 50 * 100.0 - 1.0)


def test_exit_signal_closes_the_full_position_at_market() -> None:
    risk = make_risk(atr_handler(["SPY"], 15, 5.0))
    risk.on_fill(make_fill(quantity=120, price=100.0, fill_day=1, signal_day=0))

    orders = risk.on_signal(make_signal("SPY", 14, SignalDirection.EXIT))

    assert len(orders) == 1
    assert orders[0].side is OrderSide.SELL
    assert orders[0].quantity == pytest.approx(120.0)
    assert orders[0].order_type is OrderType.MARKET


def test_exit_signal_without_position_is_rejected() -> None:
    risk = make_risk(atr_handler(["SPY"], 15, 5.0))

    assert risk.on_signal(make_signal("SPY", 14, SignalDirection.EXIT)) == []
    assert risk.rejections == {"exit_without_position": 1}


@pytest.mark.parametrize(
    "override",
    [
        {"max_drawdown_pct": 1.0},
        {"max_open_positions": 0},
        {"max_risk_per_trade": 0.0},
        {"max_position_pct": 1.5},
        {"atr_stop_multiple": 0.0},
        {"initial_capital": -1.0},
    ],
)
def test_risk_params_reject_invalid_values(override: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        replace(DEFAULT_RISK_PARAMS, **override)


# =========================================================================== #
# 6. Kill-switch
# =========================================================================== #
def _risk_with_full_position(**overrides: Any) -> RiskManager:
    """RiskManager con 1.000 azioni SPY a 100 (cassa zero, equity 100.000)."""
    risk = make_risk(ListDataHandler(["SPY"], []), **overrides)
    risk.on_fill(make_fill(quantity=1_000, price=100.0))
    return risk


def test_kill_switch_triggers_when_drawdown_reaches_limit() -> None:
    risk = _risk_with_full_position()

    risk.on_market_event(make_bar("SPY", 1, open_=100.0))
    snapshot = risk.snapshot(day(1))
    assert snapshot.drawdown == pytest.approx(0.0)
    assert not risk.kill_switch_triggered

    risk.on_market_event(make_bar("SPY", 2, open_=90.0))
    snapshot = risk.snapshot(day(2))
    assert snapshot.drawdown == pytest.approx(0.10)  # sotto la soglia del 15%
    assert snapshot.gross_exposure == pytest.approx(1.0)
    assert snapshot.positions == {"SPY": pytest.approx(1_000.0)}
    assert not risk.kill_switch_triggered

    risk.on_market_event(make_bar("SPY", 3, open_=84.0))
    snapshot = risk.snapshot(day(3))
    assert snapshot.drawdown == pytest.approx(0.16)
    assert risk.kill_switch_triggered


def test_kill_switch_is_irreversible_and_blocks_new_orders() -> None:
    risk = _risk_with_full_position()
    risk.on_market_event(make_bar("SPY", 1, open_=100.0))
    risk.snapshot(day(1))
    risk.on_market_event(make_bar("SPY", 2, open_=84.0))
    risk.snapshot(day(2))
    assert risk.kill_switch_triggered

    risk.on_market_event(make_bar("SPY", 3, open_=100.0))  # il prezzo recupera
    risk.snapshot(day(3))

    assert risk.kill_switch_triggered
    assert risk.on_signal(make_signal("QQQ", 3)) == []
    assert risk.rejections == {"kill_switch_active": 1}


def test_kill_switch_liquidation_orders_close_everything_at_market() -> None:
    risk = _risk_with_full_position()
    risk.on_market_event(make_bar("SPY", 1, open_=100.0))
    risk.snapshot(day(1))
    risk.on_market_event(make_bar("SPY", 2, open_=84.0))
    risk.snapshot(day(2))

    orders = risk.liquidation_orders(day(2))

    assert len(orders) == 1
    order = orders[0]
    assert order.side is OrderSide.SELL
    assert order.quantity == pytest.approx(1_000.0)
    assert order.order_type is OrderType.MARKET
    assert order.signal_timestamp == day(2)
    assert order.strategy_id == "mean_reversion"  # lo stesso che ha aperto la posizione
    assert order in risk.pending_orders


def test_kill_switch_halt_only_does_not_liquidate() -> None:
    risk = _risk_with_full_position(kill_switch_action=KillSwitchAction.HALT_ONLY)
    risk.on_market_event(make_bar("SPY", 1, open_=100.0))
    risk.snapshot(day(1))
    risk.on_market_event(make_bar("SPY", 2, open_=84.0))
    risk.snapshot(day(2))

    assert risk.kill_switch_triggered
    assert risk.liquidation_orders(day(2)) == []


# =========================================================================== #
# 7. End-to-end con l'engine
# =========================================================================== #
def test_engine_executes_signal_on_next_bar_and_respects_warmup() -> None:
    # open = close = prezzo; sizing 10 azioni; warmup 3 -> primo segnale alla 3a barra
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    result, risk, _ = run_engine(prices, warmup_bars=3, fixed_shares=10)

    # Segnale alla chiusura del giorno 2, nessun segnale prima del warmup
    assert len(result.signals) == 1
    assert result.signals[0].timestamp == day(2)
    assert result.orders[0].timestamp == day(2)

    # Fill alla barra successiva (giorno 3), all'open 103 piu' 5 bps
    assert len(result.fills) == 1
    fill = result.fills[0]
    assert fill.signal_timestamp == day(2)
    assert fill.timestamp == day(3)
    assert fill.reference_price == pytest.approx(103.0)
    assert fill.fill_price == pytest.approx(103.0515)
    assert fill.commission == pytest.approx(1.0)
    assert all(f.timestamp > f.signal_timestamp for f in result.fills)

    # Equity finale = cassa + valore di mercato, con costi inclusi
    assert result.bars_processed == 6
    assert len(result.snapshots) == 6
    assert result.snapshots[0].equity == pytest.approx(100_000.0)
    assert result.snapshots[1].equity == pytest.approx(100_000.0)
    expected_equity = 100_000.0 - 10 * 103.0515 - 1.0 + 10 * 105.0
    assert result.snapshots[-1].equity == pytest.approx(expected_equity)
    assert risk.equity == pytest.approx(expected_equity)
    assert not result.halted


def test_analyzer_excludes_warmup_snapshots_from_metrics() -> None:
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    result, _, _ = run_engine(prices, warmup_bars=3, fixed_shares=10)

    analyzer = PerformanceAnalyzer(
        result.snapshots,
        result.fills,
        initial_capital=100_000.0,
        start_date=day(2),
        timeframe="1d",
    )

    assert analyzer.summary()["bars"] == 4  # giorni 2..5: i primi due sono warmup
    assert analyzer.equity_curve().index[0] == day(2)
    assert analyzer.summary()["initial_equity"] == pytest.approx(100_000.0)


def test_engine_halts_and_liquidates_when_drawdown_breaches_limit() -> None:
    # Segnale al giorno 0, fill al giorno 1: 900 azioni a 100,05 (cassa 9.950,50).
    # Giorno 2 (prezzo 90): drawdown 9,05%. Giorno 3 (prezzo 80): drawdown 18,05%
    # -> kill-switch; liquidazione eseguita all'open del giorno 4 (75 - 5 bps).
    prices = [100.0, 100.0, 90.0, 80.0, 75.0, 70.0, 65.0, 60.0]
    result, risk, execution = run_engine(
        prices, warmup_bars=1, fixed_shares=900, max_position_pct=1.0, max_gross_exposure=1.0
    )

    assert result.halted
    assert result.halt_timestamp == day(3)
    assert risk.kill_switch_triggered
    assert result.bars_processed == 5  # si ferma a liquidazione eseguita
    assert result.snapshots[2].drawdown < 0.15
    assert result.snapshots[3].drawdown > 0.15

    assert len(result.signals) == 1  # dopo l'arresto la Strategy non viene piu' interpellata
    assert [o.side for o in result.orders] == [OrderSide.BUY, OrderSide.SELL]
    assert [f.side for f in result.fills] == [OrderSide.BUY, OrderSide.SELL]
    assert result.fills[-1].timestamp == day(4)
    assert result.fills[-1].fill_price == pytest.approx(74.9625)

    assert risk.positions == {}
    assert not execution.has_pending_orders()
    assert risk.cash == pytest.approx(77_412.25)  # 9.950,50 + 900 x 74,9625 - 4,50
    assert risk.realized_pnl == pytest.approx(-22_587.75)
    assert risk.total_commissions == pytest.approx(9.0)


def test_analyzer_trade_log_and_costs_match_the_engine_run() -> None:
    prices = [100.0, 100.0, 90.0, 80.0, 75.0, 70.0, 65.0, 60.0]
    result, risk, _ = run_engine(
        prices, warmup_bars=1, fixed_shares=900, max_position_pct=1.0, max_gross_exposure=1.0
    )

    analyzer = PerformanceAnalyzer.from_result(
        result, initial_capital=100_000.0, start_date=None, timeframe="1d"
    )
    summary = analyzer.summary()
    trade = analyzer.trade_log().iloc[0]

    assert summary["n_trades"] == 1
    assert trade["direction"] == "LONG"
    assert trade["quantity"] == pytest.approx(900.0)
    assert trade["entry_price"] == pytest.approx(100.05)
    assert trade["exit_price"] == pytest.approx(74.9625)
    # 900 x (74,9625 - 100,05) - commissioni 9,00
    assert trade["pnl"] == pytest.approx(-22_587.75)
    assert trade["pnl"] == pytest.approx(risk.realized_pnl)

    # Il PnL del trade spiega esattamente la variazione di equity (posizione chiusa)
    assert summary["final_equity"] - summary["initial_equity"] == pytest.approx(trade["pnl"])

    assert summary["total_commissions"] == pytest.approx(9.0)
    assert summary["total_slippage"] == pytest.approx(78.75)  # 900 x 0,05 + 900 x 0,0375
    assert summary["total_costs"] == pytest.approx(87.75)
    assert summary["max_drawdown"] >= 0.15
    assert summary["n_open_positions"] == 0


# =========================================================================== #
# 8. Coerenza con la configurazione
# =========================================================================== #
def test_params_are_read_from_settings_dict() -> None:
    assert RiskParams.from_settings(SETTINGS) == DEFAULT_RISK_PARAMS
    assert SimulatedExecutionParams.from_settings(SETTINGS) == DEFAULT_EXEC_PARAMS
    assert EngineConfig.from_settings(SETTINGS) == EngineConfig(
        warmup_bars=200, fill_timing=FillTiming.NEXT_OPEN
    )


def test_real_settings_yaml_builds_every_component() -> None:
    path = PROJECT_ROOT / "config" / "settings.yaml"
    if not path.is_file():
        pytest.skip("config/settings.yaml non trovato")
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))

    RiskParams.from_settings(settings)
    SimulatedExecutionParams.from_settings(settings)
    engine_config = EngineConfig.from_settings(settings)
    strategy = MeanReversionStrategy(
        ListDataHandler(list(settings["universe"]["symbols"]), []),
        MeanReversionParams.from_settings(settings),
    )

    strategy.ensure_warmup(engine_config.warmup_bars)  # warmup coerente con gli indicatori