"""
src/analytics/performance.py

`PerformanceAnalyzer`: analisi quantitativa di un backtest.

Input (tutti disponibili in `EngineResult`):
    - snapshots: equity curve registrata dall'engine con `RiskManager.snapshot()`;
    - fills: storia dei `FillEvent` (con commissioni e slippage);
    - initial_capital, start_date e timeframe da settings.yaml.

Output:
    - `summary()`: dizionario piatto di metriche, pronto per JSON/Streamlit/SQLite;
    - `equity_curve()`, `drawdown_series()`, `trade_log()`: DataFrame formattati;
    - `to_sqlite()`: salvataggio opzionale di tutto in SQLite.

Esclusione del warmup:
    Gli snapshot con `timestamp < start_date` (barre di riscaldamento, equity
    piatta al capitale iniziale) sono esclusi da rendimenti, volatilita', Sharpe,
    Sortino, CAGR e drawdown. L'ultimo snapshot di warmup fa da livello di
    partenza: il primo rendimento utile e' quello della prima barra >= start_date.
    Senza snapshot di warmup, il livello di partenza e' `initial_capital`.

Convenzioni:
    - Rendimenti, drawdown, win rate e CAGR sono frazioni (0.12 = 12%).
    - Il drawdown e' positivo (0.15 = -15% dal massimo), come in `PortfolioSnapshot`.
    - Annualizzazione secondo il timeframe (vedi `PERIODS_PER_YEAR`).
    - Una metrica non definita (es. Sharpe con volatilita' nulla, profit factor
      senza perdite) vale `None`, mai NaN o infinito, per restare serializzabile.

Trade log:
    Un trade va da quando la posizione su un simbolo passa da flat a non-flat
    fino al ritorno a flat. Il PnL usa lo stesso prezzo medio di carico del
    `RiskManager` (`Position.apply_fill`) ed e' al netto delle commissioni, cosi'
    i trade chiusi sono coerenti con il PnL realizzato del portafoglio. Le
    posizioni ancora aperte a fine backtest non compaiono nel log (vedi
    `n_open_positions`). Un'inversione di posizione chiude il trade e ne apre
    uno nuovo con la quantita' residua.

Allineamento con config/settings.yaml:
    general.initial_capital, backtest.start_date, universe.timeframe
    (opzionale: analytics.risk_free_rate, default 0.0).
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd

from src.core.engine import EngineResult, PortfolioSnapshot
from src.core.events import FillEvent
from src.risk.risk_manager import Position

__all__ = ["PERIODS_PER_YEAR", "PerformanceAnalyzer"]

#: Barre per anno per timeframe (252 sedute; 6,5 ore di mercato al giorno).
PERIODS_PER_YEAR: Final[Mapping[str, int]] = {
    "1d": 252,
    "1h": 1_638,
    "15m": 6_552,
    "5m": 19_656,
    "1m": 98_280,
}

_SECONDS_PER_YEAR: Final[float] = 365.25 * 24 * 3600
_EPS: Final[float] = 1e-9

_TRADE_COLUMNS: Final[list[str]] = [
    "trade_id",
    "symbol",
    "strategy_id",
    "direction",
    "entry_time",
    "exit_time",
    "holding_days",
    "quantity",
    "entry_price",
    "exit_price",
    "pnl",
    "pnl_pct",
    "commissions",
    "slippage_cost",
]


def _clean(value: float | None) -> float | None:
    """Restituisce None per valori assenti, NaN o infiniti; altrimenti un float."""
    if value is None or not math.isfinite(value):
        return None
    return float(value)


@dataclass(slots=True)
class _OpenTrade:
    """Accumulatore di un trade in corso su un simbolo."""

    symbol: str
    strategy_id: str
    direction: str
    entry_time: datetime
    entry_qty: float = 0.0
    entry_value: float = 0.0
    exit_qty: float = 0.0
    exit_value: float = 0.0
    pnl: float = 0.0
    commissions: float = 0.0
    slippage_cost: float = 0.0

    def to_row(self, exit_time: datetime, trade_id: int) -> dict[str, Any]:
        """Riga del trade log per un trade chiuso."""
        entry_price = self.entry_value / self.entry_qty
        exit_price = self.exit_value / self.exit_qty
        return {
            "trade_id": trade_id,
            "symbol": self.symbol,
            "strategy_id": self.strategy_id,
            "direction": self.direction,
            "entry_time": self.entry_time,
            "exit_time": exit_time,
            "holding_days": (exit_time - self.entry_time).total_seconds() / 86_400.0,
            "quantity": self.entry_qty,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": self.pnl,
            "pnl_pct": self.pnl / self.entry_value,
            "commissions": self.commissions,
            "slippage_cost": self.slippage_cost,
        }


class PerformanceAnalyzer:
    """
    Calcola metriche di performance, serie e trade log di un backtest.

    Args:
        snapshots: Equity curve (uno snapshot per barra elaborata dall'engine).
        fills: Storia di tutti i FillEvent, in ordine cronologico.
        initial_capital: Capitale iniziale (`general.initial_capital`).
        start_date: Prima data del backtest (`backtest.start_date`): le barre
            precedenti sono warmup ed escluse dalle metriche. None = nessuna esclusione.
        timeframe: Granularita' delle barre (`universe.timeframe`).
        risk_free_rate: Tasso privo di rischio annuo (frazione) per Sharpe e Sortino.

    Raises:
        ValueError: input incoerenti o nessuno snapshot da `start_date` in poi.
    """

    def __init__(
        self,
        snapshots: Sequence[PortfolioSnapshot],
        fills: Sequence[FillEvent],
        *,
        initial_capital: float,
        start_date: str | date | datetime | None = None,
        timeframe: str = "1d",
        risk_free_rate: float = 0.0,
    ) -> None:
        if initial_capital <= 0:
            raise ValueError("'initial_capital' deve essere > 0.")
        if timeframe not in PERIODS_PER_YEAR:
            raise ValueError(
                f"Timeframe '{timeframe}' non supportato: usa uno tra {sorted(PERIODS_PER_YEAR)}."
            )
        if not snapshots:
            raise ValueError("Nessuno snapshot: l'equity curve e' vuota.")

        self._initial_capital = float(initial_capital)
        self._periods_per_year = PERIODS_PER_YEAR[timeframe]
        self._risk_free_rate = float(risk_free_rate)
        self._start_ts = pd.Timestamp(start_date) if start_date is not None else None
        self._fills = sorted(fills, key=lambda fill: fill.timestamp)

        self._equity_curve: pd.DataFrame
        self._drawdown: pd.DataFrame
        self._returns: np.ndarray
        self._base_equity: float
        self._base_ts: pd.Timestamp | None
        self._build_series(snapshots)

        self._trades: pd.DataFrame
        self._n_open_positions: int
        self._trades, self._n_open_positions = self._build_trades()

        self._summary: dict[str, Any] = self._build_summary()

    # ------------------------------------------------------------------ #
    # Costruttori da risultato dell'engine e da settings.yaml
    # ------------------------------------------------------------------ #
    @classmethod
    def from_result(
        cls,
        result: EngineResult,
        *,
        initial_capital: float,
        start_date: str | date | datetime | None = None,
        timeframe: str = "1d",
        risk_free_rate: float = 0.0,
    ) -> PerformanceAnalyzer:
        """Costruisce l'analizzatore da un `EngineResult` (snapshots + fills)."""
        return cls(
            result.snapshots,
            result.fills,
            initial_capital=initial_capital,
            start_date=start_date,
            timeframe=timeframe,
            risk_free_rate=risk_free_rate,
        )

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any], result: EngineResult) -> PerformanceAnalyzer:
        """
        Costruisce l'analizzatore da settings.yaml e dal risultato dell'engine.

        Args:
            settings: Contenuto di settings.yaml gia' caricato (`yaml.safe_load`).
            result: Risultato di `TradingEngine.run()`.
        """
        return cls.from_result(
            result,
            initial_capital=float(settings["general"]["initial_capital"]),
            start_date=settings["backtest"]["start_date"],
            timeframe=str(settings["universe"]["timeframe"]),
            risk_free_rate=float(settings.get("analytics", {}).get("risk_free_rate", 0.0)),
        )

    # ------------------------------------------------------------------ #
    # Output pubblico
    # ------------------------------------------------------------------ #
    def summary(self) -> dict[str, Any]:
        """
        Dizionario piatto delle metriche (copia).

        Chiavi: start, end, bars, initial_equity, final_equity, total_return, cagr,
        annualized_volatility, sharpe_ratio, sortino_ratio, max_drawdown,
        calmar_ratio, n_trades, win_rate, profit_factor, avg_trade_pnl, avg_win,
        avg_loss, total_commissions, total_slippage, total_costs,
        n_open_positions, periods_per_year, risk_free_rate.
        """
        return dict(self._summary)

    def equity_curve(self) -> pd.DataFrame:
        """
        Equity curve dalla `start_date`, indicizzata per timestamp.

        Colonne: equity, cash, gross_exposure, returns (rendimento per barra),
        cumulative_return (rispetto al livello di partenza).
        """
        return self._equity_curve.copy()

    def drawdown_series(self) -> pd.DataFrame:
        """
        Serie del drawdown dalla `start_date`, indicizzata per timestamp.

        Colonne: equity, peak_equity (massimo corrente), drawdown (frazione >= 0).
        """
        return self._drawdown.copy()

    def trade_log(self) -> pd.DataFrame:
        """
        Trade chiusi, ordinati per data di uscita.

        Colonne: trade_id, symbol, strategy_id, direction, entry_time, exit_time,
        holding_days, quantity, entry_price, exit_price, pnl (netto commissioni),
        pnl_pct (sul controvalore d'ingresso), commissions, slippage_cost.
        """
        return self._trades.copy()

    def to_sqlite(
        self,
        db_path: str | Path,
        run_id: str,
        if_exists: Literal["append", "replace", "fail"] = "append",
    ) -> None:
        """
        Salva riepilogo, equity curve, drawdown e trade log in SQLite.

        Tabelle: `perf_summary`, `perf_equity_curve`, `perf_drawdown`,
        `perf_trades`, ciascuna con la colonna `run_id` per distinguere piu' esecuzioni.

        Args:
            db_path: Percorso del database (creato se assente).
            run_id: Identificativo dell'esecuzione (non vuoto).
            if_exists: Comportamento se la tabella esiste (`to_sql` di pandas).
        """
        if not run_id.strip():
            raise ValueError("'run_id' non puo' essere vuoto.")
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        tables: dict[str, pd.DataFrame] = {
            "perf_summary": pd.DataFrame([self._summary]),
            "perf_equity_curve": self._equity_curve.reset_index(),
            "perf_drawdown": self._drawdown.reset_index(),
            "perf_trades": self._trades,
        }
        with closing(sqlite3.connect(path)) as conn:
            for name, frame in tables.items():
                out = frame.copy()
                out.insert(0, "run_id", run_id)
                out.to_sql(name, conn, if_exists=if_exists, index=False)
            conn.commit()

    # ------------------------------------------------------------------ #
    # Serie: equity, rendimenti, drawdown
    # ------------------------------------------------------------------ #
    def _build_series(self, snapshots: Sequence[PortfolioSnapshot]) -> None:
        """Isola la finestra dalla start_date e costruisce equity curve e drawdown."""
        frame = pd.DataFrame(
            {
                "equity": [s.equity for s in snapshots],
                "cash": [s.cash for s in snapshots],
                "gross_exposure": [s.gross_exposure for s in snapshots],
            },
            index=pd.Index([s.timestamp for s in snapshots], name="timestamp"),
        ).sort_index()

        if self._start_ts is None:
            pre = frame.iloc[0:0]
            window = frame
        else:
            pre = frame[frame.index < self._start_ts]
            window = frame[frame.index >= self._start_ts]
        if window.empty:
            raise ValueError("Nessuno snapshot a partire da start_date: niente da analizzare.")

        self._base_equity = float(pre["equity"].iloc[-1]) if not pre.empty else self._initial_capital
        self._base_ts = pre.index[-1] if not pre.empty else None

        levels = np.concatenate(([self._base_equity], window["equity"].to_numpy(dtype=float)))
        with np.errstate(divide="ignore", invalid="ignore"):
            returns = levels[1:] / levels[:-1] - 1.0
            peaks = np.maximum.accumulate(levels)
            drawdown = 1.0 - levels / peaks
        self._returns = returns

        self._equity_curve = window.assign(
            returns=returns,
            cumulative_return=levels[1:] / self._base_equity - 1.0,
        )
        self._drawdown = pd.DataFrame(
            {"equity": levels[1:], "peak_equity": peaks[1:], "drawdown": drawdown[1:]},
            index=window.index,
        )
        # Il massimo drawdown include il livello di partenza.
        self._max_drawdown = float(np.max(drawdown))

    # ------------------------------------------------------------------ #
    # Trade log
    # ------------------------------------------------------------------ #
    def _build_trades(self) -> tuple[pd.DataFrame, int]:
        """
        Ricostruisce i trade chiusi dai fill con lo stesso costo medio del RiskManager.

        Returns:
            (trade log, numero di posizioni ancora aperte a fine backtest).
        """
        positions: dict[str, Position] = {}
        open_trades: dict[str, _OpenTrade] = {}
        rows: list[dict[str, Any]] = []

        for fill in self._fills:
            symbol = fill.symbol
            position = positions.setdefault(symbol, Position(symbol, fill.strategy_id))
            previous = position.quantity
            realized = position.apply_fill(
                fill.signed_quantity, fill.fill_price, fill.commission, fill.timestamp
            )
            current = position.quantity

            flat_before = abs(previous) < _EPS
            if flat_before:
                open_trades[symbol] = _OpenTrade(
                    symbol=symbol,
                    strategy_id=fill.strategy_id,
                    direction="LONG" if fill.signed_quantity > 0 else "SHORT",
                    entry_time=fill.timestamp,
                )
            trade = open_trades[symbol]

            if flat_before or previous * fill.signed_quantity > 0:
                closing_qty, opening_qty = 0.0, fill.quantity
            else:
                closing_qty = min(fill.quantity, abs(previous))
                opening_qty = fill.quantity - closing_qty
            flipped = (not flat_before) and previous * current < 0

            if closing_qty > 0:
                trade.exit_qty += closing_qty
                trade.exit_value += closing_qty * fill.fill_price
            if opening_qty > 0 and not flipped:
                trade.entry_qty += opening_qty
                trade.entry_value += opening_qty * fill.fill_price
            trade.pnl += realized
            trade.commissions += fill.commission
            trade.slippage_cost += fill.slippage_cost

            if abs(current) < _EPS or flipped:
                rows.append(trade.to_row(fill.timestamp, trade_id=len(rows) + 1))
                del open_trades[symbol]
                if flipped:  # la quantita' residua apre un nuovo trade
                    open_trades[symbol] = _OpenTrade(
                        symbol=symbol,
                        strategy_id=fill.strategy_id,
                        direction="LONG" if current > 0 else "SHORT",
                        entry_time=fill.timestamp,
                        entry_qty=opening_qty,
                        entry_value=opening_qty * fill.fill_price,
                    )
                else:
                    positions.pop(symbol, None)

        trades = pd.DataFrame(rows, columns=_TRADE_COLUMNS)
        return trades, len(open_trades)

    # ------------------------------------------------------------------ #
    # Metriche
    # ------------------------------------------------------------------ #
    def _build_summary(self) -> dict[str, Any]:
        """Calcola il dizionario riepilogativo delle metriche."""
        returns = self._returns
        ppy = self._periods_per_year
        rf_per_period = self._risk_free_rate / ppy
        n_bars = len(returns)

        base = self._base_equity
        final = float(self._equity_curve["equity"].iloc[-1])
        total_return = final / base - 1.0

        first_ts = self._equity_curve.index[0]
        last_ts = self._equity_curve.index[-1]
        span_start = self._base_ts if self._base_ts is not None else first_ts
        years = (last_ts - span_start).total_seconds() / _SECONDS_PER_YEAR
        cagr = (final / base) ** (1.0 / years) - 1.0 if years > 0 and final > 0 else None

        mean = float(np.mean(returns))
        std = float(np.std(returns, ddof=1)) if n_bars > 1 else math.nan
        volatility = std * math.sqrt(ppy) if math.isfinite(std) else None
        sharpe = (mean - rf_per_period) / std * math.sqrt(ppy) if std > 0 else None

        downside = np.minimum(returns - rf_per_period, 0.0)
        downside_dev = float(np.sqrt(np.mean(downside**2)))
        sortino = (mean - rf_per_period) / downside_dev * math.sqrt(ppy) if downside_dev > 0 else None

        max_dd = self._max_drawdown
        calmar = cagr / max_dd if cagr is not None and max_dd > 0 else None

        trade_stats = self._trade_statistics()
        total_commissions = float(sum(fill.commission for fill in self._fills))
        total_slippage = float(sum(fill.slippage_cost for fill in self._fills))

        summary: dict[str, Any] = {
            "start": first_ts.date().isoformat(),
            "end": last_ts.date().isoformat(),
            "bars": n_bars,
            "initial_equity": base,
            "final_equity": final,
            "total_return": total_return,
            "cagr": cagr,
            "annualized_volatility": volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            **trade_stats,
            "total_commissions": total_commissions,
            "total_slippage": total_slippage,
            "total_costs": total_commissions + total_slippage,
            "n_open_positions": self._n_open_positions,
            "periods_per_year": ppy,
            "risk_free_rate": self._risk_free_rate,
        }
        return {key: _clean(v) if isinstance(v, float) or v is None else v for key, v in summary.items()}

    def _trade_statistics(self) -> dict[str, float | int | None]:
        """Win rate, profit factor e medie dei trade chiusi."""
        pnl = self._trades["pnl"].to_numpy(dtype=float)
        n_trades = len(pnl)
        if n_trades == 0:
            return {
                "n_trades": 0,
                "win_rate": None,
                "profit_factor": None,
                "avg_trade_pnl": None,
                "avg_win": None,
                "avg_loss": None,
            }

        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        gross_loss = float(-losses.sum())
        return {
            "n_trades": n_trades,
            "win_rate": len(wins) / n_trades,
            "profit_factor": float(wins.sum()) / gross_loss if gross_loss > 0 else None,
            "avg_trade_pnl": float(pnl.mean()),
            "avg_win": float(wins.mean()) if len(wins) else None,
            "avg_loss": float(losses.mean()) if len(losses) else None,
        }