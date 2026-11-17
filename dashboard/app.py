"""
dashboard/app.py

Dashboard interattiva (Streamlit + Plotly) dell'Algo Trading Engine.

Avvio dalla root del progetto:
    streamlit run dashboard/app.py

Funzionamento:
    - La barra laterale modifica i parametri chiave (capitale iniziale, ticker,
      soglie Z-score, rischio massimo per trade); il backtest parte solo al clic
      su "Avvia backtest".
    - Il backtest usa lo stesso codice del terminale (`main.run_backtest`):
      HistoricalDataHandler -> MeanReversionStrategy -> RiskManager ->
      SimulatedExecutionHandler -> TradingEngine -> PerformanceAnalyzer.
    - Modalita' locale: `auto_download=True` (i dati mancanti vengono scaricati
      con yfinance). Modalita' online (Streamlit Cloud): nessun accesso alla rete,
      si leggono solo i dati gia' presenti in `data/market_data.db`.
      La modalita' si deduce dal percorso `/mount/src` di Streamlit Cloud ed e'
      forzabile con la variabile d'ambiente ALGO_DASHBOARD_MODE=cloud|local.
    - I risultati sono in cache (`st.cache_data`) per combinazione di parametri:
      cambiare i filtri del trade log non rilancia il backtest.
    - Il benchmark e' il buy & hold di uno dei ticker testati, normalizzato allo
      stesso capitale di partenza della strategia.

Allineamento con config/settings.yaml:
    La dashboard parte dai valori di settings.yaml e ne sovrascrive solo:
    general.initial_capital, universe.symbols,
    strategy.mean_reversion.zscore_entry / zscore_exit / zscore_stop,
    risk.max_risk_per_trade.
"""

from __future__ import annotations

import inspect
import os
import re
import sqlite3
import sys
from collections.abc import Callable
from contextlib import closing
from datetime import date
from pathlib import Path
from typing import Any, Final

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import DEFAULT_CONFIG, BacktestOutcome, load_settings, run_backtest  # noqa: E402
from src.data_handler.historical import DataUnavailableError  # noqa: E402

_ALL_BARS: Final[int] = 10_000_000
_DEFAULT_BENCHMARK: Final[str] = "SPY"
_TICKER_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Z0-9.\-]{1,10}$")
_NOT_AVAILABLE: Final[str] = "n/d"


# --------------------------------------------------------------------------- #
# Utilita' di formattazione e compatibilita'
# --------------------------------------------------------------------------- #
def _pct(value: float | None, digits: int = 2) -> str:
    """Frazione -> percentuale testuale (None -> n/d)."""
    return _NOT_AVAILABLE if value is None else f"{value * 100:,.{digits}f}%"


def _num(value: float | None, digits: int = 2) -> str:
    """Numero con separatore delle migliaia (None -> n/d)."""
    return _NOT_AVAILABLE if value is None else f"{value:,.{digits}f}"


def _money(value: float | None, currency: str) -> str:
    """Importo in valuta base (None -> n/d)."""
    return _NOT_AVAILABLE if value is None else f"{value:,.2f} {currency}"


def _counts(counts: dict[str, int]) -> str:
    """Contatori 'motivo=n' in ordine alfabetico, oppure 'nessuno'."""
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "nessuno"


def _stretch(widget: Callable[..., Any]) -> dict[str, Any]:
    """
    Argomento di larghezza piena compatibile con la versione di Streamlit:
    `use_container_width=True` (versioni correnti e precedenti) oppure
    `width="stretch"` (versioni in cui il primo e' stato rimosso).
    """
    parameters = inspect.signature(widget).parameters
    if "use_container_width" in parameters:
        return {"use_container_width": True}
    if "width" in parameters:
        return {"width": "stretch"}
    return {}


def _is_cloud() -> bool:
    """True se la dashboard gira su Streamlit Cloud (nessuna rete, solo dati in cache)."""
    mode = os.getenv("ALGO_DASHBOARD_MODE", "").strip().lower()
    if mode == "cloud":
        return True
    if mode == "local":
        return False
    return str(PROJECT_ROOT).startswith("/mount/src")


# --------------------------------------------------------------------------- #
# Dati: simboli in cache, esecuzione e estrazione dei risultati
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=300, show_spinner=False)
def _available_symbols(db_path: str, timeframe: str, adjustment: str) -> list[str]:
    """Ticker con barre nel database SQLite (lista vuota se il database manca)."""
    path = Path(db_path)
    if not path.is_file():
        return []
    try:
        with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as conn:
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM ohlcv_bars WHERE timeframe = ? AND adjustment = ? "
                "ORDER BY symbol",
                (timeframe, adjustment),
            ).fetchall()
    except sqlite3.Error:
        return []
    return [str(row[0]) for row in rows]


def _drawdown_from_equity(series: pd.Series, base_equity: float) -> pd.Series:
    """Drawdown (frazione >= 0) di una serie di equity, con picco iniziale = base_equity."""
    peak = series.cummax().clip(lower=base_equity)
    return 1.0 - series / peak


def _benchmark_equities(outcome: BacktestOutcome) -> pd.DataFrame:
    """
    Buy & hold di ogni ticker dell'universo, in unita' di equity.

    Ogni colonna parte dallo stesso livello della strategia (`initial_equity`) e
    usa il close dell'ultima barra precedente alla finestra come base, cosi' il
    primo rendimento e' confrontabile con quello della strategia.
    """
    equity = outcome.analyzer.equity_curve()
    base_equity = float(outcome.analyzer.summary()["initial_equity"])
    first_ts = equity.index[0]

    columns: dict[str, pd.Series] = {}
    for symbol in outcome.data_handler.symbols:
        closes = outcome.data_handler.get_latest_bars(symbol, _ALL_BARS)["close"]
        if closes.empty:
            continue
        prior = closes[closes.index < first_ts]
        base_close = float(prior.iloc[-1]) if len(prior) else float(closes.iloc[0])
        aligned = closes.reindex(equity.index, method="ffill")
        columns[symbol] = base_equity * aligned / base_close
    return pd.DataFrame(columns, index=equity.index)


def _extract(outcome: BacktestOutcome) -> dict[str, Any]:
    """Estrae da un backtest solo dati semplici (dict, DataFrame) adatti alla cache."""
    result = outcome.engine_result
    return {
        "summary": outcome.analyzer.summary(),
        "equity": outcome.analyzer.equity_curve(),
        "drawdown": outcome.analyzer.drawdown_series(),
        "trades": outcome.analyzer.trade_log(),
        "benchmarks": _benchmark_equities(outcome),
        "currency": str(outcome.settings["general"]["base_currency"]),
        "n_signals": len(result.signals),
        "n_orders": len(result.orders),
        "n_fills": len(result.fills),
        "halted": result.halted,
        "halt_timestamp": result.halt_timestamp,
        "rejections": outcome.risk_manager.rejections,
        "discarded": outcome.execution_handler.discarded,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def _run_cached(
    initial_capital: float,
    symbols: tuple[str, ...],
    zscore_entry: float,
    zscore_exit: float,
    zscore_stop: float,
    max_risk_per_trade: float,
    auto_download: bool,
    config_path: str,
) -> dict[str, Any]:
    """
    Esegue il backtest con i parametri della barra laterale (risultato in cache).

    Raises:
        ValueError, KeyError, FileNotFoundError, DataUnavailableError: vedi `main.run_backtest`.
    """
    settings = load_settings(Path(config_path))
    settings["general"]["initial_capital"] = initial_capital
    settings["universe"]["symbols"] = list(symbols)
    settings["strategy"]["mean_reversion"].update(
        zscore_entry=zscore_entry, zscore_exit=zscore_exit, zscore_stop=zscore_stop
    )
    settings["risk"]["max_risk_per_trade"] = max_risk_per_trade
    return _extract(run_backtest(settings, auto_download=auto_download))


# --------------------------------------------------------------------------- #
# Barra laterale
# --------------------------------------------------------------------------- #
def _parse_extra_tickers(text: str) -> tuple[list[str], list[str]]:
    """Separa i ticker validi da quelli scartati in una stringa separata da virgole."""
    valid: list[str] = []
    invalid: list[str] = []
    for token in (part.strip().upper() for part in text.split(",")):
        if not token:
            continue
        (valid if _TICKER_PATTERN.match(token) else invalid).append(token)
    return valid, invalid


def _render_sidebar(settings: dict[str, Any], cloud: bool) -> tuple[dict[str, Any], bool]:
    """
    Disegna la barra laterale.

    Returns:
        (parametri correnti, True se e' stato premuto "Avvia backtest").
    """
    side = st.sidebar
    side.header("Parametri del backtest")

    universe = settings["universe"]
    mean_reversion = settings["strategy"]["mean_reversion"]

    if cloud:
        download = False
        side.caption("Modalita' online: i dati vengono letti da `data/market_data.db`.")
    else:
        download = side.checkbox(
            "Scarica i dati mancanti (yfinance)",
            value=True,
            help="Disattivalo per usare solo i dati gia' presenti in SQLite.",
        )

    capital = side.number_input(
        f"Capitale iniziale ({settings['general']['base_currency']})",
        min_value=1_000.0,
        value=float(settings["general"]["initial_capital"]),
        step=10_000.0,
    )

    cached = _available_symbols(
        str(settings["database"]["path"]),
        str(universe["timeframe"]),
        str(universe.get("price_adjustment", "adjusted")),
    )
    defaults = [str(symbol) for symbol in universe["symbols"]]
    options = sorted(set(cached) | set(defaults)) if download else cached
    if not options:
        st.error(
            "Nessun dato in `data/market_data.db`: esegui prima `python main.py` in locale "
            "e includi il database nel repository."
        )
        st.stop()

    selected = side.multiselect(
        "Ticker da testare", options, default=[s for s in defaults if s in options]
    )
    extra: list[str] = []
    if download:
        extra_text = side.text_input("Ticker aggiuntivi (separati da virgola)", value="")
        extra, rejected = _parse_extra_tickers(extra_text)
        if rejected:
            side.warning(f"Ticker non validi ignorati: {', '.join(rejected)}")
    symbols = tuple(dict.fromkeys([*selected, *extra]))  # unici, ordine preservato

    side.subheader("Strategia Mean Reversion")
    zscore_entry = round(
        side.slider("Z-score di ingresso", -4.0, -0.5, float(mean_reversion["zscore_entry"]), 0.1), 2
    )
    zscore_exit = round(
        side.slider("Z-score di uscita (target)", -1.0, 2.0, float(mean_reversion["zscore_exit"]), 0.1), 2
    )
    zscore_stop = round(
        side.slider("Z-score di stop", -6.0, -1.0, float(mean_reversion["zscore_stop"]), 0.1), 2
    )

    side.subheader("Rischio")
    max_risk = round(
        side.slider(
            "Rischio max per trade (% del capitale)",
            0.1,
            5.0,
            float(settings["risk"]["max_risk_per_trade"]) * 100,
            0.1,
        )
        / 100,
        4,
    )

    valid = True
    if not symbols:
        side.warning("Seleziona almeno un ticker.")
        valid = False
    if not zscore_stop < zscore_entry < zscore_exit:
        side.warning("Richiesto: Z stop < Z ingresso < Z uscita.")
        valid = False

    params: dict[str, Any] = {
        "initial_capital": float(capital),
        "symbols": symbols,
        "zscore_entry": zscore_entry,
        "zscore_exit": zscore_exit,
        "zscore_stop": zscore_stop,
        "max_risk_per_trade": max_risk,
        "auto_download": download,
        "config_path": str(DEFAULT_CONFIG),
    }
    clicked = side.button(
        "Avvia backtest", type="primary", disabled=not valid, **_stretch(side.button)
    )
    return params, clicked


# --------------------------------------------------------------------------- #
# Grafici
# --------------------------------------------------------------------------- #
def _equity_figure(
    equity: pd.Series, benchmark: pd.Series | None, benchmark_name: str | None, currency: str
) -> go.Figure:
    """Equity curve della strategia contro il benchmark buy & hold."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.to_numpy(),
            name="Strategia",
            mode="lines",
            line={"width": 2.5},
            hovertemplate="%{y:,.2f}",
        )
    )
    if benchmark is not None:
        fig.add_trace(
            go.Scatter(
                x=benchmark.index,
                y=benchmark.to_numpy(),
                name=f"Buy & hold {benchmark_name}",
                mode="lines",
                line={"width": 1.5, "dash": "dot"},
                hovertemplate="%{y:,.2f}",
            )
        )
    fig.update_layout(
        hovermode="x unified",
        yaxis_title=f"Equity ({currency})",
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    return fig


def _drawdown_figure(
    drawdown: pd.Series, bench_drawdown: pd.Series | None, benchmark_name: str | None
) -> go.Figure:
    """Drawdown della strategia (area) e del benchmark (linea), in percentuale negativa."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=-drawdown.to_numpy() * 100,
            name="Strategia",
            mode="lines",
            fill="tozeroy",
            line={"width": 1.5},
            hovertemplate="%{y:.2f}%",
        )
    )
    if bench_drawdown is not None:
        fig.add_trace(
            go.Scatter(
                x=bench_drawdown.index,
                y=-bench_drawdown.to_numpy() * 100,
                name=f"Buy & hold {benchmark_name}",
                mode="lines",
                line={"width": 1.2, "dash": "dot"},
                hovertemplate="%{y:.2f}%",
            )
        )
    fig.update_layout(
        hovermode="x unified",
        yaxis_title="Drawdown",
        yaxis_ticksuffix="%",
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    return fig


# --------------------------------------------------------------------------- #
# Sezioni dei risultati
# --------------------------------------------------------------------------- #
def _render_kpis(
    summary: dict[str, Any],
    currency: str,
    benchmark_name: str | None,
    bench_return: float | None,
    bench_drawdown_max: float | None,
) -> None:
    """KPI principali (riga 1) e secondari (riga 2)."""
    total_return = summary["total_return"]
    max_dd = summary["max_drawdown"]

    return_delta = (
        f"{(total_return - bench_return) * 100:+.2f} pp vs {benchmark_name}"
        if bench_return is not None
        else None
    )
    dd_delta = (
        f"{(bench_drawdown_max - max_dd) * 100:+.2f} pp vs {benchmark_name}"
        if bench_drawdown_max is not None
        else None
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return", _pct(total_return), delta=return_delta)
    c2.metric(
        "Sharpe Ratio",
        _num(summary["sharpe_ratio"]),
        help="Rendimento medio in eccesso / volatilita', annualizzato.",
    )
    c3.metric(
        "Max Drawdown",
        _pct(-max_dd + 0.0),
        delta=dd_delta,
        help="Peggior calo dal massimo precedente. Il delta e' positivo se la strategia "
        "ha un drawdown minore del benchmark.",
    )
    c4.metric(
        "Win Rate",
        _pct(summary["win_rate"]),
        delta=f"{summary['n_trades']} trade chiusi",
        delta_color="off",
    )

    d1, d2, d3, d4, d5, d6 = st.columns(6)
    d1.metric("CAGR", _pct(summary["cagr"]))
    d2.metric("Volatilita' ann.", _pct(summary["annualized_volatility"]))
    d3.metric("Sortino", _num(summary["sortino_ratio"]))
    d4.metric("Calmar", _num(summary["calmar_ratio"]))
    d5.metric("Profit Factor", _num(summary["profit_factor"]))
    d6.metric(
        "Costi totali",
        _money(summary["total_costs"], currency),
        delta=f"{_pct(summary['total_costs'] / summary['initial_equity'])} del capitale",
        delta_color="off",
        help="Commissioni + slippage.",
    )


def _render_trade_log(trades: pd.DataFrame, currency: str) -> None:
    """Tabella dei trade con filtri per ticker, direzione, esito e periodo."""
    if trades.empty:
        st.info("Nessun trade chiuso con questi parametri.")
        return

    c1, c2, c3 = st.columns(3)
    symbol_options = sorted(trades["symbol"].unique())
    chosen_symbols = c1.multiselect("Ticker", symbol_options, default=symbol_options)
    direction_options = sorted(trades["direction"].unique())
    chosen_directions = c2.multiselect("Direzione", direction_options, default=direction_options)
    outcome = c3.selectbox("Esito", ["Tutti", "Vincenti", "Perdenti"])

    exit_dates = pd.to_datetime(trades["exit_time"]).dt.date
    min_date, max_date = exit_dates.min(), exit_dates.max()
    period = st.date_input(
        "Periodo di uscita", value=(min_date, max_date), min_value=min_date, max_value=max_date
    )
    start_date, end_date = (
        (period[0], period[1]) if isinstance(period, tuple) and len(period) == 2 else (min_date, max_date)
    )

    mask = (
        trades["symbol"].isin(chosen_symbols)
        & trades["direction"].isin(chosen_directions)
        & (exit_dates >= start_date)
        & (exit_dates <= end_date)
    )
    if outcome == "Vincenti":
        mask &= trades["pnl"] > 0
    elif outcome == "Perdenti":
        mask &= trades["pnl"] < 0
    filtered = trades[mask]

    if filtered.empty:
        st.warning("Nessun trade corrisponde ai filtri selezionati.")
        return

    wins = int((filtered["pnl"] > 0).sum())
    st.caption(
        f"{len(filtered)} trade - PnL totale {_money(float(filtered['pnl'].sum()), currency)} - "
        f"win rate {_pct(wins / len(filtered))}"
    )

    view = filtered.assign(pnl_pct=filtered["pnl_pct"] * 100).drop(columns=["strategy_id"])
    st.dataframe(
        view,
        hide_index=True,
        column_config={
            "trade_id": st.column_config.NumberColumn("ID", format="%d"),
            "symbol": "Ticker",
            "direction": "Direzione",
            "entry_time": st.column_config.DatetimeColumn("Ingresso", format="YYYY-MM-DD"),
            "exit_time": st.column_config.DatetimeColumn("Uscita", format="YYYY-MM-DD"),
            "holding_days": st.column_config.NumberColumn("Giorni", format="%.0f"),
            "quantity": st.column_config.NumberColumn("Quantita'", format="%.0f"),
            "entry_price": st.column_config.NumberColumn("Prezzo ingresso", format="%.2f"),
            "exit_price": st.column_config.NumberColumn("Prezzo uscita", format="%.2f"),
            "pnl": st.column_config.NumberColumn(f"PnL ({currency})", format="%.2f"),
            "pnl_pct": st.column_config.NumberColumn("PnL %", format="%.2f"),
            "commissions": st.column_config.NumberColumn("Commissioni", format="%.2f"),
            "slippage_cost": st.column_config.NumberColumn("Slippage", format="%.2f"),
        },
        **_stretch(st.dataframe),
    )
    st.download_button(
        "Scarica CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="trade_log.csv",
        mime="text/csv",
    )


def _render_metrics_table(data: dict[str, Any]) -> None:
    """Tabella completa di metriche, costi e diagnostica dell'engine."""
    s = data["summary"]
    currency = data["currency"]
    halted = (
        f"SI' ({data['halt_timestamp']:%Y-%m-%d})"
        if data["halted"] and data["halt_timestamp"]
        else "no"
    )
    rows = [
        ("Periodo", f"{s['start']} -> {s['end']}"),
        ("Barre analizzate", f"{s['bars']:,}"),
        ("Capitale di partenza", _money(s["initial_equity"], currency)),
        ("Capitale finale", _money(s["final_equity"], currency)),
        ("Rendimento totale", _pct(s["total_return"])),
        ("CAGR", _pct(s["cagr"])),
        ("Volatilita' annualizzata", _pct(s["annualized_volatility"])),
        ("Sharpe Ratio", _num(s["sharpe_ratio"])),
        ("Sortino Ratio", _num(s["sortino_ratio"])),
        ("Max Drawdown", _pct(s["max_drawdown"])),
        ("Calmar Ratio", _num(s["calmar_ratio"])),
        ("Trade chiusi", f"{s['n_trades']:,}"),
        ("Win rate", _pct(s["win_rate"])),
        ("Profit factor", _num(s["profit_factor"])),
        ("PnL medio per trade", _money(s["avg_trade_pnl"], currency)),
        ("Vincita media", _money(s["avg_win"], currency)),
        ("Perdita media", _money(s["avg_loss"], currency)),
        ("Posizioni aperte a fine test", str(s["n_open_positions"])),
        ("Commissioni totali", _money(s["total_commissions"], currency)),
        ("Slippage totale", _money(s["total_slippage"], currency)),
        ("Costi totali", _money(s["total_costs"], currency)),
        ("Segnali / ordini / fill", f"{data['n_signals']:,} / {data['n_orders']:,} / {data['n_fills']:,}"),
        ("Kill-switch attivato", halted),
        ("Segnali rifiutati (rischio)", _counts(data["rejections"])),
        ("Ordini scartati (esecuzione)", _counts(data["discarded"])),
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=["Metrica", "Valore"]),
        hide_index=True,
        **_stretch(st.dataframe),
    )


def _render_results(data: dict[str, Any], active: dict[str, Any]) -> None:
    """Disegna KPI, grafici, trade log e metriche complete."""
    summary = data["summary"]
    currency = data["currency"]
    benchmarks: pd.DataFrame = data["benchmarks"]
    base_equity = float(summary["initial_equity"])

    st.caption(
        f"Ticker: {', '.join(active['symbols'])} - Z ingresso {active['zscore_entry']} / "
        f"uscita {active['zscore_exit']} / stop {active['zscore_stop']} - "
        f"rischio per trade {active['max_risk_per_trade']:.2%} - "
        f"capitale {active['initial_capital']:,.0f} {currency}"
    )
    if data["halted"] and data["halt_timestamp"]:
        st.warning(
            f"Kill-switch attivato il {data['halt_timestamp']:%Y-%m-%d}: "
            "il backtest si e' fermato dopo la liquidazione delle posizioni."
        )

    benchmark_name: str | None = None
    bench_series: pd.Series | None = None
    bench_drawdown: pd.Series | None = None
    if not benchmarks.empty:
        names = list(benchmarks.columns)
        default_index = names.index(_DEFAULT_BENCHMARK) if _DEFAULT_BENCHMARK in names else 0
        benchmark_name = st.selectbox("Benchmark (buy & hold)", names, index=default_index)
        bench_series = benchmarks[benchmark_name].dropna()
        if bench_series.empty:
            bench_series = None
        else:
            bench_drawdown = _drawdown_from_equity(bench_series, base_equity)

    bench_return = float(bench_series.iloc[-1]) / base_equity - 1.0 if bench_series is not None else None
    bench_dd_max = float(bench_drawdown.max()) if bench_drawdown is not None else None

    _render_kpis(summary, currency, benchmark_name, bench_return, bench_dd_max)

    tab_equity, tab_drawdown, tab_trades, tab_metrics = st.tabs(
        ["Equity Curve", "Drawdown", "Trade Log", "Metriche complete"]
    )
    with tab_equity:
        st.plotly_chart(
            _equity_figure(data["equity"]["equity"], bench_series, benchmark_name, currency),
            **_stretch(st.plotly_chart),
        )
    with tab_drawdown:
        st.plotly_chart(
            _drawdown_figure(data["drawdown"]["drawdown"], bench_drawdown, benchmark_name),
            **_stretch(st.plotly_chart),
        )
    with tab_trades:
        _render_trade_log(data["trades"], currency)
    with tab_metrics:
        _render_metrics_table(data)

    st.caption(
        "Backtest storico a scopo dimostrativo: i risultati passati non garantiscono "
        "risultati futuri e non costituiscono consulenza finanziaria."
    )


# --------------------------------------------------------------------------- #
# Applicazione
# --------------------------------------------------------------------------- #
def run_app() -> None:
    """Punto di ingresso della dashboard."""
    st.set_page_config(page_title="Algo Trading Engine", page_icon="📈", layout="wide")
    st.title("Algo Trading Engine")
    st.caption("Backtest event-driven di una strategia mean reversion, con costi e rischio realistici.")

    settings = load_settings(DEFAULT_CONFIG)
    cloud = _is_cloud()
    params, clicked = _render_sidebar(settings, cloud)

    if clicked:
        st.session_state["params"] = params
    active: dict[str, Any] | None = st.session_state.get("params")

    if active is None:
        st.info(
            "Imposta i parametri nella barra laterale e premi **Avvia backtest**. "
            "Con le impostazioni di partenza vengono usati i valori di `config/settings.yaml`."
        )
        return
    if params != active:
        st.sidebar.info("Parametri modificati: premi 'Avvia backtest' per aggiornare i risultati.")

    try:
        with st.spinner("Backtest in corso..."):
            data = _run_cached(**active)
    except (ValueError, KeyError, FileNotFoundError, DataUnavailableError) as exc:
        st.error(f"Impossibile eseguire il backtest: {exc}")
        return

    _render_results(data, active)


run_app()