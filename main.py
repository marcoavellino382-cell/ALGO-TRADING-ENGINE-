"""
main.py

Punto di ingresso da terminale per il backtest dell'Algo Trading Engine.

Uso (dalla root del progetto):
    python main.py                       # backtest con config/settings.yaml
    python main.py --offline             # solo dati gia' presenti in SQLite
    python main.py --log-level DEBUG     # log piu' dettagliato
    python main.py --save-results        # salva metriche e serie in SQLite
    python main.py --config altro.yaml   # configurazione alternativa

Flusso:
    1. carica settings.yaml (i percorsi relativi del database sono risolti
       rispetto alla root del progetto, quindi il comando funziona da qualsiasi
       cartella);
    2. costruisce e collega i componenti
           HistoricalDataHandler -> MeanReversionStrategy -> RiskManager
           -> SimulatedExecutionHandler -> TradingEngine;
    3. esegue la simulazione e calcola le metriche con PerformanceAnalyzer;
    4. stampa un report nel terminale.

`run_backtest()` e `format_report()` sono funzioni pure di modulo, riutilizzabili
dalla dashboard Streamlit senza passare dalla riga di comando.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import yaml

from src.analytics.performance import PerformanceAnalyzer
from src.core.engine import EngineConfig, EngineResult, TradingEngine
from src.data_handler.historical import DataUnavailableError, HistoricalDataHandler
from src.execution.simulated import SimulatedExecutionHandler
from src.risk.risk_manager import RiskManager
from src.strategy.mean_reversion import MeanReversionStrategy

__all__ = ["BacktestOutcome", "load_settings", "run_backtest", "format_report", "main"]

logger = logging.getLogger("main")

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
DEFAULT_CONFIG: Final[Path] = PROJECT_ROOT / "config" / "settings.yaml"

_REPORT_WIDTH: Final[int] = 66
_NOT_AVAILABLE: Final[str] = "n/d"


# --------------------------------------------------------------------------- #
# Risultato dell'esecuzione
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class BacktestOutcome:
    """
    Tutto cio' che serve per leggere e mostrare un backtest concluso.

    Attributes:
        settings: Configurazione usata (percorsi gia' risolti).
        data_handler: DataHandler storico utilizzato.
        strategy: Strategia utilizzata (parametri in `strategy.params`).
        risk_manager: RiskManager a fine simulazione (stato finale, `rejections`).
        execution_handler: ExecutionHandler a fine simulazione (`discarded`).
        engine_result: Risultato grezzo dell'engine (snapshots, fills, orders, signals).
        analyzer: PerformanceAnalyzer con metriche, serie e trade log.
    """

    settings: Mapping[str, Any]
    data_handler: HistoricalDataHandler
    strategy: MeanReversionStrategy
    risk_manager: RiskManager
    execution_handler: SimulatedExecutionHandler
    engine_result: EngineResult
    analyzer: PerformanceAnalyzer


# --------------------------------------------------------------------------- #
# Configurazione
# --------------------------------------------------------------------------- #
def load_settings(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """
    Carica settings.yaml e risolve i percorsi relativi rispetto alla root del progetto.

    Args:
        path: Percorso del file YAML.

    Raises:
        FileNotFoundError: se il file non esiste.
        ValueError: se il contenuto non e' un dizionario YAML.
    """
    if not path.is_file():
        raise FileNotFoundError(f"File di configurazione non trovato: {path}")
    with path.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    if not isinstance(settings, dict):
        raise ValueError(f"{path} e' vuoto o non e' un dizionario YAML valido.")

    database = settings["database"]
    for key in ("path", "raw_data_dir"):
        if key in database:
            resolved = Path(database[key])
            if not resolved.is_absolute():
                resolved = PROJECT_ROOT / resolved
            database[key] = str(resolved)
    return settings


def _configure_logging(level_name: str) -> None:
    """Imposta il logging del processo."""
    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Livello di log '{level_name}' non valido.")
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _seed_everything(seed: int) -> None:
    """Fissa i generatori casuali per la riproducibilita' dei risultati."""
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- #
# Esecuzione del backtest
# --------------------------------------------------------------------------- #
def run_backtest(settings: Mapping[str, Any], *, auto_download: bool = True) -> BacktestOutcome:
    """
    Costruisce i componenti, esegue il backtest e calcola le metriche.

    Args:
        settings: Contenuto di settings.yaml (da `load_settings`).
        auto_download: Se False non usa la rete (solo dati gia' in SQLite).

    Raises:
        DataUnavailableError: dati mancanti per uno o piu' simboli.
        ValueError: configurazione incoerente (es. warmup insufficiente).
    """
    data_handler = HistoricalDataHandler.from_settings(settings, auto_download=auto_download)
    strategy = MeanReversionStrategy.from_settings(settings, data_handler)
    risk_manager = RiskManager.from_settings(settings, data_handler)
    execution_handler = SimulatedExecutionHandler.from_settings(settings)

    engine_config = EngineConfig.from_settings(settings)
    strategy.ensure_warmup(engine_config.warmup_bars)

    engine = TradingEngine(engine_config, data_handler, strategy, risk_manager, execution_handler)
    engine_result = engine.run()

    analyzer = PerformanceAnalyzer.from_settings(settings, engine_result)
    return BacktestOutcome(
        settings=settings,
        data_handler=data_handler,
        strategy=strategy,
        risk_manager=risk_manager,
        execution_handler=execution_handler,
        engine_result=engine_result,
        analyzer=analyzer,
    )


# --------------------------------------------------------------------------- #
# Report
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


def _counts(counts: Mapping[str, int]) -> str:
    """Contatori 'motivo=n' in ordine alfabetico, oppure 'nessuno'."""
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "nessuno"


def _section(title: str, rows: Sequence[tuple[str, str]]) -> list[str]:
    """Sezione del report: titolo, filetto e righe etichetta/valore."""
    lines = ["", f" {title}", " " + "-" * (_REPORT_WIDTH - 2)]
    lines.extend(f" {label:<30}{value}" for label, value in rows)
    return lines


def format_report(outcome: BacktestOutcome) -> str:
    """
    Compone il report testuale del backtest.

    Args:
        outcome: Risultato di `run_backtest`.

    Returns:
        Stringa multilinea pronta per la stampa nel terminale.
    """
    settings = outcome.settings
    summary = outcome.analyzer.summary()
    result = outcome.engine_result
    currency = str(settings["general"]["base_currency"])
    params = outcome.strategy.params

    trend = f"SMA{params.trend_filter_window}" if params.use_trend_filter else "disattivato"
    halted = (
        f"SI' ({result.halt_timestamp:%Y-%m-%d})" if result.halted and result.halt_timestamp else "no"
    )

    lines = ["=" * _REPORT_WIDTH, " ALGO TRADING ENGINE - REPORT DI BACKTEST", "=" * _REPORT_WIDTH]
    lines += _section(
        "CONFIGURAZIONE",
        [
            ("Strategia", outcome.strategy.strategy_id),
            (
                "Parametri",
                f"lookback={params.lookback_window}, Z entrata={params.zscore_entry}, "
                f"uscita={params.zscore_exit}, stop={params.zscore_stop}",
            ),
            ("Filtro di trend", f"{trend}; max {params.max_holding_bars} barre in posizione"),
            (
                "Universo",
                f"{', '.join(settings['universe']['symbols'])} ({settings['universe']['timeframe']})",
            ),
            ("Periodo analizzato", f"{summary['start']} -> {summary['end']} ({summary['bars']:,} barre)"),
            ("Capitale iniziale", _money(summary["initial_equity"], currency)),
        ],
    )
    lines += _section(
        "RENDIMENTO",
        [
            ("Capitale finale", _money(summary["final_equity"], currency)),
            ("Rendimento totale", _pct(summary["total_return"])),
            ("CAGR", _pct(summary["cagr"])),
            ("Volatilita' annualizzata", _pct(summary["annualized_volatility"])),
        ],
    )
    lines += _section(
        "RISCHIO E RENDIMENTO AGGIUSTATO",
        [
            ("Sharpe Ratio", _num(summary["sharpe_ratio"])),
            ("Sortino Ratio", _num(summary["sortino_ratio"])),
            ("Max Drawdown", _pct(summary["max_drawdown"])),
            ("Calmar Ratio", _num(summary["calmar_ratio"])),
        ],
    )
    lines += _section(
        "TRADE",
        [
            ("Trade chiusi", f"{summary['n_trades']:,}"),
            ("Win rate", _pct(summary["win_rate"])),
            ("Profit factor", _num(summary["profit_factor"])),
            ("PnL medio per trade", _money(summary["avg_trade_pnl"], currency)),
            ("Vincita media", _money(summary["avg_win"], currency)),
            ("Perdita media", _money(summary["avg_loss"], currency)),
            ("Posizioni aperte a fine test", f"{summary['n_open_positions']}"),
        ],
    )
    initial = summary["initial_equity"]
    costs_pct = summary["total_costs"] / initial if initial else None
    lines += _section(
        "COSTI DI TRANSAZIONE",
        [
            ("Commissioni totali", _money(summary["total_commissions"], currency)),
            ("Slippage totale", _money(summary["total_slippage"], currency)),
            ("Costi totali", _money(summary["total_costs"], currency)),
            ("Incidenza sul capitale iniziale", _pct(costs_pct)),
        ],
    )
    lines += _section(
        "ENGINE",
        [
            ("Segnali / ordini / fill", f"{len(result.signals):,} / {len(result.orders):,} / {len(result.fills):,}"),
            ("Kill-switch attivato", halted),
            ("Segnali rifiutati (rischio)", _counts(outcome.risk_manager.rejections)),
            ("Ordini scartati (esecuzione)", _counts(outcome.execution_handler.discarded)),
        ],
    )
    lines += ["", "=" * _REPORT_WIDTH]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Riga di comando
# --------------------------------------------------------------------------- #
def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Definisce e legge gli argomenti da riga di comando."""
    parser = argparse.ArgumentParser(
        description="Esegue un backtest dell'Algo Trading Engine e stampa il report."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="percorso di settings.yaml (default: config/settings.yaml)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="non scaricare dati: usa solo quelli gia' presenti nel database SQLite",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="livello di log (DEBUG, INFO, WARNING, ERROR); default: general.log_level",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="salva metriche, equity curve, drawdown e trade log nel database SQLite",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Entry point da terminale.

    Returns:
        0 se il backtest e' riuscito, 1 in caso di errore di configurazione o di dati.
    """
    args = _parse_args(argv)
    try:
        settings = load_settings(args.config)
        general = settings["general"]
        _configure_logging(args.log_level or str(general.get("log_level", "INFO")))
        if "random_seed" in general:
            _seed_everything(int(general["random_seed"]))

        run_mode = str(general.get("run_mode", "backtest"))
        if run_mode != "backtest":
            raise ValueError(
                f"general.run_mode = '{run_mode}': main.py esegue solo backtest "
                "(imposta run_mode: backtest)."
            )

        outcome = run_backtest(settings, auto_download=not args.offline)
    except FileNotFoundError as exc:
        print(f"ERRORE: {exc}", file=sys.stderr)
        return 1
    except yaml.YAMLError as exc:
        print(f"ERRORE: settings.yaml non e' un YAML valido: {exc}", file=sys.stderr)
        return 1
    except KeyError as exc:
        print(f"ERRORE: chiave mancante in settings.yaml: {exc}", file=sys.stderr)
        return 1
    except (ValueError, DataUnavailableError) as exc:
        print(f"ERRORE: {exc}", file=sys.stderr)
        return 1

    print(format_report(outcome))

    if args.save_results:
        run_id = f"{outcome.strategy.strategy_id}-{datetime.now():%Y%m%d-%H%M%S}"
        db_path = settings["database"]["path"]
        outcome.analyzer.to_sqlite(db_path, run_id)
        print(f"\nRisultati salvati in {db_path} (run_id = {run_id}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())