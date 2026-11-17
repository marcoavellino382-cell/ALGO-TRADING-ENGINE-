"""
src/data_handler/historical.py

`HistoricalDataHandler`: sorgente dati per il backtest, con cache SQLite e
download automatico da yfinance.

Flusso:
    1. Calcola l'intervallo necessario: [start_date - storico di warmup, end_date].
    2. Per ogni simbolo controlla nella cache SQLite (`download_log`) se
       l'intervallo e' gia' coperto; se no (tabella vuota, simbolo nuovo o date
       piu' ampie) scarica da yfinance, pulisce i dati e li salva nel DB.
    3. Carica tutto da SQLite e rilascia i dati UN timestamp alla volta tramite
       `_next_batch()`. Lo storico accessibile con `get_latest_bars()` e'
       gestito da `BaseDataHandler` e contiene solo barre gia' emesse.

Warmup (allineamento con `backtest.warmup_bars` e con l'engine):
    L'engine consente segnali quando un simbolo ha visto `warmup_bars` barre.
    Per far partire i segnali esattamente alla prima barra >= `start_date`,
    l'handler emette `warmup_bars - 1` barre PRECEDENTI a `start_date` (barre
    di riscaldamento, nessun segnale possibile) e poi la barra di `start_date`
    e successive. Chi calcola le metriche deve quindi escludere gli snapshot
    con timestamp < `start_date` (vedi proprieta' `start_date`).

Note sui dati:
    - `price_adjustment = adjusted`: prezzi corretti per split e dividendi
      (yfinance `auto_adjust=True`). I prezzi aggiustati cambiano retroattivamente
      a ogni nuovo dividendo: i dati in cache sono quelli del giorno del
      download. Per rinfrescarli, elimina le righe del simbolo da `download_log`.
    - `price_adjustment = raw`: prezzi corretti solo per gli split (cosi' li
      fornisce Yahoo), senza dividendi.
    - Il download e' pensato per il timeframe `1d`: yfinance limita la storia
      intraday (circa 60 giorni per 1m/5m/15m, 730 per 1h).

Allineamento con config/settings.yaml:
    universe.symbols / timeframe / price_adjustment,
    backtest.start_date / end_date / warmup_bars, database.path.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import closing
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Final, NamedTuple

import pandas as pd

from src.core.events import MarketEvent
from src.data_handler.base import BAR_COLUMNS, BaseDataHandler

__all__ = ["DataUnavailableError", "HistoricalDataHandler"]

logger = logging.getLogger(__name__)

_TS_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
_PRICE_COLUMNS: Final[list[str]] = ["open", "high", "low", "close"]

# Timeframe del progetto -> intervallo yfinance.
_YF_INTERVALS: Final[Mapping[str, str]] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "1d": "1d",
}

# Giorni di calendario per barra giornaliera (7/5 + festivi) e margine fisso:
# garantiscono abbastanza storico precedente a start_date.
_CALENDAR_DAYS_PER_BAR: Final[float] = 1.5
_CALENDAR_BUFFER_DAYS: Final[int] = 30

_CREATE_BARS_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS ohlcv_bars (
    symbol      TEXT NOT NULL,
    timeframe   TEXT NOT NULL,
    adjustment  TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    open        REAL NOT NULL,
    high        REAL NOT NULL,
    low         REAL NOT NULL,
    close       REAL NOT NULL,
    volume      REAL NOT NULL,
    PRIMARY KEY (symbol, timeframe, adjustment, timestamp)
)
"""

_CREATE_LOG_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS download_log (
    symbol           TEXT NOT NULL,
    timeframe        TEXT NOT NULL,
    adjustment       TEXT NOT NULL,
    requested_start  TEXT NOT NULL,
    requested_end    TEXT NOT NULL,
    downloaded_at    TEXT NOT NULL,
    PRIMARY KEY (symbol, timeframe, adjustment)
)
"""


class DataUnavailableError(RuntimeError):
    """Dati mancanti: download fallito, simbolo sconosciuto o cache vuota."""


class _Coverage(NamedTuple):
    """Intervallo di date gia' scaricato (estremi inclusi) per un simbolo."""

    start: date
    end: date


def _to_date(value: str | date | datetime) -> date:
    """Converte stringhe ISO, date o datetime in `date` (YAML puo' dare entrambi)."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


class HistoricalDataHandler(BaseDataHandler):
    """
    DataHandler storico con cache SQLite e download automatico yfinance.

    I dati vengono preparati (cache + eventuale download) e caricati nel
    costruttore, cosi' ogni errore di dati emerge prima dell'avvio dell'engine.

    Args:
        symbols: Universo di ticker (`universe.symbols`).
        timeframe: Granularita' (`universe.timeframe`), es. "1d".
        start_date: Prima data del backtest (`backtest.start_date`).
        end_date: Ultima data del backtest, inclusa (`backtest.end_date`).
        db_path: Percorso del database SQLite (`database.path`), relativo alla
            cartella da cui si lancia `main.py`.
        warmup_bars: Barre di riscaldamento dell'engine (`backtest.warmup_bars`).
        price_adjustment: "adjusted" oppure "raw" (`universe.price_adjustment`).
        auto_download: Se False non usa mai la rete (utile su Streamlit Cloud e
            nei test): lavora solo con i dati gia' presenti nel DB.

    Raises:
        ValueError: parametri incoerenti.
        DataUnavailableError: dati mancanti per uno o piu' simboli.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        timeframe: str,
        start_date: str | date | datetime,
        end_date: str | date | datetime,
        db_path: str | Path,
        warmup_bars: int,
        price_adjustment: str = "adjusted",
        auto_download: bool = True,
    ) -> None:
        super().__init__(symbols, timeframe)

        if timeframe not in _YF_INTERVALS:
            raise ValueError(
                f"Timeframe '{timeframe}' non supportato: usa uno tra {sorted(_YF_INTERVALS)}."
            )
        if price_adjustment not in ("adjusted", "raw"):
            raise ValueError("'price_adjustment' deve essere 'adjusted' oppure 'raw'.")
        if warmup_bars < 0:
            raise ValueError(f"'warmup_bars' deve essere >= 0 (ricevuto {warmup_bars}).")

        self._start_date: date = _to_date(start_date)
        self._end_date: date = _to_date(end_date)
        if self._start_date > self._end_date:
            raise ValueError("'start_date' deve essere <= 'end_date'.")

        self._db_path = Path(db_path)
        self._adjustment: str = price_adjustment
        self._auto_download = auto_download

        # Barre di riscaldamento da emettere PRIMA di start_date (vedi docstring modulo).
        self._pre_start_bars: int = max(warmup_bars - 1, 0)
        buffer_days = (
            math.ceil(self._pre_start_bars * _CALENDAR_DAYS_PER_BAR) + _CALENDAR_BUFFER_DAYS
            if self._pre_start_bars > 0
            else 0
        )
        self._fetch_start: date = self._start_date - timedelta(days=buffer_days)

        # Mai oltre ieri: la barra odierna puo' essere incompleta.
        self._effective_end: date = min(self._end_date, date.today() - timedelta(days=1))
        if self._effective_end < self._start_date:
            raise ValueError("L'intervallo richiesto non contiene alcuna barra chiusa.")

        self._groups: list[tuple[pd.Timestamp, pd.DataFrame]] = []
        self._cursor: int = 0
        self._prepare_data()

    # ------------------------------------------------------------------ #
    # Costruzione da settings.yaml
    # ------------------------------------------------------------------ #
    @classmethod
    def from_settings(
        cls, settings: Mapping[str, Any], *, auto_download: bool = True
    ) -> HistoricalDataHandler:
        """
        Costruisce l'handler dal dizionario completo di settings.yaml.

        Args:
            settings: Contenuto di settings.yaml gia' caricato (`yaml.safe_load`).
            auto_download: Se False disabilita il download da yfinance.
        """
        universe = settings["universe"]
        backtest = settings["backtest"]
        database = settings["database"]
        return cls(
            symbols=list(universe["symbols"]),
            timeframe=str(universe["timeframe"]),
            start_date=backtest["start_date"],
            end_date=backtest["end_date"],
            db_path=database["path"],
            warmup_bars=int(backtest["warmup_bars"]),
            price_adjustment=str(universe.get("price_adjustment", "adjusted")),
            auto_download=auto_download,
        )

    # ------------------------------------------------------------------ #
    # Proprieta'
    # ------------------------------------------------------------------ #
    @property
    def start_date(self) -> date:
        """Prima data del backtest: le barre precedenti sono solo warmup."""
        return self._start_date

    @property
    def end_date(self) -> date:
        """Ultima data richiesta del backtest (inclusa)."""
        return self._end_date

    # ------------------------------------------------------------------ #
    # Interfaccia BaseDataHandler
    # ------------------------------------------------------------------ #
    def has_next(self) -> bool:
        """True se esistono altri timestamp da emettere."""
        return self._cursor < len(self._groups)

    def _next_batch(self) -> list[MarketEvent]:
        """Restituisce le barre di UN solo timestamp, nell'ordine dell'universo."""
        timestamp, rows = self._groups[self._cursor]
        self._cursor += 1
        bar_time = timestamp.to_pydatetime()
        return [
            MarketEvent(
                timestamp=bar_time,
                symbol=row.symbol,
                timeframe=self._timeframe,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
            )
            for row in rows.itertuples(index=False)
        ]

    # ------------------------------------------------------------------ #
    # Preparazione dati: cache -> download -> caricamento
    # ------------------------------------------------------------------ #
    def _prepare_data(self) -> None:
        """Garantisce la cache, carica i dati e li organizza per timestamp."""
        with closing(self._connect()) as conn:
            conn.execute(_CREATE_BARS_SQL)
            conn.execute(_CREATE_LOG_SQL)
            conn.commit()

        for symbol in self._symbols:
            self._ensure_symbol_cached(symbol)

        frame = self._load_frame()
        self._groups = [(ts, grp) for ts, grp in frame.groupby(level=0, sort=False)]
        logger.info(
            "HistoricalDataHandler pronto: %d simboli, %d timestamp (%s -> %s), "
            "%d barre di warmup prima di %s.",
            len(self._symbols),
            len(self._groups),
            self._groups[0][0].date(),
            self._groups[-1][0].date(),
            self._pre_start_bars,
            self._start_date,
        )

    def _connect(self) -> sqlite3.Connection:
        """Apre la connessione SQLite creando la cartella se necessario."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self._db_path)

    def _ensure_symbol_cached(self, symbol: str) -> None:
        """Scarica e salva i dati del simbolo solo se la cache non copre l'intervallo."""
        with closing(self._connect()) as conn:
            coverage = self._read_coverage(conn, symbol)

        if (
            coverage is not None
            and coverage.start <= self._fetch_start
            and coverage.end >= self._effective_end
        ):
            logger.debug("[%s] cache SQLite sufficiente (%s -> %s).", symbol, *coverage)
            return

        if not self._auto_download:
            logger.warning("[%s] cache incompleta e auto_download disattivato.", symbol)
            return

        # Si scarica l'unione dell'intervallo richiesto e di quello in cache,
        # cosi' `download_log` descrive sempre un intervallo senza buchi.
        dl_start = min(coverage.start, self._fetch_start) if coverage else self._fetch_start
        dl_end = max(coverage.end, self._effective_end) if coverage else self._effective_end

        logger.info("[%s] download yfinance %s -> %s", symbol, dl_start, dl_end)
        bars = self._download_symbol(symbol, dl_start, dl_end)
        self._store_bars(symbol, bars, dl_start, dl_end)

    def _read_coverage(self, conn: sqlite3.Connection, symbol: str) -> _Coverage | None:
        """Legge da `download_log` l'intervallo gia' scaricato per il simbolo."""
        row = conn.execute(
            "SELECT requested_start, requested_end FROM download_log "
            "WHERE symbol = ? AND timeframe = ? AND adjustment = ?",
            (symbol, self._timeframe, self._adjustment),
        ).fetchone()
        if row is None:
            return None
        return _Coverage(date.fromisoformat(row[0]), date.fromisoformat(row[1]))

    def _download_symbol(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """
        Scarica le barre da yfinance e le restituisce pulite.

        Returns:
            DataFrame indicizzato per timestamp con colonne `BAR_COLUMNS`.

        Raises:
            DataUnavailableError: yfinance non installato, errore di rete o
                nessun dato restituito (es. ticker errato).
        """
        try:
            import yfinance as yf
        except ImportError as exc:
            raise DataUnavailableError(
                "yfinance non installato: aggiungilo a requirements.txt (pip install yfinance)."
            ) from exc

        try:
            raw = yf.Ticker(symbol).history(
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),  # 'end' di yfinance e' esclusivo
                interval=_YF_INTERVALS[self._timeframe],
                auto_adjust=(self._adjustment == "adjusted"),
                actions=False,
            )
        except Exception as exc:  # errori di rete / API: li rendiamo espliciti
            raise DataUnavailableError(f"[{symbol}] download yfinance fallito: {exc}") from exc

        if raw is None or raw.empty:
            raise DataUnavailableError(
                f"[{symbol}] yfinance non ha restituito dati tra {start} e {end} "
                "(ticker errato o fuori intervallo?)."
            )

        frame = raw.rename(columns=str.lower).loc[:, list(BAR_COLUMNS)].copy()
        frame.index = self._normalize_index(pd.DatetimeIndex(frame.index))
        frame.index.name = "timestamp"
        return self._sanitize(frame, symbol)

    def _normalize_index(self, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Rende l'indice naive: data locale per `1d`, UTC per l'intraday."""
        if self._timeframe == "1d":
            if index.tz is not None:
                index = index.tz_localize(None)
            return index.normalize()
        if index.tz is not None:
            index = index.tz_convert("UTC").tz_localize(None)
        return index

    @staticmethod
    def _sanitize(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Pulisce le barre in modo che superino la validazione di `MarketEvent`.

        - rimuove duplicati e righe con prezzi NaN o <= 0;
        - volume NaN -> 0, volume negativo -> 0;
        - ripara incoerenze OHLC (high < max, low > min) forzando high/low.
        """
        out = frame[~frame.index.duplicated(keep="last")].sort_index()
        out = out.dropna(subset=_PRICE_COLUMNS)
        out["volume"] = out["volume"].fillna(0.0).clip(lower=0.0)

        positive = (out[_PRICE_COLUMNS] > 0).all(axis=1)
        dropped = int((~positive).sum())
        out = out[positive].copy()

        high = out[_PRICE_COLUMNS].max(axis=1)
        low = out[_PRICE_COLUMNS].min(axis=1)
        repaired = int(((out["high"] < high) | (out["low"] > low)).sum())
        out["high"], out["low"] = high, low

        if dropped or repaired:
            logger.warning(
                "[%s] pulizia dati: %d righe scartate, %d barre OHLC riparate.",
                symbol,
                dropped,
                repaired,
            )
        return out

    def _store_bars(self, symbol: str, bars: pd.DataFrame, start: date, end: date) -> None:
        """Salva barre e intervallo coperto in una singola transazione."""
        rows = [
            (
                symbol,
                self._timeframe,
                self._adjustment,
                ts.strftime(_TS_FORMAT),
                float(o),
                float(h),
                float(l),
                float(c),
                float(v),
            )
            for ts, o, h, l, c, v in zip(
                bars.index, bars["open"], bars["high"], bars["low"], bars["close"], bars["volume"]
            )
        ]
        with closing(self._connect()) as conn, conn:  # `with conn` = commit/rollback
            conn.executemany(
                "INSERT OR REPLACE INTO ohlcv_bars "
                "(symbol, timeframe, adjustment, timestamp, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.execute(
                "INSERT OR REPLACE INTO download_log "
                "(symbol, timeframe, adjustment, requested_start, requested_end, downloaded_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    symbol,
                    self._timeframe,
                    self._adjustment,
                    start.isoformat(),
                    end.isoformat(),
                    datetime.now().strftime(_TS_FORMAT),
                ),
            )
        logger.info("[%s] salvate %d barre in %s", symbol, len(rows), self._db_path)

    def _load_frame(self) -> pd.DataFrame:
        """
        Carica da SQLite le barre di tutti i simboli e applica il taglio di warmup.

        Returns:
            DataFrame ordinato per timestamp, con colonna `symbol` e `BAR_COLUMNS`;
            a parita' di timestamp i simboli seguono l'ordine dell'universo.

        Raises:
            DataUnavailableError: se qualche simbolo non ha barre in [start, end].
        """
        lower = self._fetch_start.isoformat()
        upper = (self._end_date + timedelta(days=1)).isoformat()  # estremo esclusivo

        frames: list[pd.DataFrame] = []
        with closing(self._connect()) as conn:
            for symbol in self._symbols:
                df = pd.read_sql_query(
                    "SELECT timestamp, open, high, low, close, volume FROM ohlcv_bars "
                    "WHERE symbol = ? AND timeframe = ? AND adjustment = ? "
                    "AND timestamp >= ? AND timestamp < ? ORDER BY timestamp",
                    conn,
                    params=(symbol, self._timeframe, self._adjustment, lower, upper),
                    parse_dates=["timestamp"],
                    index_col="timestamp",
                )
                if not df.empty:
                    df["symbol"] = symbol
                    frames.append(df)

        if not frames:
            raise DataUnavailableError(
                f"Nessun dato in {self._db_path} per {list(self._symbols)}: "
                "abilita `auto_download` o popola il database."
            )

        # concat in ordine d'universo + sort stabile => ordine simboli deterministico
        frame = pd.concat(frames).sort_index(kind="stable")

        start_ts = pd.Timestamp(self._start_date)
        if self._pre_start_bars > 0:
            before = frame.index[frame.index < start_ts].unique()
            if len(before) < self._pre_start_bars:
                logger.warning(
                    "Storico precedente a %s insufficiente: %d barre su %d di warmup.",
                    self._start_date,
                    len(before),
                    self._pre_start_bars,
                )
            first_ts = before[-self._pre_start_bars :][0] if len(before) else start_ts
        else:
            first_ts = start_ts
        frame = frame[frame.index >= first_ts]

        in_window = frame.loc[frame.index >= start_ts, "symbol"].unique()
        missing = [s for s in self._symbols if s not in in_window]
        if missing:
            raise DataUnavailableError(
                f"Nessuna barra tra {self._start_date} e {self._end_date} per: {missing}."
            )
        return frame