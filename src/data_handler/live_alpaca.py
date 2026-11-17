"""
src/data_handler/live_alpaca.py

`AlpacaDataHandler`: sorgente dati per Paper Trading basata sul polling REST
dell'API dati di Alpaca (libreria `alpaca-py`).

Espone la stessa identica interfaccia dell'handler storico (`has_next`,
`update_bars`, `get_latest_bars`, ereditati da `BaseDataHandler`), quindi
`TradingEngine` non cambia di una riga tra backtest e live.

Funzionamento:
    1. Avvio: scarica le ultime `warmup_bars` barre CHIUSE (raggruppate per
       timestamp) e le mette in coda. L'engine le consuma subito come "replay":
       costruiscono lo storico e, grazie al conteggio di `warmup_bars`, solo
       l'ultima (la piu' recente) e' abilitata a generare segnali.
    2. Regime: `_next_batch()` svuota la coda; se e' vuota interroga Alpaca ogni
       `poll_interval_seconds`, accoda le nuove barre chiuse (una per timestamp,
       in ordine) e rimane in attesa finche' non ne arriva una.
    3. Arresto: `stop()` (es. da un handler di SIGINT in main.py) oppure
       `max_live_batches` fanno terminare il ciclo in modo pulito.

Solo barre CHIUSE (zero look-ahead anche in live):
    - Intraday: la barra e' chiusa se `inizio + durata + settle_seconds <= adesso`.
    - Giornaliero: la barra del giorno D e' chiusa dopo le 16:00 New York di D
      + `settle_seconds`. Si usa sempre le 16:00, anche nelle mezze giornate
      (13:00): si attende di piu' ma non si consuma mai una barra parziale.
    Una barra pubblicata in ritardo per un timestamp gia' emesso viene scartata.

Credenziali:
    Lette SOLO da variabili d'ambiente, i cui nomi stanno in settings.yaml
    (`broker.api_key_env`, `broker.api_secret_env`). Le chiavi non vengono mai
    scritte nel codice, nei file di configurazione o nei log.

Allineamento con config/settings.yaml:
    universe.symbols / timeframe / price_adjustment, backtest.warmup_bars,
    broker.api_key_env / api_secret_env
    (opzionali: broker.data_feed, broker.poll_interval_seconds).
"""

from __future__ import annotations

import logging
import math
import os
import threading
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, time, timedelta, timezone
from typing import Any, Final, NamedTuple
from zoneinfo import ZoneInfo

from src.core.events import MarketEvent
from src.data_handler.base import BaseDataHandler
from src.data_handler.historical import DataUnavailableError

__all__ = ["MissingCredentialsError", "AlpacaDataHandler"]

logger = logging.getLogger(__name__)

_NY: Final[ZoneInfo] = ZoneInfo("America/New_York")
_MARKET_CLOSE: Final[time] = time(16, 0)

# Margine di calendario per coprire festivi e weekend nel recupero iniziale.
_CALENDAR_DAYS_PER_TRADING_DAY: Final[float] = 1.5
_CALENDAR_BUFFER_DAYS: Final[int] = 10


class MissingCredentialsError(RuntimeError):
    """Variabili d'ambiente con le credenziali Alpaca mancanti o vuote."""


class _TimeframeSpec(NamedTuple):
    """Descrizione di un timeframe del progetto in termini Alpaca."""

    amount: int
    unit: str                  # nome membro di alpaca TimeFrameUnit
    duration: timedelta | None  # None = barra giornaliera (regola dedicata)
    bars_per_day: int          # barre per seduta regolare (stima per il lookback)


_TIMEFRAMES: Final[Mapping[str, _TimeframeSpec]] = {
    "1m": _TimeframeSpec(1, "Minute", timedelta(minutes=1), 390),
    "5m": _TimeframeSpec(5, "Minute", timedelta(minutes=5), 78),
    "15m": _TimeframeSpec(15, "Minute", timedelta(minutes=15), 26),
    "1h": _TimeframeSpec(1, "Hour", timedelta(hours=1), 7),
    "1d": _TimeframeSpec(1, "Day", None, 1),
}


class _RawBar(NamedTuple):
    """Barra grezza restituita da Alpaca (timestamp = INIZIO barra, UTC aware)."""

    symbol: str
    start_utc: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class AlpacaDataHandler(BaseDataHandler):
    """
    DataHandler live/paper basato su polling REST di Alpaca.

    Args:
        symbols: Universo di ticker (`universe.symbols`).
        timeframe: Granularita' (`universe.timeframe`): 1m, 5m, 15m, 1h, 1d.
        warmup_bars: Barre di riscaldamento dell'engine (`backtest.warmup_bars`),
            scaricate all'avvio.
        price_adjustment: "adjusted" (split + dividendi) oppure "raw" (solo
            split, coerente con `HistoricalDataHandler`).
        feed: Feed dati Alpaca: "iex" (gratuito) o "sip" (a pagamento). Con IEX i
            volumi sono solo quelli della borsa IEX, non quelli consolidati.
        api_key_env: Nome della variabile d'ambiente con la API key.
        api_secret_env: Nome della variabile d'ambiente con la API secret.
        poll_interval_seconds: Pausa tra due interrogazioni quando non ci sono
            nuove barre (per il timeframe `1d` bastano valori alti, es. 300).
        settle_seconds: Margine dopo la chiusura teorica prima di considerare
            definitiva una barra (evita barre parziali e pubblicazioni tardive).
        max_consecutive_errors: Errori API consecutivi tollerati prima di
            interrompere con `DataUnavailableError`.
        max_live_batches: Se impostato, termina dopo N timestamp live (utile per
            prove brevi in Paper Trading). Il replay iniziale non conta.
        clock: Sorgente dell'ora corrente (UTC aware); iniettabile nei test.

    Raises:
        ValueError: parametri incoerenti.
        MissingCredentialsError: variabili d'ambiente assenti.
        DataUnavailableError: `alpaca-py` non installato o dati iniziali assenti.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        timeframe: str,
        warmup_bars: int,
        *,
        price_adjustment: str = "adjusted",
        feed: str = "iex",
        api_key_env: str = "ALPACA_API_KEY",
        api_secret_env: str = "ALPACA_API_SECRET",
        poll_interval_seconds: float = 60.0,
        settle_seconds: float = 60.0,
        max_consecutive_errors: int = 5,
        max_live_batches: int | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(symbols, timeframe)

        if timeframe not in _TIMEFRAMES:
            raise ValueError(
                f"Timeframe '{timeframe}' non supportato: usa uno tra {sorted(_TIMEFRAMES)}."
            )
        if price_adjustment not in ("adjusted", "raw"):
            raise ValueError("'price_adjustment' deve essere 'adjusted' oppure 'raw'.")
        if warmup_bars < 0:
            raise ValueError(f"'warmup_bars' deve essere >= 0 (ricevuto {warmup_bars}).")
        if poll_interval_seconds <= 0:
            raise ValueError("'poll_interval_seconds' deve essere > 0.")
        if settle_seconds < 0:
            raise ValueError("'settle_seconds' deve essere >= 0.")
        if max_consecutive_errors < 1:
            raise ValueError("'max_consecutive_errors' deve essere >= 1.")
        if max_live_batches is not None and max_live_batches < 1:
            raise ValueError("'max_live_batches' deve essere >= 1 oppure None.")

        self._spec: _TimeframeSpec = _TIMEFRAMES[timeframe]
        self._warmup_bars = warmup_bars
        self._poll_interval = poll_interval_seconds
        self._settle = timedelta(seconds=settle_seconds)
        self._max_consecutive_errors = max_consecutive_errors
        self._max_live_batches = max_live_batches
        self._clock: Callable[[], datetime] = clock or (lambda: datetime.now(timezone.utc))

        trading_days = math.ceil(max(warmup_bars, 1) / self._spec.bars_per_day)
        self._lookback_days: int = (
            math.ceil(trading_days * _CALENDAR_DAYS_PER_TRADING_DAY) + _CALENDAR_BUFFER_DAYS
        )

        # Oggetti alpaca-py (import lazy) e client autenticato.
        self._client: Any
        self._request_cls: Any
        self._alpaca_timeframe: Any
        self._alpaca_adjustment: Any
        self._alpaca_feed: Any
        self._build_alpaca(api_key_env, api_secret_env, feed, price_adjustment == "adjusted")

        self._queue: deque[list[MarketEvent]] = deque()
        self._last_ts: datetime | None = None
        self._stop_event = threading.Event()
        self._replay_remaining = 0
        self._live_emitted = 0
        self._consecutive_errors = 0

        self._load_initial_batches()

    # ------------------------------------------------------------------ #
    # Costruzione da settings.yaml
    # ------------------------------------------------------------------ #
    @classmethod
    def from_settings(
        cls,
        settings: Mapping[str, Any],
        *,
        max_live_batches: int | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> AlpacaDataHandler:
        """
        Costruisce l'handler dal dizionario completo di settings.yaml.

        Chiavi opzionali (con default) nella sezione `broker`:
        `data_feed` (default "iex") e `poll_interval_seconds` (default 60).

        Args:
            settings: Contenuto di settings.yaml gia' caricato (`yaml.safe_load`).
            max_live_batches: Limite opzionale di timestamp live.
            clock: Sorgente dell'ora corrente (solo test).
        """
        universe = settings["universe"]
        broker = settings["broker"]
        return cls(
            symbols=list(universe["symbols"]),
            timeframe=str(universe["timeframe"]),
            warmup_bars=int(settings["backtest"]["warmup_bars"]),
            price_adjustment=str(universe.get("price_adjustment", "adjusted")),
            feed=str(broker.get("data_feed", "iex")),
            api_key_env=str(broker["api_key_env"]),
            api_secret_env=str(broker["api_secret_env"]),
            poll_interval_seconds=float(broker.get("poll_interval_seconds", 60.0)),
            max_live_batches=max_live_batches,
            clock=clock,
        )

    # ------------------------------------------------------------------ #
    # Controllo del ciclo
    # ------------------------------------------------------------------ #
    @property
    def is_stopped(self) -> bool:
        """True se e' stato richiesto l'arresto con `stop()`."""
        return self._stop_event.is_set()

    def stop(self) -> None:
        """
        Richiede l'arresto pulito. Thread-safe e utilizzabile da un handler di
        segnale: interrompe subito anche l'attesa tra due polling.
        """
        self._stop_event.set()

    # ------------------------------------------------------------------ #
    # Interfaccia BaseDataHandler
    # ------------------------------------------------------------------ #
    def has_next(self) -> bool:
        """
        False dopo `stop()` o al raggiungimento di `max_live_batches`; altrimenti
        True (il prossimo timestamp e' in coda oppure verra' atteso).
        """
        if self._stop_event.is_set():
            return False
        if self._max_live_batches is not None and self._live_emitted >= self._max_live_batches:
            return False
        return True

    def _next_batch(self) -> list[MarketEvent]:
        """
        Restituisce il prossimo timestamp di barre chiuse, attendendo (polling) se
        non ancora disponibile. Lista vuota solo se e' stato richiesto lo stop.
        """
        while not self._stop_event.is_set():
            if self._queue:
                batch = self._queue.popleft()
                if self._replay_remaining > 0:
                    self._replay_remaining -= 1
                else:
                    self._live_emitted += 1
                return batch

            self._refill_queue()
            if not self._queue:
                self._stop_event.wait(self._poll_interval)
        return []

    # ------------------------------------------------------------------ #
    # Alpaca: costruzione client e richieste
    # ------------------------------------------------------------------ #
    def _build_alpaca(
        self, api_key_env: str, api_secret_env: str, feed: str, adjusted: bool
    ) -> None:
        """Legge le credenziali dall'ambiente e prepara client e parametri Alpaca."""
        api_key = os.environ.get(api_key_env, "").strip()
        api_secret = os.environ.get(api_secret_env, "").strip()
        if not api_key or not api_secret:
            raise MissingCredentialsError(
                f"Imposta le variabili d'ambiente {api_key_env} e {api_secret_env} "
                "con le chiavi Alpaca Paper Trading (mai nel codice o in settings.yaml)."
            )

        try:
            from alpaca.data.enums import Adjustment, DataFeed
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        except ImportError as exc:
            raise DataUnavailableError(
                "alpaca-py non installato: aggiungilo a requirements.txt (pip install alpaca-py)."
            ) from exc

        try:
            self._alpaca_feed = DataFeed(feed.lower())
        except ValueError as exc:
            raise ValueError(f"Feed Alpaca '{feed}' non valido (usa 'iex' o 'sip').") from exc

        self._alpaca_timeframe = TimeFrame(self._spec.amount, getattr(TimeFrameUnit, self._spec.unit))
        self._alpaca_adjustment = Adjustment.ALL if adjusted else Adjustment.SPLIT
        self._request_cls = StockBarsRequest
        self._client = StockHistoricalDataClient(api_key, api_secret)

    def _fetch_bars(self, start: datetime, end: datetime) -> list[_RawBar]:
        """
        Unico punto di I/O verso Alpaca: scarica le barre di tutti i simboli
        nell'intervallo [start, end] (datetime UTC aware).
        """
        request = self._request_cls(
            symbol_or_symbols=list(self._symbols),
            timeframe=self._alpaca_timeframe,
            start=start,
            end=end,
            adjustment=self._alpaca_adjustment,
            feed=self._alpaca_feed,
        )
        barset = self._client.get_stock_bars(request)
        return [
            _RawBar(
                symbol=symbol,
                start_utc=bar.timestamp,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(bar.volume),
            )
            for symbol, bars in barset.data.items()
            for bar in bars
        ]

    # ------------------------------------------------------------------ #
    # Caricamento iniziale e polling
    # ------------------------------------------------------------------ #
    def _load_initial_batches(self) -> None:
        """Accoda le ultime `warmup_bars` barre chiuse (replay + barra piu' recente)."""
        now = self._clock()
        start = now - timedelta(days=self._lookback_days)
        try:
            raw = self._fetch_bars(start, now)
        except Exception as exc:
            raise DataUnavailableError(f"Recupero storico iniziale da Alpaca fallito: {exc}") from exc

        batches = self._to_batches(raw, now=now, after=None)
        if not batches:
            raise DataUnavailableError(
                f"Alpaca non ha restituito barre chiuse per {list(self._symbols)} "
                f"({self._timeframe}): controlla ticker, feed e credenziali."
            )

        needed = max(self._warmup_bars, 1)
        if len(batches) < needed:
            logger.warning(
                "Storico iniziale insufficiente: %d timestamp su %d di warmup.", len(batches), needed
            )
        batches = batches[-needed:]

        self._queue.extend(batches)
        self._replay_remaining = len(batches)
        self._last_ts = batches[-1][0].timestamp
        logger.info(
            "AlpacaDataHandler pronto: %d barre iniziali, ultima chiusa al %s.",
            len(batches),
            self._last_ts,
        )

    def _refill_queue(self) -> None:
        """Interroga Alpaca e accoda le nuove barre chiuse, in ordine di timestamp."""
        if self._last_ts is None:  # non dovrebbe accadere: impostato all'avvio
            raise RuntimeError("Stato interno non inizializzato.")
        now = self._clock()
        try:
            raw = self._fetch_bars(self._to_aware(self._last_ts), now)
        except Exception as exc:
            self._consecutive_errors += 1
            logger.warning(
                "Errore API Alpaca (%d/%d): %s",
                self._consecutive_errors,
                self._max_consecutive_errors,
                exc,
            )
            if self._consecutive_errors >= self._max_consecutive_errors:
                raise DataUnavailableError(
                    f"{self._consecutive_errors} errori API consecutivi: interrompo."
                ) from exc
            return

        self._consecutive_errors = 0
        for batch in self._to_batches(raw, now=now, after=self._last_ts):
            self._queue.append(batch)
            self._last_ts = batch[0].timestamp
            logger.info("Nuove barre chiuse al %s (%d simboli).", self._last_ts, len(batch))

    # ------------------------------------------------------------------ #
    # Conversione barre -> batch di MarketEvent
    # ------------------------------------------------------------------ #
    def _to_batches(
        self, raw: list[_RawBar], *, now: datetime, after: datetime | None
    ) -> list[list[MarketEvent]]:
        """
        Filtra le barre chiuse piu' recenti di `after` e le raggruppa per timestamp.

        Returns:
            Lista di batch in ordine cronologico; dentro ogni batch i simboli
            seguono l'ordine dell'universo. Barre non valide vengono scartate.
        """
        groups: dict[datetime, dict[str, MarketEvent]] = {}
        for bar in raw:
            if bar.symbol not in self._history or not self._is_closed(bar.start_utc, now):
                continue
            timestamp = self._to_naive(bar.start_utc)
            if after is not None and timestamp <= after:
                continue
            try:
                event = MarketEvent(
                    timestamp=timestamp,
                    symbol=bar.symbol,
                    timeframe=self._timeframe,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                )
            except ValueError as exc:
                logger.warning("[%s %s] barra scartata: %s", bar.symbol, timestamp, exc)
                continue
            groups.setdefault(timestamp, {})[bar.symbol] = event

        return [
            [groups[ts][s] for s in self._symbols if s in groups[ts]] for ts in sorted(groups)
        ]

    def _is_closed(self, start_utc: datetime, now: datetime) -> bool:
        """True se la barra che inizia a `start_utc` e' definitivamente chiusa."""
        if self._spec.duration is None:  # giornaliera: chiude alle 16:00 New York
            session_day = start_utc.astimezone(_NY).date()
            close_time = datetime.combine(session_day, _MARKET_CLOSE, tzinfo=_NY)
            return now >= close_time + self._settle
        return now >= start_utc + self._spec.duration + self._settle

    def _to_naive(self, start_utc: datetime) -> datetime:
        """
        Timestamp naive coerente con `HistoricalDataHandler`: data locale New York
        a mezzanotte per `1d`, UTC naive per l'intraday.
        """
        if self._spec.duration is None:
            local = start_utc.astimezone(_NY)
            return local.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
        return start_utc.astimezone(timezone.utc).replace(tzinfo=None)

    def _to_aware(self, naive: datetime) -> datetime:
        """Inverso di `_to_naive`: restituisce un datetime UTC aware."""
        zone = _NY if self._spec.duration is None else timezone.utc
        return naive.replace(tzinfo=zone).astimezone(timezone.utc)