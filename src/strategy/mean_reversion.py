"""
src/strategy/mean_reversion.py

`MeanReversionStrategy`: mean reversion long-only basata sullo Z-score del
prezzo di chiusura, con filtro di trend a lungo periodo.

Logica (tutti gli indicatori usano solo barre chiuse fino alla barra corrente
inclusa, lette con `get_history` -> zero look-ahead):

    Z-score = (close - media(close, lookback_window)) / dev.std(close, lookback_window)

    INGRESSO (LONG), solo se flat sul simbolo:
        zscore_stop < Z <= zscore_entry
        e, se `use_trend_filter`, close > SMA(close, trend_filter_window).
        Il limite inferiore evita di entrare "a coltello" quando il prezzo e'
        gia' oltre il livello di stop.

    USCITA (EXIT), solo se in posizione, in ordine di priorita':
        1. Z <= zscore_stop          -> stop di emergenza
        2. Z >= zscore_exit          -> ritorno alla media (target)
        3. bars_held >= max_holding_bars -> uscita a tempo

    Il filtro di trend e' applicato solo all'ingresso, non alle uscite.

Anti-duplicati (edge-triggered):
    Un segnale non viene ripetuto nella stessa direzione finche' non cambia lo
    stato: la marcatura si azzera quando un fill conferma l'ingresso/uscita
    oppure quando la condizione che l'aveva generato non e' piu' vera
    (la strategia si "riarma"). Cosi' un ordine rifiutato dal RiskManager non
    produce una raffica di segnali identici barra dopo barra.

Allineamento con config/settings.yaml:
    strategy.name                                -> strategy_id ("mean_reversion")
    strategy.mean_reversion.lookback_window      -> MeanReversionParams.lookback_window
    strategy.mean_reversion.zscore_entry         -> MeanReversionParams.zscore_entry
    strategy.mean_reversion.zscore_exit          -> MeanReversionParams.zscore_exit
    strategy.mean_reversion.zscore_stop          -> MeanReversionParams.zscore_stop
    strategy.mean_reversion.max_holding_bars     -> MeanReversionParams.max_holding_bars
    strategy.mean_reversion.use_trend_filter     -> MeanReversionParams.use_trend_filter
    strategy.mean_reversion.trend_filter_window  -> MeanReversionParams.trend_filter_window
    risk.position_sizing.atr_window              -> MeanReversionParams.atr_window
        (solo per calcolare l'ATR informativo nei metadata del segnale)
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Final

import pandas as pd

from src.core.events import MarketEvent, SignalDirection, SignalEvent
from src.data_handler.base import BaseDataHandler
from src.strategy.base import BaseStrategy

__all__ = ["STRATEGY_NAME", "MeanReversionParams", "MeanReversionStrategy"]

logger = logging.getLogger(__name__)

#: Valore di `strategy.name` in settings.yaml che seleziona questa strategia.
STRATEGY_NAME: Final[str] = "mean_reversion"

#: Una deviazione standard sotto questa frazione della media e' considerata nulla.
_MIN_REL_STD: Final[float] = 1e-9


# --------------------------------------------------------------------------- #
# Parametri
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class MeanReversionParams:
    """
    Parametri della strategia, validati in fase di costruzione.

    Attributes:
        lookback_window: Barre per media e deviazione standard dello Z-score (>= 2).
        zscore_entry: Soglia di ingresso long (< 0), es. -2.0.
        zscore_exit: Soglia di uscita sul ritorno alla media (> zscore_entry), es. 0.0.
        zscore_stop: Soglia di stop di emergenza (< zscore_entry), es. -3.5.
        max_holding_bars: Barre massime in posizione, barra di ingresso inclusa (>= 1).
        use_trend_filter: Se True entra solo con close > SMA di lungo periodo.
        trend_filter_window: Finestra della SMA del filtro di trend (>= 2).
        atr_window: Finestra dell'ATR informativo (>= 1), da `risk.position_sizing`.
    """

    lookback_window: int
    zscore_entry: float
    zscore_exit: float
    zscore_stop: float
    max_holding_bars: int
    use_trend_filter: bool
    trend_filter_window: int
    atr_window: int

    def __post_init__(self) -> None:
        if self.lookback_window < 2:
            raise ValueError("'lookback_window' deve essere >= 2.")
        if self.max_holding_bars < 1:
            raise ValueError("'max_holding_bars' deve essere >= 1.")
        if self.trend_filter_window < 2:
            raise ValueError("'trend_filter_window' deve essere >= 2.")
        if self.atr_window < 1:
            raise ValueError("'atr_window' deve essere >= 1.")
        if not self.zscore_entry < 0:
            raise ValueError("'zscore_entry' deve essere < 0 (strategia long-only).")
        if not self.zscore_stop < self.zscore_entry < self.zscore_exit:
            raise ValueError(
                "Richiesto zscore_stop < zscore_entry < zscore_exit "
                f"(ricevuti {self.zscore_stop}, {self.zscore_entry}, {self.zscore_exit})."
            )

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> MeanReversionParams:
        """
        Legge i parametri dal dizionario completo di settings.yaml.

        Args:
            settings: Contenuto di settings.yaml gia' caricato (`yaml.safe_load`).
        """
        params = settings["strategy"]["mean_reversion"]
        return cls(
            lookback_window=int(params["lookback_window"]),
            zscore_entry=float(params["zscore_entry"]),
            zscore_exit=float(params["zscore_exit"]),
            zscore_stop=float(params["zscore_stop"]),
            max_holding_bars=int(params["max_holding_bars"]),
            use_trend_filter=bool(params["use_trend_filter"]),
            trend_filter_window=int(params["trend_filter_window"]),
            atr_window=int(settings["risk"]["position_sizing"]["atr_window"]),
        )


# --------------------------------------------------------------------------- #
# Strategia
# --------------------------------------------------------------------------- #
class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion long-only su Z-score, con filtro di trend e uscite a soglia/tempo.

    Metadata dei segnali (tutti `float`):
        - sempre:  "zscore", "mean", "std", "close"
        - se disponibili: "atr" (media semplice del True Range su `atr_window`
          barre) e "trend_sma" (solo con filtro di trend attivo)
        - solo EXIT: "exit_reason" (vedi costanti EXIT_REASON_*) e "bars_held"

    Args:
        data_handler: DataHandler da cui leggere lo storico.
        params: Parametri validati della strategia.
        strategy_id: Identificativo (default `strategy.name` = "mean_reversion").
    """

    #: Codici numerici di `metadata["exit_reason"]`.
    EXIT_REASON_TARGET: ClassVar[float] = 1.0
    EXIT_REASON_STOP: ClassVar[float] = 2.0
    EXIT_REASON_TIME: ClassVar[float] = 3.0

    def __init__(
        self,
        data_handler: BaseDataHandler,
        params: MeanReversionParams,
        strategy_id: str = STRATEGY_NAME,
    ) -> None:
        super().__init__(strategy_id, data_handler)
        self._params = params
        # Ultimo segnale emesso e non ancora confermato da un fill (anti-duplicati).
        self._pending_direction: dict[str, SignalDirection] = {}

    @classmethod
    def from_settings(
        cls, settings: Mapping[str, Any], data_handler: BaseDataHandler
    ) -> MeanReversionStrategy:
        """
        Costruisce la strategia dal dizionario completo di settings.yaml.

        Raises:
            ValueError: se `strategy.name` non e' "mean_reversion" o i parametri
                sono incoerenti.
        """
        name = str(settings["strategy"]["name"])
        if name != STRATEGY_NAME:
            raise ValueError(
                f"strategy.name = '{name}': MeanReversionStrategy richiede '{STRATEGY_NAME}'."
            )
        return cls(data_handler, MeanReversionParams.from_settings(settings), strategy_id=name)

    # ------------------------------------------------------------------ #
    # Interfaccia BaseStrategy
    # ------------------------------------------------------------------ #
    @property
    def params(self) -> MeanReversionParams:
        """Parametri correnti della strategia (immutabili)."""
        return self._params

    @property
    def required_history(self) -> int:
        """Max tra finestra Z-score, ATR (+1 per il close precedente) e filtro di trend."""
        p = self._params
        return max(
            p.lookback_window,
            p.atr_window + 1,
            p.trend_filter_window if p.use_trend_filter else 0,
        )

    def on_fill(self, fill: FillEvent) -> None:  # noqa: F821 (import sotto, per leggibilita')
        """Un fill conferma il cambio di stato: azzera la marcatura anti-duplicati."""
        super().on_fill(fill)
        if fill.strategy_id == self.strategy_id:
            self._pending_direction.pop(fill.symbol, None)

    def generate_signals(self, event: MarketEvent) -> list[SignalEvent]:
        """Valuta la barra chiusa `event` ed emette al piu' un segnale."""
        symbol = event.symbol
        bars = self.get_history(symbol, self.required_history, event.timestamp)
        if bars is None:
            return []  # storico insufficiente: nessun segnale

        closes = bars["close"]
        stats = self._zscore(closes, self._params.lookback_window)
        if stats is None:
            return []  # deviazione standard nulla: Z-score non definito
        zscore, mean, std = stats

        metadata: dict[str, float] = {
            "zscore": zscore,
            "mean": mean,
            "std": std,
            "close": event.close,
        }
        atr = self._average_true_range(bars, self._params.atr_window)
        if atr is not None:
            metadata["atr"] = atr

        if self.is_flat(symbol):
            return self._entry_signals(event, closes, zscore, metadata)
        if self.is_long(symbol):
            return self._exit_signals(event, zscore, metadata)
        return []  # posizione corta: non gestita da questa strategia long-only

    # ------------------------------------------------------------------ #
    # Regole di ingresso e uscita
    # ------------------------------------------------------------------ #
    def _entry_signals(
        self,
        event: MarketEvent,
        closes: pd.Series,
        zscore: float,
        metadata: dict[str, float],
    ) -> list[SignalEvent]:
        """Ingresso long se Z-score nella banda di ingresso e trend favorevole."""
        p = self._params
        symbol = event.symbol

        trend_ok = True
        if p.use_trend_filter:
            trend_sma = float(closes.iloc[-p.trend_filter_window :].mean())
            metadata["trend_sma"] = trend_sma
            trend_ok = event.close > trend_sma

        in_entry_band = p.zscore_stop < zscore <= p.zscore_entry
        if not (in_entry_band and trend_ok):
            # Condizione non piu' vera: la strategia si riarma per il prossimo ingresso.
            if self._pending_direction.get(symbol) is SignalDirection.LONG:
                del self._pending_direction[symbol]
            return []

        if self._pending_direction.get(symbol) is SignalDirection.LONG:
            return []  # LONG gia' emesso e non ancora eseguito: niente duplicati

        self._pending_direction[symbol] = SignalDirection.LONG
        strength = min(1.0, abs(zscore) / abs(p.zscore_stop))
        logger.debug("[%s %s] LONG zscore=%.2f", symbol, event.timestamp, zscore)
        return [self.make_signal(event, SignalDirection.LONG, strength, metadata)]

    def _exit_signals(
        self, event: MarketEvent, zscore: float, metadata: dict[str, float]
    ) -> list[SignalEvent]:
        """Uscita per stop, ritorno alla media o scadenza temporale (in quest'ordine)."""
        p = self._params
        symbol = event.symbol
        held = self.bars_held(symbol)

        reason: float | None
        if zscore <= p.zscore_stop:
            reason = self.EXIT_REASON_STOP
        elif zscore >= p.zscore_exit:
            reason = self.EXIT_REASON_TARGET
        elif held >= p.max_holding_bars:
            reason = self.EXIT_REASON_TIME
        else:
            reason = None

        if reason is None:
            if self._pending_direction.get(symbol) is SignalDirection.EXIT:
                del self._pending_direction[symbol]  # riarmo
            return []

        if self._pending_direction.get(symbol) is SignalDirection.EXIT:
            return []  # EXIT gia' emesso e non ancora eseguito

        self._pending_direction[symbol] = SignalDirection.EXIT
        metadata["exit_reason"] = reason
        metadata["bars_held"] = float(held)
        logger.debug(
            "[%s %s] EXIT reason=%s zscore=%.2f held=%d", symbol, event.timestamp, reason, zscore, held
        )
        return [self.make_signal(event, SignalDirection.EXIT, 1.0, metadata)]

    # ------------------------------------------------------------------ #
    # Indicatori (funzioni pure sulle barre passate)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _zscore(closes: pd.Series, window: int) -> tuple[float, float, float] | None:
        """
        Z-score dell'ultimo close rispetto agli ultimi `window` close (inclusa la
        barra corrente, gia' chiusa).

        Returns:
            (zscore, media, deviazione standard campionaria) oppure None se la
            deviazione standard non e' definita o e' trascurabile.
        """
        sample = closes.iloc[-window:]
        mean = float(sample.mean())
        std = float(sample.std(ddof=1))
        if not math.isfinite(std) or std <= _MIN_REL_STD * abs(mean):
            return None
        return (float(closes.iloc[-1]) - mean) / std, mean, std

    @staticmethod
    def _average_true_range(bars: pd.DataFrame, window: int) -> float | None:
        """
        ATR = media semplice del True Range sulle ultime `window` barre, con
        TR = max(high - low, |high - close_prec|, |low - close_prec|).

        Returns:
            ATR oppure None se servono piu' di `window + 1` barre.
        """
        tail = bars.iloc[-(window + 1) :]
        if len(tail) < window + 1:
            return None
        prev_close = tail["close"].shift(1)
        true_range = pd.concat(
            [
                tail["high"] - tail["low"],
                (tail["high"] - prev_close).abs(),
                (tail["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = float(true_range.iloc[1:].mean())
        return atr if math.isfinite(atr) else None