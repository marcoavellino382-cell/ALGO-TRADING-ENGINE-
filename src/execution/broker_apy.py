"""
src/execution/broker_api.py

`AlpacaExecutionHandler`: esecuzione reale in Paper Trading tramite le API REST
di Alpaca (libreria `alpaca-py`).

Espone lo stesso protocollo di `SimulatedExecutionHandler`
(`submit_order`, `on_market_event`, `has_pending_orders`): `TradingEngine` non
cambia tra backtest e paper trading.

Flusso:
    1. `submit_order(order)`: invia l'ordine ad Alpaca (market o limit, time in
       force "day" di default) con `client_order_id = order.order_id`, cosi'
       l'invio e' idempotente. In caso di errore l'ordine NON viene tracciato:
       il RiskManager lo fara' scadere alla barra successiva del simbolo.
    2. `on_market_event(event)`: alla barra t+1 interroga lo stato degli ordini
       del simbolo. Un ordine ancora aperto viene annullato (valido una sola
       barra, come nel simulatore). Per ogni ordine eseguito (anche in parte)
       restituisce UN FillEvent con quantita' e prezzo medio realmente eseguiti.

Anti look-ahead: si valutano solo ordini con `signal_timestamp < event.timestamp`
e il FillEvent ha `timestamp = event.timestamp`, quindi strettamente successivo
al segnale (verificato anche da `FillEvent`).

Prezzo di riferimento e slippage: Alpaca non fornisce un prezzo "pulito", quindi
`reference_price` e' l'open della barra t+1, limitato al prezzo eseguito
(un'esecuzione migliore dell'open conta come slippage zero, come richiede
`FillEvent`). Lo slippage reale e' gia' incluso nel prezzo medio eseguito.

Sicurezza:
    - Solo Paper Trading: il client e' creato sempre con `paper=True` e la
      configurazione con `paper_trading: false` viene rifiutata.
    - Credenziali lette solo con `os.getenv` dalle variabili indicate in
      settings.yaml; mai nel codice e mai nei log.
    - Errori di rete e 5xx/429: nuovi tentativi con attesa crescente, ma prima
      di ogni ripetizione si verifica se l'ordine esiste gia' presso il broker
      (per `client_order_id`), cosi' non si inviano mai ordini duplicati.
    - Errori 4xx (es. fondi insufficienti, ordine non valido): rifiuto
      definitivo, senza nuovi tentativi.

Limiti noti:
    - `stop_price` dell'OrderEvent e' informativo: nessuno stop viene inviato.
    - Commissioni: Alpaca non applica commissioni sulle azioni USA; si puo'
      impostare `commission_per_share` per stimare commissioni regolamentari.
    - Eventuali esecuzioni arrivate dopo un annullamento non vengono registrate:
      usa `fetch_open_positions()` per confrontare lo stato col broker.

Allineamento con config/settings.yaml:
    broker.provider, paper_trading, base_url, api_key_env, api_secret_env
    (opzionale: broker.time_in_force, default "day").
"""

from __future__ import annotations

import logging
import os
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from src.core.events import FillEvent, MarketEvent, OrderEvent, OrderSide, OrderType
from src.data_handler.live_alpaca import MissingCredentialsError

__all__ = ["BrokerError", "BrokerConnectionError", "AlpacaExecutionHandler"]

logger = logging.getLogger(__name__)

#: Stati Alpaca che non evolvono piu'.
_TERMINAL_STATUSES: Final[frozenset[str]] = frozenset(
    {"filled", "canceled", "expired", "rejected", "done_for_day"}
)

#: Time in force supportati (ordini che non restano aperti oltre la seduta).
_ALLOWED_TIME_IN_FORCE: Final[frozenset[str]] = frozenset({"day", "opg"})


class BrokerError(RuntimeError):
    """Errore generico del livello broker (es. libreria non installata)."""


class BrokerConnectionError(BrokerError):
    """Impossibile contattare Alpaca o conto non operativo."""


@dataclass(slots=True)
class _TrackedOrder:
    """Ordine inviato ad Alpaca e non ancora regolato."""

    order: OrderEvent
    broker_order_id: str


def _status_of(broker_order: Any) -> str:
    """Stato dell'ordine Alpaca normalizzato in minuscolo (es. 'filled')."""
    status = getattr(broker_order, "status", "")
    return str(getattr(status, "value", status)).lower()


def _status_code(exc: Exception) -> int | None:
    """Codice HTTP di un errore API, se disponibile (None per errori di rete)."""
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def _to_float(value: Any) -> float:
    """Converte in float i campi numerici di Alpaca (stringhe o None); 0.0 se assenti."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class AlpacaExecutionHandler:
    """
    ExecutionHandler per Paper Trading su Alpaca.

    Args:
        api_key_env: Nome della variabile d'ambiente con la API key.
        api_secret_env: Nome della variabile d'ambiente con la API secret.
        time_in_force: "day" (default) oppure "opg" (market-on-open).
        commission_per_share: Commissione stimata per azione (default 0.0).
        max_submit_attempts: Tentativi massimi per l'invio di un ordine (>= 1).
        retry_backoff_seconds: Attesa base tra tentativi (cresce linearmente).
        client: Client `TradingClient` gia' pronto (solo test); se None viene creato.
        sleep: Funzione di attesa iniettabile (solo test).

    Raises:
        ValueError: parametri incoerenti.
        MissingCredentialsError: variabili d'ambiente assenti.
        BrokerError: `alpaca-py` non installato.
        BrokerConnectionError: Alpaca non raggiungibile o conto bloccato.
    """

    def __init__(
        self,
        *,
        api_key_env: str = "ALPACA_API_KEY",
        api_secret_env: str = "ALPACA_API_SECRET",
        time_in_force: str = "day",
        commission_per_share: float = 0.0,
        max_submit_attempts: int = 3,
        retry_backoff_seconds: float = 1.0,
        client: Any | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if time_in_force.lower() not in _ALLOWED_TIME_IN_FORCE:
            raise ValueError(f"'time_in_force' deve essere uno tra {sorted(_ALLOWED_TIME_IN_FORCE)}.")
        if commission_per_share < 0:
            raise ValueError("'commission_per_share' deve essere >= 0.")
        if max_submit_attempts < 1:
            raise ValueError("'max_submit_attempts' deve essere >= 1.")
        if retry_backoff_seconds < 0:
            raise ValueError("'retry_backoff_seconds' deve essere >= 0.")

        self._commission_per_share = commission_per_share
        self._max_attempts = max_submit_attempts
        self._backoff = retry_backoff_seconds
        self._sleep: Callable[[float], None] = sleep or time.sleep

        self._pending: dict[str, _TrackedOrder] = {}  # order_id -> ordine tracciato
        self._diagnostics: Counter[str] = Counter()

        self._client: Any
        self._side_enum: Any
        self._tif: Any
        self._market_request_cls: Any
        self._limit_request_cls: Any
        self._build_alpaca(api_key_env, api_secret_env, time_in_force.lower(), client)
        self._verify_account()

    @classmethod
    def from_settings(
        cls,
        settings: Mapping[str, Any],
        *,
        client: Any | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> AlpacaExecutionHandler:
        """
        Costruisce l'handler dal dizionario completo di settings.yaml.

        Raises:
            ValueError: provider diverso da alpaca, `paper_trading: false` o
                `base_url` che non punta all'ambiente paper.
        """
        broker = settings["broker"]
        if str(broker.get("provider", "alpaca")).lower() != "alpaca":
            raise ValueError("broker.provider deve essere 'alpaca'.")
        if not bool(broker.get("paper_trading", True)):
            raise ValueError(
                "AlpacaExecutionHandler supporta solo Paper Trading: imposta broker.paper_trading: true."
            )
        base_url = str(broker.get("base_url", ""))
        if base_url and "paper" not in base_url:
            raise ValueError(f"broker.base_url '{base_url}' non e' un endpoint di Paper Trading.")

        return cls(
            api_key_env=str(broker["api_key_env"]),
            api_secret_env=str(broker["api_secret_env"]),
            time_in_force=str(broker.get("time_in_force", "day")),
            client=client,
            sleep=sleep,
        )

    # ------------------------------------------------------------------ #
    # Proprieta' e utilita' di sola lettura
    # ------------------------------------------------------------------ #
    @property
    def pending_orders(self) -> tuple[OrderEvent, ...]:
        """Ordini inviati ad Alpaca e non ancora regolati."""
        return tuple(tracked.order for tracked in self._pending.values())

    @property
    def diagnostics(self) -> dict[str, int]:
        """Contatori di anomalie (invii falliti, ordini rifiutati/scaduti, errori API)."""
        return dict(self._diagnostics)

    def fetch_open_positions(self) -> dict[str, float]:
        """
        Posizioni aperte presso il broker (simbolo -> quantita' con segno), per
        confrontarle con quelle del RiskManager.

        Raises:
            BrokerConnectionError: se la richiesta fallisce.
        """
        try:
            positions = self._client.get_all_positions()
            return {position.symbol: _to_float(position.qty) for position in positions}
        except Exception as exc:
            raise BrokerConnectionError(f"Lettura posizioni da Alpaca fallita: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Protocollo dell'engine
    # ------------------------------------------------------------------ #
    def submit_order(self, order: OrderEvent) -> None:
        """
        Invia l'ordine ad Alpaca. Non solleva errori di rete o di rifiuto: li
        registra in `diagnostics` e non traccia l'ordine.

        Raises:
            ValueError: ordine duplicato o simbolo con ordine gia' pendente
                (errore di programmazione, non del broker).
        """
        if order.order_id in self._pending:
            raise ValueError(f"Ordine {order.order_id} gia' presente.")
        if any(t.order.symbol == order.symbol for t in self._pending.values()):
            raise ValueError(f"[{order.symbol}] esiste gia' un ordine pendente.")

        broker_order = self._submit_with_retry(order)
        if broker_order is None:
            return
        self._pending[order.order_id] = _TrackedOrder(order, str(broker_order.id))
        logger.info(
            "[%s] ordine %s %s %s inviato ad Alpaca (id %s).",
            order.symbol,
            order.side,
            order.quantity,
            order.order_type,
            str(broker_order.id)[:8],
        )

    def has_pending_orders(self) -> bool:
        """True se esistono ordini inviati e non ancora regolati."""
        return bool(self._pending)

    def on_market_event(self, event: MarketEvent) -> list[FillEvent]:
        """
        Regola gli ordini del simbolo alla barra `event`.

        Considera solo ordini con `signal_timestamp < event.timestamp`. Un ordine
        regolato (eseguito, rifiutato, scaduto o annullato) esce dai pendenti.

        Returns:
            Un FillEvent per ogni ordine eseguito, anche solo in parte.
        """
        fills: list[FillEvent] = []
        for tracked in list(self._pending.values()):
            order = tracked.order
            if order.symbol != event.symbol:
                continue
            if order.signal_timestamp >= event.timestamp:
                continue  # mai nella stessa barra del segnale
            fill = self._settle(tracked, event)
            if fill is not None:
                fills.append(fill)
        return fills

    # ------------------------------------------------------------------ #
    # Costruzione client
    # ------------------------------------------------------------------ #
    def _build_alpaca(
        self, api_key_env: str, api_secret_env: str, time_in_force: str, client: Any | None
    ) -> None:
        """Importa alpaca-py e prepara client (sempre paper) e tipi di richiesta."""
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.enums import OrderSide as AlpacaOrderSide
            from alpaca.trading.enums import TimeInForce
            from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
        except ImportError as exc:
            raise BrokerError(
                "alpaca-py non installato: aggiungilo a requirements.txt (pip install alpaca-py)."
            ) from exc

        self._side_enum = AlpacaOrderSide
        self._tif = TimeInForce(time_in_force)
        self._market_request_cls = MarketOrderRequest
        self._limit_request_cls = LimitOrderRequest

        if client is not None:
            self._client = client
            return

        api_key = (os.getenv(api_key_env) or "").strip()
        api_secret = (os.getenv(api_secret_env) or "").strip()
        if not api_key or not api_secret:
            raise MissingCredentialsError(
                f"Imposta le variabili d'ambiente {api_key_env} e {api_secret_env} "
                "con le chiavi Alpaca Paper Trading (mai nel codice o in settings.yaml)."
            )
        self._client = TradingClient(api_key, api_secret, paper=True)

    def _verify_account(self) -> None:
        """Verifica la connessione e che il conto paper sia operativo."""
        try:
            account = self._client.get_account()
        except Exception as exc:
            raise BrokerConnectionError(f"Connessione ad Alpaca fallita: {exc}") from exc
        if getattr(account, "trading_blocked", False):
            raise BrokerConnectionError("Il conto Alpaca risulta bloccato per il trading.")
        logger.info("Conto Alpaca Paper connesso (stato: %s).", getattr(account, "status", "n/d"))

    # ------------------------------------------------------------------ #
    # Invio ordini
    # ------------------------------------------------------------------ #
    def _build_request(self, order: OrderEvent) -> Any:
        """Converte un OrderEvent nella richiesta Alpaca corrispondente."""
        side = self._side_enum.BUY if order.side is OrderSide.BUY else self._side_enum.SELL
        common: dict[str, Any] = {
            "symbol": order.symbol,
            "qty": order.quantity,
            "side": side,
            "time_in_force": self._tif,
            "client_order_id": order.order_id,
        }
        if order.order_type is OrderType.LIMIT:
            return self._limit_request_cls(limit_price=order.limit_price, **common)
        return self._market_request_cls(**common)

    def _submit_with_retry(self, order: OrderEvent) -> Any | None:
        """
        Invia l'ordine con nuovi tentativi sugli errori transitori.

        Prima di ogni ripetizione controlla se l'ordine esiste gia' presso il
        broker (per `client_order_id`): evita duplicati se l'invio precedente era
        andato a buon fine nonostante l'errore (es. timeout in risposta).

        Returns:
            L'ordine Alpaca, oppure None se l'invio e' fallito o e' stato rifiutato.
        """
        request = self._build_request(order)

        for attempt in range(1, self._max_attempts + 1):
            try:
                return self._client.submit_order(order_data=request)
            except Exception as exc:
                existing = self._lookup_by_client_id(order.order_id)
                if existing is not None:
                    logger.warning("[%s] invio in errore ma ordine presente presso il broker.", order.symbol)
                    return existing

                status = _status_code(exc)
                if status is not None and 400 <= status < 500 and status != 429:
                    self._diagnostics["submit_rejected"] += 1
                    logger.error("[%s] ordine rifiutato da Alpaca (HTTP %s): %s", order.symbol, status, exc)
                    return None

                logger.warning(
                    "[%s] errore transitorio nell'invio (%d/%d): %s",
                    order.symbol,
                    attempt,
                    self._max_attempts,
                    exc,
                )
                if attempt < self._max_attempts:
                    self._sleep(self._backoff * attempt)

        self._diagnostics["submit_failed"] += 1
        logger.error("[%s] invio non riuscito dopo %d tentativi.", order.symbol, self._max_attempts)
        return None

    def _lookup_by_client_id(self, client_order_id: str) -> Any | None:
        """Cerca un ordine presso il broker per `client_order_id` (None se assente o errore)."""
        try:
            return self._client.get_order_by_client_id(client_order_id)
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Regolamento ordini
    # ------------------------------------------------------------------ #
    def _fetch(self, tracked: _TrackedOrder) -> Any | None:
        """Stato corrente dell'ordine presso Alpaca (None se la richiesta fallisce)."""
        try:
            return self._client.get_order_by_id(tracked.broker_order_id)
        except Exception as exc:
            logger.warning("[%s] stato ordine non disponibile: %s", tracked.order.symbol, exc)
            return None

    def _settle(self, tracked: _TrackedOrder, event: MarketEvent) -> FillEvent | None:
        """
        Regola un ordine alla barra `event`: lo annulla se ancora aperto e lo
        converte in FillEvent se (parzialmente) eseguito.

        Se lo stato non e' leggibile l'ordine resta pendente e si riprova alla
        barra successiva.
        """
        order = tracked.order
        broker_order = self._fetch(tracked)
        if broker_order is None:
            self._diagnostics["order_status_unavailable"] += 1
            return None

        status = _status_of(broker_order)
        if status not in _TERMINAL_STATUSES:
            logger.info("[%s] ordine ancora aperto (%s): annullo (valido una barra).", order.symbol, status)
            try:
                self._client.cancel_order_by_id(tracked.broker_order_id)
            except Exception as exc:
                logger.warning("[%s] annullamento non riuscito: %s", order.symbol, exc)
            refreshed = self._fetch(tracked)
            if refreshed is not None:
                broker_order, status = refreshed, _status_of(refreshed)
            if status not in _TERMINAL_STATUSES:
                self._diagnostics["order_not_terminal_after_cancel"] += 1
                logger.warning(
                    "[%s] stato '%s' dopo l'annullamento: eventuali esecuzioni tardive "
                    "non verranno registrate; verifica con fetch_open_positions().",
                    order.symbol,
                    status,
                )

        del self._pending[order.order_id]

        filled_qty = min(_to_float(getattr(broker_order, "filled_qty", None)), order.quantity)
        avg_price = _to_float(getattr(broker_order, "filled_avg_price", None))
        if filled_qty <= 0 or avg_price <= 0:
            self._diagnostics[f"order_{status}"] += 1
            logger.warning("[%s] ordine chiuso senza esecuzione (stato: %s).", order.symbol, status)
            return None

        return self._to_fill(order, event, filled_qty, avg_price)

    def _to_fill(
        self, order: OrderEvent, event: MarketEvent, filled_qty: float, avg_price: float
    ) -> FillEvent:
        """Converte un'esecuzione confermata dal broker in FillEvent."""
        # Slippage sempre >= 0 per contratto di FillEvent: un'esecuzione migliore
        # dell'open della barra conta come slippage zero.
        if order.side is OrderSide.BUY:
            reference = min(event.open, avg_price)
        else:
            reference = max(event.open, avg_price)

        return FillEvent(
            timestamp=event.timestamp,
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=filled_qty,
            reference_price=reference,
            fill_price=avg_price,
            commission=filled_qty * self._commission_per_share,
            signal_timestamp=order.signal_timestamp,
            strategy_id=order.strategy_id,
        )