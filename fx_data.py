"""Public, no-API-key FX market data fetchers with clean error handling and fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, Optional, Tuple

import pandas as pd
import requests


ER_API_URL = "https://open.er-api.com/v6/latest/{base}"
FRANKFURTER_LATEST_URL = "https://api.frankfurter.app/latest"
FRANKFURTER_CURRENCIES_URL = "https://api.frankfurter.app/currencies"
FRANKFURTER_RANGE_URL = "https://api.frankfurter.app/{start}..{end}"
KRAKEN_TICKER_URL = "https://api.kraken.com/0/public/Ticker"

DEFAULT_TIMEOUT = 10

# Common remittance and G10 currencies with human-readable labels.
COMMON_CURRENCIES: Dict[str, str] = {
    "USD": "US Dollar",
    "EUR": "Euro",
    "GBP": "British Pound",
    "JPY": "Japanese Yen",
    "CHF": "Swiss Franc",
    "CAD": "Canadian Dollar",
    "AUD": "Australian Dollar",
    "NZD": "New Zealand Dollar",
    "MXN": "Mexican Peso",
    "INR": "Indian Rupee",
    "PHP": "Philippine Peso",
    "BRL": "Brazilian Real",
    "CNY": "Chinese Yuan",
    "HKD": "Hong Kong Dollar",
    "SGD": "Singapore Dollar",
    "KRW": "South Korean Won",
    "PLN": "Polish Zloty",
    "SEK": "Swedish Krona",
    "NOK": "Norwegian Krone",
    "DKK": "Danish Krone",
    "ZAR": "South African Rand",
    "COP": "Colombian Peso",
    "NGN": "Nigerian Naira",
    "VND": "Vietnamese Dong",
    "TRY": "Turkish Lira",
}

# Kraken fiat pairs mapped to (kraken_query_pair, base, quote).
KRAKEN_FIAT_PAIRS: Dict[str, Tuple[str, str]] = {
    "EURUSD": ("EUR", "USD"),
    "GBPUSD": ("GBP", "USD"),
    "EURGBP": ("EUR", "GBP"),
    "USDJPY": ("USD", "JPY"),
    "USDCAD": ("USD", "CAD"),
    "USDCHF": ("USD", "CHF"),
    "AUDUSD": ("AUD", "USD"),
    "EURCAD": ("EUR", "CAD"),
    "EURCHF": ("EUR", "CHF"),
    "EURJPY": ("EUR", "JPY"),
}


class FXDataError(RuntimeError):
    """Raised when public FX endpoints fail or return invalid payloads."""


@dataclass
class SpotRatesResult:
    """Normalized spot exchange rate snapshot relative to a base currency."""

    base: str
    rates: Dict[str, float]
    provider: str
    timestamp_label: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class OrderBookQuote:
    """Live top-of-book bid/ask quote for a currency pair (base -> quote)."""

    base: str
    quote: str
    bid: float
    ask: float
    last: float
    source: str
    is_inverted: bool = False

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_bps(self) -> float:
        if self.mid <= 0:
            return 0.0
        return ((self.ask - self.bid) / self.mid) * 10_000.0


def _get_json(url: str, params: Optional[dict] = None, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Perform an HTTP GET and validate a JSON dictionary response."""
    try:
        resp = requests.get(
            url,
            params=params,
            timeout=timeout,
            headers={"Accept": "application/json", "User-Agent": "fx-arbitrage-monitor/1.0"},
        )
        resp.raise_for_status()
        payload = resp.json()
    except requests.exceptions.Timeout as exc:
        raise FXDataError(f"Request timed out calling {url}") from exc
    except requests.exceptions.RequestException as exc:
        raise FXDataError(f"Network error calling {url}: {exc}") from exc
    except ValueError as exc:
        raise FXDataError(f"Invalid JSON returned from {url}") from exc

    if not isinstance(payload, dict):
        raise FXDataError(f"Unexpected payload structure from {url}")
    return payload


def fetch_er_api_rates(base: str = "USD") -> SpotRatesResult:
    """Fetch live spot rates from open.er-api.com (no API key required)."""
    base_clean = base.upper().strip()
    payload = _get_json(ER_API_URL.format(base=base_clean))
    if payload.get("result") != "success":
        err_type = payload.get("error-type", "unknown error")
        raise FXDataError(f"open.er-api.com error for {base_clean}: {err_type}")

    raw_rates = payload.get("rates")
    if not isinstance(raw_rates, dict) or not raw_rates:
        raise FXDataError("open.er-api.com returned empty rates")

    rates = {str(k).upper(): float(v) for k, v in raw_rates.items() if isinstance(v, (int, float)) and v > 0}
    rates[base_clean] = 1.0
    ts = str(payload.get("time_last_update_utc", "Unknown timestamp"))
    return SpotRatesResult(
        base=base_clean,
        rates=rates,
        provider="open.er-api.com",
        timestamp_label=ts,
    )


def fetch_frankfurter_latest(base: str = "USD") -> SpotRatesResult:
    """Fetch latest ECB reference rates from api.frankfurter.app."""
    base_clean = base.upper().strip()
    payload = _get_json(FRANKFURTER_LATEST_URL, params={"from": base_clean})
    raw_rates = payload.get("rates")
    if not isinstance(raw_rates, dict) or not raw_rates:
        raise FXDataError(f"Frankfurter returned empty rates for {base_clean}")

    rates = {str(k).upper(): float(v) for k, v in raw_rates.items() if isinstance(v, (int, float)) and v > 0}
    rates[base_clean] = 1.0
    dt = str(payload.get("date", "Unknown date"))
    return SpotRatesResult(
        base=base_clean,
        rates=rates,
        provider="api.frankfurter.app (ECB)",
        timestamp_label=f"ECB Close {dt}",
    )


def fetch_spot_rates(base: str = "USD", preferred_provider: str = "open.er-api.com") -> SpotRatesResult:
    """Fetch spot rates from preferred provider with automatic fallback."""
    providers = (
        [fetch_er_api_rates, fetch_frankfurter_latest]
        if preferred_provider == "open.er-api.com"
        else [fetch_frankfurter_latest, fetch_er_api_rates]
    )

    warnings: list[str] = []
    last_err: Optional[Exception] = None
    for idx, fetcher in enumerate(providers):
        try:
            result = fetcher(base)
            result.warnings.extend(warnings)
            return result
        except FXDataError as exc:
            last_err = exc
            if idx == 0:
                warnings.append(f"Primary provider failed ({exc}); fell back to secondary provider.")

    raise FXDataError(f"All spot FX providers failed: {last_err}")


def _normalize_kraken_pair_key(raw_key: str) -> Optional[Tuple[str, str]]:
    """Map Kraken response pair key (e.g. 'ZEURZUSD' or 'EURUSD') to (base, quote)."""
    cleaned = raw_key.upper().strip()
    if cleaned in KRAKEN_FIAT_PAIRS:
        return KRAKEN_FIAT_PAIRS[cleaned]

    # Kraken often prefixes fiat currencies with 'Z' (e.g., ZEURZUSD, ZUSDZJPY)
    stripped = cleaned.replace("Z", "")
    if stripped in KRAKEN_FIAT_PAIRS:
        return KRAKEN_FIAT_PAIRS[stripped]

    # Fallback: match 6-char currency codes among known fiat codes
    if len(cleaned) == 8 and cleaned.startswith("Z"):
        candidate = cleaned[1:4] + cleaned[5:8]
        if candidate in KRAKEN_FIAT_PAIRS:
            return KRAKEN_FIAT_PAIRS[candidate]
    return None


def fetch_kraken_orderbook_quotes() -> Dict[Tuple[str, str], OrderBookQuote]:
    """Fetch live top-of-book bid/ask quotes from Kraken's public Ticker API.

    Returns a dictionary keyed by (base, quote) including both direct pairs
    and mathematically inverted pairs (bid_inv = 1 / ask, ask_inv = 1 / bid).
    """
    pair_list = ",".join(KRAKEN_FIAT_PAIRS.keys())
    payload = _get_json(KRAKEN_TICKER_URL, params={"pair": pair_list})

    errors = payload.get("error")
    if isinstance(errors, list) and errors:
        raise FXDataError(f"Kraken API returned error: {', '.join(map(str, errors))}")

    result = payload.get("result")
    if not isinstance(result, dict) or not result:
        raise FXDataError("Kraken API returned empty ticker result")

    quotes: Dict[Tuple[str, str], OrderBookQuote] = {}
    for raw_pair, ticker in result.items():
        mapped = _normalize_kraken_pair_key(raw_pair)
        if mapped is None or not isinstance(ticker, dict):
            continue

        try:
            ask = float(ticker["a"][0])
            bid = float(ticker["b"][0])
            last = float(ticker["c"][0])
        except (KeyError, IndexError, TypeError, ValueError):
            continue

        if bid <= 0 or ask <= 0 or bid > ask * 1.1:
            continue

        base, quote = mapped
        quotes[(base, quote)] = OrderBookQuote(
            base=base,
            quote=quote,
            bid=bid,
            ask=ask,
            last=last,
            source="Kraken Public Order Book",
            is_inverted=False,
        )
        # Inverted quote: selling 1 unit of `quote` for `base` hits 1 / ask;
        # buying 1 unit of `quote` with `base` lifts 1 / bid.
        quotes[(quote, base)] = OrderBookQuote(
            base=quote,
            quote=base,
            bid=1.0 / ask,
            ask=1.0 / bid,
            last=1.0 / last if last > 0 else (1.0 / ask + 1.0 / bid) / 2.0,
            source="Kraken Public Order Book (Inverted)",
            is_inverted=True,
        )

    if not quotes:
        raise FXDataError("No usable fiat FX pairs parsed from Kraken Ticker response")
    return quotes


def fetch_frankfurter_currencies() -> Dict[str, str]:
    """Fetch currency names supported by Frankfurter (ECB)."""
    payload = _get_json(FRANKFURTER_CURRENCIES_URL)
    return {str(k).upper(): str(v) for k, v in payload.items()}


def fetch_frankfurter_history(
    base: str,
    target: str,
    days: int = 365,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """Fetch daily historical FX rates from api.frankfurter.app.

    Returns a DataFrame with columns ['date', 'rate'] sorted chronologically.
    """
    base_clean = base.upper().strip()
    target_clean = target.upper().strip()
    if base_clean == target_clean:
        raise FXDataError("Base and target currencies must be different for historical trend analysis.")

    end_dt = end_date or date.today()
    start_dt = end_dt - timedelta(days=max(days, 7))
    url = FRANKFURTER_RANGE_URL.format(start=start_dt.isoformat(), end=end_dt.isoformat())

    payload = _get_json(url, params={"from": base_clean, "to": target_clean})
    raw_rates = payload.get("rates")
    if not isinstance(raw_rates, dict) or not raw_rates:
        raise FXDataError(f"No historical Frankfurter data returned for {base_clean}/{target_clean}.")

    rows = []
    for dt_str, rate_map in raw_rates.items():
        if isinstance(rate_map, dict) and target_clean in rate_map:
            val = float(rate_map[target_clean])
            if val > 0:
                rows.append({"date": pd.to_datetime(dt_str), "rate": val})

    if not rows:
        raise FXDataError(f"Frankfurter had no {target_clean} observations vs {base_clean}.")

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df
