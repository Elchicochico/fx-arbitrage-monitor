"""Unit tests for FX math engine and public API error handling."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import requests

from fx_data import FXDataError, OrderBookQuote, fetch_spot_rates
from fx_math import (
    analyze_remittance_tiers,
    build_cross_rate_matrix,
    compute_cross_rate,
    compute_historical_analytics,
    evaluate_triangular_arbitrage,
)


SAMPLE_RATES = {
    "USD": 1.0,
    "EUR": 0.90,
    "GBP": 0.80,
    "JPY": 150.0,
    "MXN": 20.0,
}


def test_cross_rate_and_matrix_identity():
    assert compute_cross_rate(SAMPLE_RATES, "USD", "EUR") == pytest.approx(0.90)
    assert compute_cross_rate(SAMPLE_RATES, "EUR", "GBP") == pytest.approx(0.80 / 0.90)
    matrix = build_cross_rate_matrix(SAMPLE_RATES, ["USD", "EUR", "GBP"])
    for ccy in ["USD", "EUR", "GBP"]:
        assert matrix.loc[ccy, ccy] == pytest.approx(1.0)
    assert matrix.loc["EUR", "USD"] * matrix.loc["USD", "EUR"] == pytest.approx(1.0)


def test_triangular_no_arbitrage_equilibrium_and_friction():
    res = evaluate_triangular_arbitrage(
        rates_vs_base=SAMPLE_RATES,
        ccy_a="USD",
        ccy_b="EUR",
        ccy_c="GBP",
        initial_amount=100_000.0,
        half_spread_bps=2.5,
        fee_bps_per_leg=1.0,
    )
    assert res.implied_product_mid == pytest.approx(1.0)
    assert res.gross_profit == pytest.approx(0.0, abs=1e-6)
    # 3 legs * (2.5 + 1.0) = 10.5 bps drag => negative net P&L
    assert res.net_profit < 0
    assert res.net_return_bps == pytest.approx(-10.5, abs=0.05)


def test_triangular_positive_arbitrage_under_mispricing_shock():
    res = evaluate_triangular_arbitrage(
        rates_vs_base=SAMPLE_RATES,
        ccy_a="USD",
        ccy_b="EUR",
        ccy_c="GBP",
        initial_amount=100_000.0,
        half_spread_bps=2.0,
        fee_bps_per_leg=1.0,
        leg_shock_bps={("USD", "EUR"): 25.0},
    )
    assert res.gross_return_bps == pytest.approx(25.0, abs=1e-4)
    # Net return approx 25 - 9 = +16 bps
    assert res.net_profit > 0
    assert res.net_return_bps == pytest.approx(16.0, abs=0.1)


def test_triangular_with_kraken_quotes():
    quotes = {
        ("USD", "EUR"): OrderBookQuote("USD", "EUR", bid=0.8998, ask=0.9002, last=0.9000, source="Kraken"),
        ("EUR", "GBP"): OrderBookQuote("EUR", "GBP", bid=0.8887, ask=0.8891, last=0.8889, source="Kraken"),
    }
    res = evaluate_triangular_arbitrage(
        rates_vs_base=SAMPLE_RATES,
        ccy_a="USD",
        ccy_b="EUR",
        ccy_c="GBP",
        initial_amount=10_000.0,
        half_spread_bps=2.5,
        fee_bps_per_leg=0.0,
        kraken_quotes=quotes,
        use_kraken_live_spreads=True,
    )
    assert res.legs[0].rate_source == "Kraken"
    assert res.legs[2].rate_source == "Spot Mid +/- Assumed Spread"
    assert res.net_profit < 0


def test_remittance_markup_and_annual_savings():
    tiers = [
        ("Mid-Market", 0.0, 0.0),
        ("Traditional Bank", 4.0, 35.0),
        ("Low-Cost Tier", 0.4, 4.50),
    ]
    results = analyze_remittance_tiers(
        mid_rate=20.0,
        send_amount=2_500.0,
        transfers_per_year=12,
        tiers=tiers,
        benchmark_bank_index=1,
    )
    bank = results[1]
    low_cost = results[2]

    # Bank: $35 wire + 4% of ($2500 - $35 = $2465) = $35 + $98.60 = $133.60
    assert bank.total_cost_source == pytest.approx(133.60)
    assert bank.saved_per_transfer == pytest.approx(0.0)

    # Low-cost: $4.50 fee + 0.4% of $2495.50 = $4.50 + $9.982 = $14.482
    assert low_cost.total_cost_source == pytest.approx(14.482)
    assert low_cost.saved_per_transfer == pytest.approx(133.60 - 14.482)
    assert low_cost.saved_per_year == pytest.approx((133.60 - 14.482) * 12)


def test_historical_volatility_computation():
    dates = pd.date_range("2026-01-01", periods=60, freq="B")
    rng = np.random.default_rng(42)
    rates = 1.10 * np.exp(np.cumsum(rng.normal(0.0, 0.005, size=60)))
    df = pd.DataFrame({"date": dates, "rate": rates})

    enriched, summary = compute_historical_analytics(df, vol_window=20)
    assert len(enriched) == 60
    assert summary["observations"] == 60
    assert summary["full_ann_vol_pct"] > 0
    assert summary["max_drawdown_pct"] <= 0
    assert enriched["rolling_vol_ann_pct"].dropna().iloc[-1] > 0


def test_fetch_spot_rates_fallback_on_primary_failure():
    class FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, **kwargs):
        if "open.er-api.com" in url:
            raise requests.exceptions.Timeout("timeout")
        return FakeResp({"base": "USD", "date": "2026-11-24", "rates": {"EUR": 0.91, "GBP": 0.76}})

    with patch("fx_data.requests.get", side_effect=fake_get):
        res = fetch_spot_rates("USD", preferred_provider="open.er-api.com")
        assert res.provider == "api.frankfurter.app (ECB)"
        assert res.rates["EUR"] == 0.91
        assert len(res.warnings) == 1

    with patch("fx_data.requests.get", side_effect=requests.exceptions.ConnectionError("down")):
        with pytest.raises(FXDataError):
            fetch_spot_rates("USD")
