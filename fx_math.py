"""CFA cross-rate, triangular arbitrage, remittance markup, and FX volatility math."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from fx_data import OrderBookQuote


@dataclass
class ArbitrageLeg:
    """Single conversion leg in a triangular FX path."""

    step: int
    from_ccy: str
    to_ccy: str
    mid_rate: float
    effective_rate: float
    half_spread_bps: float
    fee_bps: float
    input_amount: float
    output_amount: float
    rate_source: str


@dataclass
class TriangularArbitrageResult:
    """Full result of a round-trip triangular FX trade (A -> B -> C -> A)."""

    path_label: str
    currencies: Tuple[str, str, str]
    initial_amount: float
    gross_final_amount: float
    net_final_amount: float
    gross_profit: float
    net_profit: float
    gross_return_bps: float
    net_return_bps: float
    implied_product_mid: float
    break_even_total_friction_bps: float
    legs: List[ArbitrageLeg]


@dataclass
class RemittanceTierComparison:
    """Cost & savings breakdown for a remittance provider tier."""

    tier_name: str
    spread_pct: float
    wire_fee: float
    net_converted_source: float
    effective_rate: float
    received_dest: float
    mid_benchmark_dest: float
    hidden_fx_cost_source: float
    total_cost_source: float
    total_cost_pct: float
    saved_per_transfer: float
    saved_per_year: float


def compute_cross_rate(rates_vs_base: Dict[str, float], from_ccy: str, to_ccy: str) -> float:
    """Compute mid cross-rate S(from_ccy -> to_ccy) = units of `to_ccy` per 1 `from_ccy`.

    Given spot rates expressed as units of currency per 1 unit of base (e.g., USD):
    S(A -> B) = R(base -> B) / R(base -> A).
    """
    src = from_ccy.upper().strip()
    dst = to_ccy.upper().strip()
    if src == dst:
        return 1.0
    if src not in rates_vs_base or dst not in rates_vs_base:
        missing = [c for c in (src, dst) if c not in rates_vs_base]
        raise KeyError(f"Missing currency rate(s): {', '.join(missing)}")

    r_src = float(rates_vs_base[src])
    r_dst = float(rates_vs_base[dst])
    if r_src <= 0 or r_dst <= 0:
        raise ValueError("Spot rates must be strictly positive.")
    return r_dst / r_src


def build_cross_rate_matrix(
    rates_vs_base: Dict[str, float],
    currencies: Iterable[str],
) -> pd.DataFrame:
    """Build an N x N cross-rate matrix where entry (row=A, col=B) is units of B per 1 unit of A."""
    ccy_list = [c.upper().strip() for c in currencies]
    matrix = []
    for row_ccy in ccy_list:
        row_vals = {}
        for col_ccy in ccy_list:
            row_vals[col_ccy] = compute_cross_rate(rates_vs_base, row_ccy, col_ccy)
        matrix.append(row_vals)
    return pd.DataFrame(matrix, index=ccy_list)


def evaluate_triangular_arbitrage(
    rates_vs_base: Dict[str, float],
    ccy_a: str,
    ccy_b: str,
    ccy_c: str,
    initial_amount: float = 100_000.0,
    half_spread_bps: float = 2.5,
    fee_bps_per_leg: float = 1.0,
    kraken_quotes: Optional[Dict[Tuple[str, str], OrderBookQuote]] = None,
    use_kraken_live_spreads: bool = False,
    leg_shock_bps: Optional[Dict[Tuple[str, str], float]] = None,
) -> TriangularArbitrageResult:
    """Evaluate a round-trip triangular FX trade A -> B -> C -> A.

    Parameters
    ----------
    rates_vs_base : dict
        Mid-market rates relative to a common base (e.g. USD).
    ccy_a, ccy_b, ccy_c : str
        Three distinct ISO currency codes defining path A -> B -> C -> A.
    initial_amount : float
        Starting notional in `ccy_a`.
    half_spread_bps : float
        Assumed half bid-ask spread per leg in basis points when Kraken quote is
        unavailable or live Kraken spreads are disabled.
    fee_bps_per_leg : float
        Explicit brokerage/platform execution fee per leg in basis points.
    kraken_quotes : dict, optional
        Live top-of-book quotes keyed by (base, quote).
    use_kraken_live_spreads : bool
        If True and (X, Y) exists in `kraken_quotes`, sell X for Y at Kraken's live
        bid (and compute mid from Kraken's order book) instead of the synthetic spread.
    leg_shock_bps : dict, optional
        Optional synthetic mispricing shock in bps keyed by (from_ccy, to_ccy),
        useful for simulating localized dealer mispricings.
    """
    a, b, c = ccy_a.upper().strip(), ccy_b.upper().strip(), ccy_c.upper().strip()
    if len({a, b, c}) != 3:
        raise ValueError("Please select three distinct currencies for triangular arbitrage.")
    if initial_amount <= 0:
        raise ValueError("Initial notional amount must be positive.")

    pairs = [(a, b), (b, c), (c, a)]
    shock_map = leg_shock_bps or {}

    gross_amt = float(initial_amount)
    net_amt = float(initial_amount)
    implied_product_mid = 1.0
    legs: List[ArbitrageLeg] = []

    for idx, (src, dst) in enumerate(pairs, start=1):
        shock_factor = 1.0 + (float(shock_map.get((src, dst), 0.0)) / 10_000.0)
        quote = kraken_quotes.get((src, dst)) if (use_kraken_live_spreads and kraken_quotes) else None

        if quote is not None:
            mid_rate = quote.mid * shock_factor
            # Selling 1 unit of `src` to receive `dst` executes at the bid price.
            bid_rate = quote.bid * shock_factor
            leg_half_spread_bps = ((mid_rate - bid_rate) / mid_rate) * 10_000.0 if mid_rate > 0 else 0.0
            rate_source = quote.source
        else:
            mid_rate = compute_cross_rate(rates_vs_base, src, dst) * shock_factor
            leg_half_spread_bps = max(float(half_spread_bps), 0.0)
            bid_rate = mid_rate * (1.0 - leg_half_spread_bps / 10_000.0)
            rate_source = "Spot Mid +/- Assumed Spread"

        fee_bps = max(float(fee_bps_per_leg), 0.0)
        effective_rate = bid_rate * (1.0 - fee_bps / 10_000.0)

        gross_amt = gross_amt * mid_rate
        next_net_amt = net_amt * effective_rate
        implied_product_mid *= mid_rate

        legs.append(
            ArbitrageLeg(
                step=idx,
                from_ccy=src,
                to_ccy=dst,
                mid_rate=mid_rate,
                effective_rate=effective_rate,
                half_spread_bps=leg_half_spread_bps,
                fee_bps=fee_bps,
                input_amount=net_amt,
                output_amount=next_net_amt,
                rate_source=rate_source,
            )
        )
        net_amt = next_net_amt

    gross_profit = gross_amt - initial_amount
    net_profit = net_amt - initial_amount
    gross_return_bps = (gross_profit / initial_amount) * 10_000.0
    net_return_bps = (net_profit / initial_amount) * 10_000.0

    # Break-even total friction across the 3 legs in bps:
    # if implied_product_mid > 1, friction below (1 - 1/implied_product_mid)*10_000 yields profit.
    break_even_bps = max((1.0 - (1.0 / implied_product_mid)) * 10_000.0, 0.0) if implied_product_mid > 0 else 0.0

    return TriangularArbitrageResult(
        path_label=f"{a} -> {b} -> {c} -> {a}",
        currencies=(a, b, c),
        initial_amount=initial_amount,
        gross_final_amount=gross_amt,
        net_final_amount=net_amt,
        gross_profit=gross_profit,
        net_profit=net_profit,
        gross_return_bps=gross_return_bps,
        net_return_bps=net_return_bps,
        implied_product_mid=implied_product_mid,
        break_even_total_friction_bps=break_even_bps,
        legs=legs,
    )


def analyze_remittance_tiers(
    mid_rate: float,
    send_amount: float,
    transfers_per_year: int,
    tiers: List[Tuple[str, float, float]],
    benchmark_bank_index: int = 1,
) -> List[RemittanceTierComparison]:
    """Compare remittance providers against mid-market and a benchmark traditional bank.

    Parameters
    ----------
    mid_rate : float
        Live mid-market exchange rate (units of destination currency per 1 source unit).
    send_amount : float
        Total source currency outlay per transfer (inclusive of wire fee).
    transfers_per_year : int
        Annual transfer frequency.
    tiers : list of (tier_name, spread_pct, wire_fee)
        Provider specs where spread_pct is percentage markup (e.g., 4.0 for 4%) and
        wire_fee is flat fee in source currency.
    benchmark_bank_index : int
        Index in `tiers` representing the traditional bank baseline for savings calculations.
    """
    if mid_rate <= 0:
        raise ValueError("Mid-market FX rate must be positive.")
    if send_amount <= 0:
        raise ValueError("Send amount must be positive.")
    if transfers_per_year < 1:
        raise ValueError("Transfers per year must be at least 1.")
    if not tiers:
        raise ValueError("At least one transfer tier is required.")

    mid_benchmark_dest = send_amount * mid_rate
    raw_rows = []
    for name, spread_pct, wire_fee in tiers:
        spread_frac = max(float(spread_pct), 0.0) / 100.0
        fee = max(float(wire_fee), 0.0)
        net_converted = max(send_amount - fee, 0.0)
        eff_rate = mid_rate * max(1.0 - spread_frac, 0.0)
        received_dest = net_converted * eff_rate

        hidden_fx_cost = net_converted * spread_frac
        total_cost_source = send_amount - (received_dest / mid_rate)
        total_cost_pct = (total_cost_source / send_amount) * 100.0

        raw_rows.append(
            {
                "tier_name": name,
                "spread_pct": float(spread_pct),
                "wire_fee": fee,
                "net_converted_source": net_converted,
                "effective_rate": eff_rate,
                "received_dest": received_dest,
                "mid_benchmark_dest": mid_benchmark_dest,
                "hidden_fx_cost_source": hidden_fx_cost,
                "total_cost_source": total_cost_source,
                "total_cost_pct": total_cost_pct,
            }
        )

    bench_idx = min(max(benchmark_bank_index, 0), len(raw_rows) - 1)
    bank_total_cost = raw_rows[bench_idx]["total_cost_source"]

    results: List[RemittanceTierComparison] = []
    for row in raw_rows:
        saved_per_transfer = bank_total_cost - row["total_cost_source"]
        saved_per_year = saved_per_transfer * transfers_per_year
        results.append(
            RemittanceTierComparison(
                tier_name=row["tier_name"],
                spread_pct=row["spread_pct"],
                wire_fee=row["wire_fee"],
                net_converted_source=row["net_converted_source"],
                effective_rate=row["effective_rate"],
                received_dest=row["received_dest"],
                mid_benchmark_dest=row["mid_benchmark_dest"],
                hidden_fx_cost_source=row["hidden_fx_cost_source"],
                total_cost_source=row["total_cost_source"],
                total_cost_pct=row["total_cost_pct"],
                saved_per_transfer=saved_per_transfer,
                saved_per_year=saved_per_year,
            )
        )
    return results


def compute_historical_analytics(
    df: pd.DataFrame,
    vol_window: int = 20,
    sma_short: int = 20,
    sma_long: int = 50,
) -> Tuple[pd.DataFrame, dict]:
    """Enrich a ['date', 'rate'] DataFrame with returns, SMAs, bands, and rolling volatility.

    Returns (enriched_df, summary_metrics).
    """
    if df.empty or len(df) < 3:
        raise ValueError("Need at least 3 historical observations to compute FX analytics.")

    out = df.copy().sort_values("date").reset_index(drop=True)
    out["rate"] = out["rate"].astype(float)
    out["simple_return_pct"] = out["rate"].pct_change() * 100.0
    out["log_return"] = np.log(out["rate"] / out["rate"].shift(1))

    # Annualized rolling volatility using 252 trading days convention.
    out["rolling_vol_ann_pct"] = out["log_return"].rolling(window=vol_window).std() * np.sqrt(252.0) * 100.0
    out["sma_short"] = out["rate"].rolling(window=sma_short).mean()
    out["sma_long"] = out["rate"].rolling(window=sma_long).mean()
    rolling_std_price = out["rate"].rolling(window=sma_short).std()
    out["band_upper"] = out["sma_short"] + 2.0 * rolling_std_price
    out["band_lower"] = out["sma_short"] - 2.0 * rolling_std_price

    # Drawdown from cumulative peak.
    out["cum_max"] = out["rate"].cummax()
    out["drawdown_pct"] = ((out["rate"] / out["cum_max"]) - 1.0) * 100.0

    start_rate = float(out["rate"].iloc[0])
    latest_rate = float(out["rate"].iloc[-1])
    high_rate = float(out["rate"].max())
    low_rate = float(out["rate"].min())
    total_return_pct = ((latest_rate / start_rate) - 1.0) * 100.0
    full_ann_vol_pct = float(out["log_return"].dropna().std() * np.sqrt(252.0) * 100.0)
    latest_roll_vol = out["rolling_vol_ann_pct"].dropna()
    latest_roll_vol_pct = float(latest_roll_vol.iloc[-1]) if not latest_roll_vol.empty else full_ann_vol_pct
    max_drawdown_pct = float(out["drawdown_pct"].min())
    range_percentile = ((latest_rate - low_rate) / (high_rate - low_rate) * 100.0) if high_rate > low_rate else 50.0

    summary = {
        "observations": len(out),
        "start_date": out["date"].iloc[0].date().isoformat(),
        "end_date": out["date"].iloc[-1].date().isoformat(),
        "start_rate": start_rate,
        "latest_rate": latest_rate,
        "period_high": high_rate,
        "period_low": low_rate,
        "total_return_pct": total_return_pct,
        "full_ann_vol_pct": full_ann_vol_pct,
        "latest_roll_vol_pct": latest_roll_vol_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "range_percentile": range_percentile,
    }
    return out, summary
