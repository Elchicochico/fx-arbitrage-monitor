"""FX Triangular Arbitrage & Remittance Spread Monitor (Streamlit App)."""

from __future__ import annotations

from typing import Dict, Tuple

import altair as alt
import pandas as pd
import streamlit as st

from fx_data import (
    COMMON_CURRENCIES,
    FXDataError,
    OrderBookQuote,
    SpotRatesResult,
    fetch_frankfurter_currencies,
    fetch_frankfurter_history,
    fetch_kraken_orderbook_quotes,
    fetch_spot_rates,
)
from fx_math import (
    analyze_remittance_tiers,
    build_cross_rate_matrix,
    compute_cross_rate,
    compute_historical_analytics,
    evaluate_triangular_arbitrage,
)

st.set_page_config(
    page_title="FX Triangular Arbitrage & Remittance Monitor",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.55rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=120, show_spinner=False)
def cached_spot_rates(base: str, preferred_provider: str) -> SpotRatesResult:
    return fetch_spot_rates(base=base, preferred_provider=preferred_provider)


@st.cache_data(ttl=60, show_spinner=False)
def cached_kraken_quotes() -> Dict[Tuple[str, str], OrderBookQuote]:
    return fetch_kraken_orderbook_quotes()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_frankfurter_currencies() -> Dict[str, str]:
    return fetch_frankfurter_currencies()


@st.cache_data(ttl=900, show_spinner=False)
def cached_history(base: str, target: str, days: int) -> pd.DataFrame:
    return fetch_frankfurter_history(base=base, target=target, days=days)


def format_currency_option(code: str, labels: Dict[str, str]) -> str:
    name = labels.get(code)
    return f"{code} — {name}" if name else code


# ---------------------------------------------------------------------------
# Sidebar: Data Sources & Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("💱 FX Monitor Controls")
    st.caption("Free, zero-API-key live & historical FX endpoints.")

    preferred_provider = st.selectbox(
        "Primary Spot Rate Provider",
        options=["open.er-api.com", "api.frankfurter.app (ECB)"],
        index=0,
        help="Automatically falls back to the secondary endpoint if the primary is unreachable.",
    )

    use_kraken = st.toggle(
        "Overlay Live Kraken Bid/Ask Spreads",
        value=True,
        help="Uses Kraken's public Ticker API order book (best bid/ask) for supported G10 fiat pairs.",
    )

    if st.button("🔄 Refresh Live Quotes", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    # Load Spot FX Snapshot (USD-anchored so cross rates cover all pairs)
    spot_result: SpotRatesResult | None = None
    try:
        spot_result = cached_spot_rates("USD", preferred_provider)
        st.success(f"Spot Feed: **{spot_result.provider}**")
        st.caption(f"Updated: {spot_result.timestamp_label}")
        st.caption(f"Currencies loaded: {len(spot_result.rates)}")
        for w in spot_result.warnings:
            st.warning(w)
    except FXDataError as exc:
        st.error(f"Spot FX feed unavailable: {exc}")

    # Load Kraken Live Order Book Quotes
    kraken_quotes: Dict[Tuple[str, str], OrderBookQuote] = {}
    if use_kraken:
        try:
            kraken_quotes = cached_kraken_quotes()
            st.info(f"Kraken Book: **{len(kraken_quotes) // 2} live fiat pairs** loaded")
        except FXDataError as exc:
            st.warning(f"Kraken order book unavailable (falling back to spot spreads): {exc}")

    # Load Frankfurter Supported Currencies for Tab 3
    frankfurter_currencies: Dict[str, str] = {}
    try:
        frankfurter_currencies = cached_frankfurter_currencies()
    except FXDataError:
        frankfurter_currencies = {
            k: v for k, v in COMMON_CURRENCIES.items() if k in {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "MXN", "INR", "BRL", "CNY", "SGD", "HKD"}
        }

st.title("FX Triangular Arbitrage & Remittance Spread Monitor")
st.caption(
    "Real-time cross-rate no-arbitrage diagnostics, bank remittance markup analysis, and 1-year ECB historical volatility."
)

if spot_result is None:
    st.error("Unable to fetch live spot rates from public endpoints. Check your internet connection and click Refresh.")
    st.stop()

# Build unified currency list & display labels
available_codes = sorted(spot_result.rates.keys())
priority_order = [c for c in COMMON_CURRENCIES if c in spot_result.rates]
remaining_codes = [c for c in available_codes if c not in COMMON_CURRENCIES]
ordered_codes = priority_order + remaining_codes
currency_labels = {**frankfurter_currencies, **COMMON_CURRENCIES}

tab1, tab2, tab3 = st.tabs(
    [
        "1️⃣ Triangular FX Arbitrage",
        "2️⃣ Remittance & Bank Markup",
        "3️⃣ Historical Trend & Volatility",
    ]
)

# ===========================================================================
# TAB 1: TRIANGULAR FX ARBITRAGE CALCULATOR
# ===========================================================================
with tab1:
    st.subheader("Triangular FX Arbitrage & Cross-Rate Matrix")
    st.markdown(
        "Test a round-trip conversion **$A \\rightarrow B \\rightarrow C \\rightarrow A$** against CFA no-arbitrage parity, "
        "including live Kraken order-book spreads (where available) and transaction fee drag."
    )

    col_sel1, col_sel2, col_sel3, col_notional = st.columns([1, 1, 1, 1.2])
    with col_sel1:
        ccy_a = st.selectbox(
            "Currency A (Start / End)",
            options=ordered_codes,
            index=ordered_codes.index("USD") if "USD" in ordered_codes else 0,
            format_func=lambda c: format_currency_option(c, currency_labels),
            key="tri_a",
        )
    with col_sel2:
        ccy_b = st.selectbox(
            "Currency B (Leg 1 Target)",
            options=ordered_codes,
            index=ordered_codes.index("EUR") if "EUR" in ordered_codes else 1,
            format_func=lambda c: format_currency_option(c, currency_labels),
            key="tri_b",
        )
    with col_sel3:
        ccy_c = st.selectbox(
            "Currency C (Leg 2 Target)",
            options=ordered_codes,
            index=ordered_codes.index("GBP") if "GBP" in ordered_codes else 2,
            format_func=lambda c: format_currency_option(c, currency_labels),
            key="tri_c",
        )
    with col_notional:
        notional = st.number_input(
            f"Starting Notional ({ccy_a})",
            min_value=100.0,
            max_value=100_000_000.0,
            value=100_000.0,
            step=10_000.0,
            format="%.2f",
        )

    col_sp, col_fee, col_shock = st.columns(3)
    with col_sp:
        half_spread_bps = st.slider(
            "Assumed Half Bid-Ask Spread (bps / leg)",
            min_value=0.0,
            max_value=50.0,
            value=2.5,
            step=0.5,
            help="Applied to legs without a live Kraken order-book quote (or all legs if Kraken overlay is off).",
        )
    with col_fee:
        fee_bps = st.slider(
            "Broker / Execution Fee (bps / leg)",
            min_value=0.0,
            max_value=30.0,
            value=1.0,
            step=0.5,
            help="Explicit commission / clearing fee per FX leg (1 bp = 0.01%).",
        )
    with col_shock:
        leg1_shock_bps = st.slider(
            f"Simulated Mispricing Shock on {ccy_a}→{ccy_b} (bps)",
            min_value=-50.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
            help="Inject a synthetic quote dislocation on Leg 1 to test arbitrage break-even thresholds.",
        )

    if len({ccy_a, ccy_b, ccy_c}) < 3:
        st.warning("Please choose three **distinct** currencies (A, B, and C) to evaluate a triangular path.")
    else:
        shock_dict = {(ccy_a, ccy_b): leg1_shock_bps} if leg1_shock_bps != 0.0 else None
        fwd_res = evaluate_triangular_arbitrage(
            rates_vs_base=spot_result.rates,
            ccy_a=ccy_a,
            ccy_b=ccy_b,
            ccy_c=ccy_c,
            initial_amount=notional,
            half_spread_bps=half_spread_bps,
            fee_bps_per_leg=fee_bps,
            kraken_quotes=kraken_quotes,
            use_kraken_live_spreads=use_kraken and bool(kraken_quotes),
            leg_shock_bps=shock_dict,
        )
        rev_res = evaluate_triangular_arbitrage(
            rates_vs_base=spot_result.rates,
            ccy_a=ccy_a,
            ccy_b=ccy_c,
            ccy_c=ccy_b,
            initial_amount=notional,
            half_spread_bps=half_spread_bps,
            fee_bps_per_leg=fee_bps,
            kraken_quotes=kraken_quotes,
            use_kraken_live_spreads=use_kraken and bool(kraken_quotes),
            leg_shock_bps=None,
        )

        # KPI Row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "Gross Arbitrage P&L (Mid)",
            f"{fwd_res.gross_profit:+,.2f} {ccy_a}",
            delta=f"{fwd_res.gross_return_bps:+.2f} bps",
        )
        k2.metric(
            "Net Arbitrage P&L (After Spread & Fees)",
            f"{fwd_res.net_profit:+,.2f} {ccy_a}",
            delta=f"{fwd_res.net_return_bps:+.2f} bps",
        )
        k3.metric(
            "Implied Round-Trip Factor",
            f"{fwd_res.implied_product_mid:.6f}",
            delta="Parity = 1.000000",
            delta_color="off",
        )
        k4.metric(
            "Break-Even Friction Budget",
            f"{fwd_res.break_even_total_friction_bps:.2f} bps",
            help="Total spread + fee across all 3 legs must stay below this threshold for net profit > 0.",
        )

        if fwd_res.net_profit > 0:
            st.success(
                f"**Arbitrage Opportunity Detected** on `{fwd_res.path_label}`: "
                f"Net profit of **{fwd_res.net_profit:,.2f} {ccy_a}** ({fwd_res.net_return_bps:+.2f} bps) "
                f"exceeds total execution friction."
            )
        else:
            st.info(
                f"**No Executable Arbitrage** on `{fwd_res.path_label}`: "
                f"Market friction reduces the round-trip return to **{fwd_res.net_profit:,.2f} {ccy_a}** "
                f"({fwd_res.net_return_bps:+.2f} bps), consistent with CFA no-arbitrage bounds."
            )

        # Leg-by-leg execution table + Waterfall chart
        c_legs, c_chart = st.columns([1.45, 1])
        with c_legs:
            st.markdown(f"**Execution Breakdown: `{fwd_res.path_label}`**")
            leg_rows = []
            for leg in fwd_res.legs:
                leg_rows.append(
                    {
                        "Leg": f"{leg.step}. {leg.from_ccy} → {leg.to_ccy}",
                        "Mid Rate": round(leg.mid_rate, 6),
                        "Exec Rate": round(leg.effective_rate, 6),
                        "Half Spread (bps)": round(leg.half_spread_bps, 2),
                        "Fee (bps)": round(leg.fee_bps, 2),
                        "In": f"{leg.input_amount:,.2f} {leg.from_ccy}",
                        "Out": f"{leg.output_amount:,.2f} {leg.to_ccy}",
                        "Source": leg.rate_source,
                    }
                )
            st.dataframe(pd.DataFrame(leg_rows), use_container_width=True, hide_index=True)

            st.caption(
                f"Reverse Triangle `{rev_res.path_label}` Net P&L: "
                f"**{rev_res.net_profit:+,.2f} {ccy_a}** ({rev_res.net_return_bps:+.2f} bps) | "
                f"Gross: **{rev_res.gross_profit:+,.2f} {ccy_a}** ({rev_res.gross_return_bps:+.2f} bps)"
            )

        with c_chart:
            st.markdown("**Gross vs. Net Return Attribution (bps)**")
            total_spread_bps = sum(l.half_spread_bps for l in fwd_res.legs)
            total_fee_bps = sum(l.fee_bps for l in fwd_res.legs)
            attr_df = pd.DataFrame(
                {
                    "Component": ["Gross (Mid)", "Spread Drag", "Fee Drag", "Net P&L"],
                    "Basis Points": [
                        fwd_res.gross_return_bps,
                        -total_spread_bps,
                        -total_fee_bps,
                        fwd_res.net_return_bps,
                    ],
                }
            )
            attr_df["Sign"] = attr_df["Basis Points"].apply(lambda x: "Positive" if x >= 0 else "Negative")
            chart_attr = (
                alt.Chart(attr_df)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Component:N", sort=["Gross (Mid)", "Spread Drag", "Fee Drag", "Net P&L"], title=None),
                    y=alt.Y("Basis Points:Q", title="Return / Drag (bps)"),
                    color=alt.Color(
                        "Sign:N",
                        scale=alt.Scale(domain=["Positive", "Negative"], range=["#16a34a", "#dc2626"]),
                        legend=None,
                    ),
                    tooltip=["Component", alt.Tooltip("Basis Points:Q", format="+.2f")],
                )
                .properties(height=250)
            )
            st.altair_chart(chart_attr, use_container_width=True)

        # Live Cross-Rate Matrix
        st.markdown("**Live Cross-Rate Matrix** *(row = Base currency 1.00, column = Quote currency received)*")
        matrix_peers = [ccy_a, ccy_b, ccy_c]
        for peer in ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD"]:
            if peer in spot_result.rates and peer not in matrix_peers:
                matrix_peers.append(peer)
        cross_df = build_cross_rate_matrix(spot_result.rates, matrix_peers[:8])
        st.dataframe(cross_df.style.format("{:.5f}"), use_container_width=True)

        # Live Kraken Quotes Snapshot
        if use_kraken and kraken_quotes:
            with st.expander("View Live Kraken Top-of-Book FX Bid / Ask Quotes"):
                direct_quotes = [q for q in kraken_quotes.values() if not q.is_inverted]
                k_rows = [
                    {
                        "Pair": f"{q.base}/{q.quote}",
                        "Bid": q.bid,
                        "Ask": q.ask,
                        "Mid": round(q.mid, 6),
                        "Last": q.last,
                        "Full Spread (bps)": round(q.spread_bps, 2),
                    }
                    for q in sorted(direct_quotes, key=lambda x: f"{x.base}/{x.quote}")
                ]
                st.dataframe(pd.DataFrame(k_rows), use_container_width=True, hide_index=True)

# ===========================================================================
# TAB 2: REMITTANCE & BANK MARKUP ARBITRAGE CALCULATOR
# ===========================================================================
with tab2:
    st.subheader("Remittance & Bank Markup Arbitrage Calculator")
    st.markdown(
        "Compare the live interbank mid-market rate against typical traditional bank FX markups (3–5% spread + wire fee) "
        "and low-cost fintech transfer tiers to quantify hidden spread leakage."
    )

    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    with r_col1:
        remit_src = st.selectbox(
            "Send Currency",
            options=ordered_codes,
            index=ordered_codes.index("USD") if "USD" in ordered_codes else 0,
            format_func=lambda c: format_currency_option(c, currency_labels),
            key="remit_src",
        )
    with r_col2:
        default_dst = "MXN" if "MXN" in ordered_codes else ("EUR" if "EUR" in ordered_codes else ordered_codes[1])
        remit_dst = st.selectbox(
            "Receive Currency",
            options=ordered_codes,
            index=ordered_codes.index(default_dst),
            format_func=lambda c: format_currency_option(c, currency_labels),
            key="remit_dst",
        )
    with r_col3:
        send_amount = st.number_input(
            f"Amount per Transfer ({remit_src})",
            min_value=50.0,
            max_value=5_000_000.0,
            value=2_500.0,
            step=250.0,
            format="%.2f",
        )
    with r_col4:
        transfers_per_year = st.number_input(
            "Transfers per Year",
            min_value=1,
            max_value=365,
            value=12,
            step=1,
        )

    if remit_src == remit_dst:
        st.warning("Please select two different currencies for remittance comparison.")
    else:
        live_mid = compute_cross_rate(spot_result.rates, remit_src, remit_dst)

        b_col1, b_col2, l_col1, l_col2 = st.columns(4)
        with b_col1:
            bank_spread_pct = st.slider(
                "Traditional Bank FX Spread (%)",
                min_value=1.0,
                max_value=7.0,
                value=4.0,
                step=0.25,
                help="Typical retail banks charge 3%–5% hidden FX spread markup over mid-market.",
            )
        with b_col2:
            bank_wire_fee = st.number_input(
                f"Traditional Bank Wire Fee ({remit_src})",
                min_value=0.0,
                max_value=250.0,
                value=35.0,
                step=5.0,
            )
        with l_col1:
            fintech_spread_pct = st.slider(
                "Low-Cost Tier FX Spread (%)",
                min_value=0.0,
                max_value=2.0,
                value=0.40,
                step=0.05,
            )
        with l_col2:
            fintech_fee = st.number_input(
                f"Low-Cost Tier Transfer Fee ({remit_src})",
                min_value=0.0,
                max_value=100.0,
                value=4.50,
                step=0.50,
            )

        tiers = [
            ("Mid-Market Benchmark (Interbank)", 0.0, 0.0),
            (f"Traditional Bank ({bank_spread_pct:.2f}% + {bank_wire_fee:.0f} fee)", bank_spread_pct, bank_wire_fee),
            ("Traditional Bank High Markup (5.0% + 45 fee)", 5.0, 45.0),
            ("Mid-Tier Digital Remitter (1.20% + 8 fee)", 1.20, 8.0),
            (f"Low-Cost Transfer Tier ({fintech_spread_pct:.2f}% + {fintech_fee:.2f} fee)", fintech_spread_pct, fintech_fee),
        ]

        comparisons = analyze_remittance_tiers(
            mid_rate=live_mid,
            send_amount=send_amount,
            transfers_per_year=int(transfers_per_year),
            tiers=tiers,
            benchmark_bank_index=1,
        )

        bank_row = comparisons[1]
        lowcost_row = comparisons[4]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            f"Live Mid-Market ({remit_src}/{remit_dst})",
            f"{live_mid:,.4f}",
            help=f"Source: {spot_result.provider}",
        )
        m2.metric(
            "Traditional Bank Total Cost",
            f"{bank_row.total_cost_source:,.2f} {remit_src}",
            delta=f"-{bank_row.total_cost_pct:.2f}% drag",
            delta_color="inverse",
        )
        m3.metric(
            "Saved per Transfer (vs. Bank)",
            f"{lowcost_row.saved_per_transfer:,.2f} {remit_src}",
            delta=f"{lowcost_row.total_cost_pct:.2f}% low-cost drag",
        )
        m4.metric(
            "Saved per Year (vs. Bank)",
            f"{lowcost_row.saved_per_year:,.2f} {remit_src}",
            delta=f"{int(transfers_per_year)} transfers/yr",
        )

        remit_df = pd.DataFrame(
            [
                {
                    "Provider Tier": r.tier_name,
                    "FX Spread (%)": round(r.spread_pct, 2),
                    f"Wire Fee ({remit_src})": round(r.wire_fee, 2),
                    "Effective Rate": round(r.effective_rate, 5),
                    f"Recipient Gets ({remit_dst})": round(r.received_dest, 2),
                    f"Hidden FX Cost ({remit_src})": round(r.hidden_fx_cost_source, 2),
                    f"Total Cost ({remit_src})": round(r.total_cost_source, 2),
                    "Total Cost (%)": round(r.total_cost_pct, 2),
                    f"Saved / Transfer ({remit_src})": round(r.saved_per_transfer, 2),
                    f"Saved / Year ({remit_src})": round(r.saved_per_year, 2),
                }
                for r in comparisons
            ]
        )
        st.dataframe(remit_df, use_container_width=True, hide_index=True)

        # Stacked Bar Chart: Explicit Fee vs Hidden FX Markup
        cost_melt = pd.DataFrame(
            [
                {"Tier": r.tier_name, "Cost Type": "Explicit Wire/Transfer Fee", "Amount": r.wire_fee}
                for r in comparisons[1:]
            ]
            + [
                {"Tier": r.tier_name, "Cost Type": "Hidden FX Spread Markup", "Amount": r.hidden_fx_cost_source}
                for r in comparisons[1:]
            ]
        )
        cost_chart = (
            alt.Chart(cost_melt)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                y=alt.Y("Tier:N", sort=None, title=None),
                x=alt.X("Amount:Q", title=f"Total Cost per Transfer ({remit_src})"),
                color=alt.Color(
                    "Cost Type:N",
                    scale=alt.Scale(
                        domain=["Explicit Wire/Transfer Fee", "Hidden FX Spread Markup"],
                        range=["#2563eb", "#f97316"],
                    ),
                ),
                tooltip=["Tier", "Cost Type", alt.Tooltip("Amount:Q", format=",.2f")],
            )
            .properties(height=240, title="Decomposed Remittance Cost: Explicit Wire Fee vs. Hidden FX Spread")
        )
        st.altair_chart(cost_chart, use_container_width=True)

# ===========================================================================
# TAB 3: HISTORICAL FX TREND & ROLLING VOLATILITY
# ===========================================================================
with tab3:
    st.subheader("Historical FX Trend & Rolling Volatility (Frankfurter / ECB)")
    st.markdown(
        "Pulls daily ECB reference rates from `api.frankfurter.app` to track spot trend, moving averages, "
        "and annualized rolling volatility."
    )

    hist_codes = sorted(frankfurter_currencies.keys())
    h_col1, h_col2, h_col3, h_col4 = st.columns(4)
    with h_col1:
        hist_base = st.selectbox(
            "Base Currency",
            options=hist_codes,
            index=hist_codes.index("USD") if "USD" in hist_codes else 0,
            format_func=lambda c: format_currency_option(c, frankfurter_currencies),
            key="hist_base",
        )
    with h_col2:
        hist_target_default = "EUR" if hist_base != "EUR" else "GBP"
        hist_target = st.selectbox(
            "Target Currency",
            options=hist_codes,
            index=hist_codes.index(hist_target_default) if hist_target_default in hist_codes else 1,
            format_func=lambda c: format_currency_option(c, frankfurter_currencies),
            key="hist_target",
        )
    with h_col3:
        lookback_days = st.selectbox(
            "Lookback Window",
            options=[90, 180, 365],
            index=2,
            format_func=lambda d: "1 Year (365d)" if d == 365 else f"{d} Days",
        )
    with h_col4:
        vol_window = st.selectbox(
            "Rolling Volatility Window",
            options=[10, 20, 30, 60],
            index=1,
            format_func=lambda w: f"{w} Trading Days",
        )

    if hist_base == hist_target:
        st.warning("Please select two distinct currencies to load historical FX rates.")
    else:
        try:
            with st.spinner(f"Fetching {lookback_days}-day {hist_base}/{hist_target} history from Frankfurter..."):
                raw_hist_df = cached_history(hist_base, hist_target, int(lookback_days))
                hist_df, summary = compute_historical_analytics(raw_hist_df, vol_window=int(vol_window))

            hk1, hk2, hk3, hk4, hk5 = st.columns(5)
            hk1.metric(
                f"Latest ({hist_base}/{hist_target})",
                f"{summary['latest_rate']:.4f}",
                delta=f"{summary['total_return_pct']:+.2f}% ({lookback_days}d)",
            )
            hk2.metric(
                f"{vol_window}D Rolling Vol (Ann.)",
                f"{summary['latest_roll_vol_pct']:.2f}%",
                delta=f"Period Avg {summary['full_ann_vol_pct']:.2f}%",
                delta_color="off",
            )
            hk3.metric("Period High", f"{summary['period_high']:.4f}")
            hk4.metric("Period Low", f"{summary['period_low']:.4f}")
            hk5.metric(
                "Max Drawdown",
                f"{summary['max_drawdown_pct']:.2f}%",
                delta=f"{summary['range_percentile']:.0f}th pctile",
                delta_color="off",
            )

            # Spot Price + SMAs Chart
            price_long = hist_df[["date", "rate", "sma_short", "sma_long"]].melt(
                id_vars=["date"],
                value_vars=["rate", "sma_short", "sma_long"],
                var_name="Series",
                value_name="Exchange Rate",
            )
            series_labels = {
                "rate": f"Spot ({hist_base}/{hist_target})",
                "sma_short": "20D SMA",
                "sma_long": "50D SMA",
            }
            price_long["Series"] = price_long["Series"].map(series_labels)

            band_chart = (
                alt.Chart(hist_df.dropna(subset=["band_lower", "band_upper"]))
                .mark_area(opacity=0.14, color="#60a5fa")
                .encode(
                    x=alt.X("date:T", title=None),
                    y=alt.Y("band_lower:Q", scale=alt.Scale(zero=False), title=f"{hist_target} per 1 {hist_base}"),
                    y2="band_upper:Q",
                )
            )
            line_chart = (
                alt.Chart(price_long.dropna())
                .mark_line()
                .encode(
                    x=alt.X("date:T", title=None),
                    y=alt.Y("Exchange Rate:Q", scale=alt.Scale(zero=False), title=f"{hist_target} per 1 {hist_base}"),
                    color=alt.Color(
                        "Series:N",
                        scale=alt.Scale(
                            domain=[f"Spot ({hist_base}/{hist_target})", "20D SMA", "50D SMA"],
                            range=["#1d4ed8", "#f59e0b", "#10b981"],
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("date:T", title="Date"),
                        "Series:N",
                        alt.Tooltip("Exchange Rate:Q", format=".5f"),
                    ],
                )
            )
            st.altair_chart(
                (band_chart + line_chart).properties(
                    height=330,
                    title=f"{hist_base}/{hist_target} Spot Rate, Moving Averages & ±2σ Band ({summary['start_date']} to {summary['end_date']})",
                ),
                use_container_width=True,
            )

            # Rolling Annualized Volatility Chart
            vol_df = hist_df.dropna(subset=["rolling_vol_ann_pct"])
            vol_chart = (
                alt.Chart(vol_df)
                .mark_area(line={"color": "#dc2626"}, color="#fca5a5", opacity=0.35)
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("rolling_vol_ann_pct:Q", title="Annualized Volatility (%)"),
                    tooltip=[
                        alt.Tooltip("date:T", title="Date"),
                        alt.Tooltip("rolling_vol_ann_pct:Q", title="Rolling Vol (%)", format=".2f"),
                    ],
                )
                .properties(
                    height=230,
                    title=f"{vol_window}-Day Rolling Annualized Volatility (√252 × σ of Daily Log Returns)",
                )
            )
            st.altair_chart(vol_chart, use_container_width=True)

            with st.expander("Download / Inspect Historical Data Table"):
                st.dataframe(
                    hist_df[["date", "rate", "simple_return_pct", "rolling_vol_ann_pct", "drawdown_pct"]]
                    .sort_values("date", ascending=False)
                    .head(60),
                    use_container_width=True,
                    hide_index=True,
                )
        except FXDataError as exc:
            st.error(f"Could not fetch Frankfurter historical rates: {exc}")
        except ValueError as exc:
            st.warning(str(exc))
