# FX Triangular Arbitrage & Remittance Spread Monitor

A Python + Streamlit quantitative FX monitoring dashboard using **zero-API-key, public market data endpoints** (`open.er-api.com`, `api.frankfurter.app`, and Kraken's public Ticker API) to evaluate triangular no-arbitrage parity, retail remittance markup drag, and 1-year historical currency volatility.

---

## Features

1. **Triangular FX Arbitrage Calculator (`A → B → C → A`)**
   - Pick any three currencies (e.g., `USD`, `EUR`, `GBP`) and inspect the live $N \times N$ cross-rate matrix.
   - Evaluates both forward (`A → B → C → A`) and reverse (`A → C → B → A`) round-trip legs.
   - Overlays **live Kraken top-of-book bid/ask spreads** where available (`EUR/USD`, `GBP/USD`, `EUR/GBP`, `USD/JPY`, `USD/CAD`, `USD/CHF`, `AUD/USD`, `EUR/CAD`, `EUR/CHF`, `EUR/JPY`), with configurable half-spread and execution fee sliders (in basis points) plus a synthetic leg-mispricing shock slider.
   - Displays **Gross vs. Net Arbitrage P&L**, return in bps, and the **Break-Even Friction Budget**.

2. **Remittance & Bank Markup Arbitrage Calculator**
   - Compares the live mid-market rate against typical traditional bank markups (3–5% hidden FX spread + wire fee), mid-tier online remitters, and customizable low-cost transfer tiers.
   - Decomposes total transfer drag into **Explicit Wire Fee** vs. **Hidden FX Spread Cost** and calculates **exact dollars saved per transfer and per year**.

3. **Historical FX Trend & Rolling Volatility (`api.frankfurter.app`)**
   - Fetches 90-day, 180-day, or 1-year daily ECB reference rates from Frankfurter.
   - Plots spot rate with 20D/50D Simple Moving Averages, ±2σ bands, annualized rolling volatility ($\sqrt{252}$), period high/low, and max drawdown.

---

## CFA Cross-Rate & No-Arbitrage Math

### 1. Cross-Rate Construction
Given spot rates anchored to a common numeraire (e.g., `USD`), where $R_{\text{USD} \to X}$ denotes units of currency $X$ per $1\text{ USD}$:

$$S_{A \to B} = \frac{R_{\text{USD} \to B}}{R_{\text{USD} \to A}}$$

For example, if $R_{\text{USD} \to \text{EUR}} = 0.912$ and $R_{\text{USD} \to \text{GBP}} = 0.763$, then selling $1\text{ EUR}$ yields $S_{\text{EUR} \to \text{GBP}} = \frac{0.763}{0.912} \approx 0.8366\text{ GBP}$.

### 2. Triangular No-Arbitrage Parity
In a frictionless, arbitrage-free market, converting $1$ unit of currency $A$ through $B$ and $C$ back to $A$ must equal $1$:

$$\Pi_{\text{mid}} = S_{A \to B} \times S_{B \to C} \times S_{C \to A} = 1.0000$$

If $\Pi_{\text{mid}} \neq 1$, a **gross triangular discrepancy** exists. However, an arbitrageur cannot execute at mid-market prices:
- Selling currency $X$ to receive $Y$ executes at the **bid** price: $S^{\text{bid}}_{X \to Y} = S^{\text{mid}}_{X \to Y} \cdot \left(1 - \frac{h_{XY}}{10{,}000}\right)$, where $h_{XY}$ is the half bid-ask spread in basis points.
- Explicit brokerage/clearing fees $f_{XY}$ (in bps) further scale each leg by $\left(1 - \frac{f_{XY}}{10{,}000}\right)$.

For inverted order-book pairs (e.g., deriving `USD/EUR` from Kraken's `EUR/USD` book):

$$S^{\text{bid}}_{\text{USD} \to \text{EUR}} = \frac{1}{S^{\text{ask}}_{\text{EUR} \to \text{USD}}}, \qquad S^{\text{ask}}_{\text{USD} \to \text{EUR}} = \frac{1}{S^{\text{bid}}_{\text{EUR} \to \text{USD}}}$$

**Executable Net Arbitrage Profit** on starting notional $N_A$ is:

$$\text{Net Profit}_A = N_A \left[\prod_{i=1}^{3} S^{\text{bid}}_i \left(1 - \frac{f_i}{10{,}000}\right) - 1\right]$$

No executable arbitrage exists whenever $\Pi_{\text{net}} \le 1$, i.e., when total bid-ask + fee friction across the three legs exceeds the break-even budget $\left(1 - \Pi_{\text{mid}}^{-1}\right) \times 10{,}000\text{ bps}$.

### 3. Remittance Total Cost Decomposition
For a total outlay $P$ in source currency with explicit wire fee $F$, FX spread markup $m$, and live mid-market rate $M$:
- Net converted principal: $P_{\text{net}} = \max(P - F, 0)$
- Executed rate: $R_{\text{eff}} = M (1 - m)$
- Recipient receives: $P_{\text{net}} \cdot M (1 - m)$
- **Total Cost in Source Currency**: $C_{\text{total}} = F + P_{\text{net}} \cdot m$ (exact sum of explicit wire fee + hidden FX spread).

### 4. Historical Annualized Rolling Volatility
Using daily log returns $r_t = \ln(S_t / S_{t-1})$ over a rolling window of $w$ trading days:

$$\sigma_{\text{ann}, t} = \text{std}(r_{t-w+1 \dots t}) \times \sqrt{252} \times 100\%$$

---

## Public Data Endpoints (No API Keys)

| Feed | Endpoint | Purpose |
| :--- | :--- | :--- |
| ExchangeRate-API (Open) | `https://open.er-api.com/v6/latest/USD` | 160+ live spot mid-rates |
| Frankfurter (ECB) | `https://api.frankfurter.app/latest` & range queries | ECB spot fallback & 1-year daily history |
| Kraken Public Ticker | `https://api.kraken.com/0/public/Ticker` | Live top-of-book fiat bid/ask spreads |

---

## Setup & Usage

```bash
git clone https://github.com/Elchicochico/fx-arbitrage-monitor.git
cd fx-arbitrage-monitor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Run unit tests:

```bash
pytest -v
```
