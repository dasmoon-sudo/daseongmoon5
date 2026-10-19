# app.py — One-Page Evidence Dashboard (clean, with recommended levels)
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(layout="wide")  # No title

# ================= Sidebar: ticker only =================
st.sidebar.header("Input")
ticker = st.sidebar.text_input("Enter ticker (e.g., AAPL, MSFT, TSLA)").upper().strip()

if not ticker:
    st.info("Enter a **ticker** on the left. After you enter it, the full report will appear on one page.")
    st.stop()

# ================= Safe loaders =================
def _flat(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance 새 버전은 종목 1개여도 열이 2단(MultiIndex) → 1단으로 펴기."""
    if isinstance(df, pd.DataFrame) and isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    if isinstance(df, pd.DataFrame) and "Close" in df.columns:
        df = df.dropna(subset=["Close"])
    return df

@st.cache_data(show_spinner=False)
def dl_price_1y(tk: str) -> pd.DataFrame:
    for i in range(3):
        try:
            df = yf.download(
                tk,
                period="1y",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            df = _flat(df)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception:
            time.sleep(1.0 * (i + 1))
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def dl_price_10d_1d(tk: str) -> pd.DataFrame:
    try:
        df = yf.download(
            tk,
            period="10d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        return _flat(df) if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_blocks(tk: str):
    t = yf.Ticker(tk)
    info = {}
    is_a = bs_a = cf_a = pd.DataFrame()
    qearn = pd.DataFrame()

    try:
        info = t.info if hasattr(t, "info") else {}
    except Exception:
        pass
    try:
        is_a = t.financials
    except Exception:
        pass
    try:
        bs_a = t.balance_sheet
    except Exception:
        pass
    try:
        cf_a = t.cashflow
    except Exception:
        pass
    try:
        qearn = t.quarterly_earnings
    except Exception:
        pass

    return info, is_a, bs_a, cf_a, qearn

def last_row(df: pd.DataFrame, row: str):
    try:
        s = df.loc[row]
        if isinstance(s, pd.Series) and len(s) > 0:
            return float(s.iloc[0])
    except Exception:
        pass
    return None

# ================= Load =================
price = dl_price_1y(ticker)
p5 = dl_price_10d_1d(ticker)
info, is_a, bs_a, cf_a, qearn = load_blocks(ticker)

# Latest price
last_px = None
if not price.empty and "Close" in price.columns:
    last_px = float(price["Close"].iloc[-1])
else:
    try:
        last_px = (yf.Ticker(ticker).fast_info or {}).get("last_price") or info.get("currentPrice")
    except Exception:
        last_px = info.get("currentPrice")

if not last_px:
    st.error("Unable to load price data. Please verify the ticker.")
    st.stop()

# Info
mcap = info.get("marketCap")
shares = info.get("sharesOutstanding")
pb = info.get("priceToBook")
pe_ttm = info.get("trailingPE")
pe_fwd = info.get("forwardPE")
beta = info.get("beta") or info.get("beta3Year") or 1.0

# Sector / exchange display
sector_raw = info.get("sector") or "-"
exchange_raw = info.get("exchange") or info.get("fullExchangeName") or "-"
exchange_map = {
    "NMS": "NASDAQ", "NMSR": "NASDAQ", "NGM": "NASDAQ", "BATS": "BATS",
    "NYQ": "NYSE", "NYA": "NYSE", "PCX": "NYSE Arca", "ASE": "AMEX",
    "KSC": "KRX", "KOE": "KRX",
}
exchange_name = exchange_map.get(str(exchange_raw).upper(), str(exchange_raw))
sector_disp = str(sector_raw)

# ================= Indicators =================
def sma_series(df: pd.DataFrame, n: int) -> pd.Series:
    if df.empty or "Close" not in df:
        return pd.Series(dtype=float)
    return df["Close"].rolling(n).mean()

def sma_last(df: pd.DataFrame, n: int):
    s = sma_series(df, n).dropna()
    return float(s.iloc[-1]) if not s.empty else None

def sma_slope_pct(df: pd.DataFrame, n: int, lookback: int = 20):
    s = sma_series(df, n).dropna()
    if len(s) <= lookback:
        return None
    a, b = float(s.iloc[-1]), float(s.iloc[-1 - lookback])
    if b == 0:
        return None
    return (a / b - 1) * 100

def bbands(df: pd.DataFrame, n: int = 20, k: float = 2.0):
    if df.empty or "Close" not in df:
        return None, None, None, None
    c = df["Close"]
    m = c.rolling(n).mean()
    s = c.rolling(n).std(ddof=0)
    if m.dropna().empty or s.dropna().empty:
        return None, None, None, None
    up = m + k * s
    lo = m - k * s
    bw = (up - lo) / m
    return float(up.dropna().iloc[-1]), float(m.dropna().iloc[-1]), float(lo.dropna().iloc[-1]), float(bw.dropna().iloc[-1])

def atr(df: pd.DataFrame, n: int = 14):
    if df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
        return None
    hi, lo, cl = df["High"], df["Low"], df["Close"]
    pc = cl.shift(1)
    tr = pd.concat([(hi - lo).abs(), (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    a = tr.rolling(n, min_periods=max(2, n // 2)).mean().dropna()
    return float(a.iloc[-1]) if not a.empty else None

def realized_vol_pct(df: pd.DataFrame, window: int = 20):
    if df.empty or "Close" not in df:
        return None
    r = df["Close"].pct_change().dropna()
    if len(r) < 5:
        return None
    return float(r.tail(window).std() * 100)

def parkinson_vol_pct(df: pd.DataFrame, window: int = 20):
    if df.empty or not {"High", "Low"}.issubset(df.columns):
        return None
    hl = np.log(df["High"] / df["Low"]).dropna()
    if len(hl) < 5:
        return None
    var = (hl.tail(window) ** 2).mean() / (4 * np.log(2))
    return float(np.sqrt(max(0.0, var)) * 100)

def rsi14(df: pd.DataFrame):
    if df.empty or "Close" not in df:
        return None
    delta = df["Close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / down
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.dropna()
    return float(rsi.iloc[-1]) if not rsi.empty else None

ATR = atr(price, 14)
UB, MB, LB, BW = bbands(price, 20, 2.0)
SMA20 = sma_last(price, 20)
SMA50 = sma_last(price, 50)
SMA200 = sma_last(price, 200)
SMA252 = sma_last(price, 252)
SLOPE50 = sma_slope_pct(price, 50, 20)
SLOPE200 = sma_slope_pct(price, 200, 20)
SLOPE252 = sma_slope_pct(price, 252, 20)
RSI = rsi14(price)

# 52-week high/low (up to last 252 trading days)
high_52w = low_52w = None
if not price.empty and "Close" in price.columns:
    closes = price["Close"].dropna()
    window = min(len(closes), 252)
    lastN = closes.iloc[-window:] if window > 0 else closes
    high_52w = float(lastN.max()) if not lastN.empty else None
    low_52w = float(lastN.min()) if not lastN.empty else None

dist_to_high = (high_52w / last_px - 1) * 100 if (high_52w and last_px) else None
dist_to_low = (1 - (low_52w / last_px)) * 100 if (low_52w and last_px) else None

# Last 5 trading days (always 1D) — avoid double-multiplying percent!
hi5 = lo5 = chg5_frac = None
if isinstance(p5, pd.DataFrame) and len(p5) >= 5:
    last5 = p5.tail(5)
    hi5 = float(last5["High"].max())
    lo5 = float(last5["Low"].min())
    chg5_frac = float(last5["Close"].iloc[-1] / last5["Close"].iloc[0] - 1)  # fraction

# ================= Fundamentals & Auto DCF =================
rev = last_row(is_a, "Total Revenue")
ni = last_row(is_a, "Net Income Common Stockholders") or last_row(is_a, "Net Income")

# FCF: prefer Free Cash Flow; else CFO - CapEx
fcf = last_row(cf_a, "Free Cash Flow")
if fcf is None:
    cfo = last_row(cf_a, "Total Cash From Operating Activities")
    capex = last_row(cf_a, "Capital Expenditures")
    if cfo is not None and capex is not None:
        fcf = cfo - abs(capex)

# 3Y Revenue CAGR
rev_cagr_3y = None
try:
    cols = list(is_a.columns)
    if len(cols) >= 4:
        r_now = float(is_a.loc["Total Revenue", cols[0]])
        r_3y = float(is_a.loc["Total Revenue", cols[3]])
        if r_now > 0 and r_3y > 0:
            rev_cagr_3y = (r_now / r_3y) ** (1 / 3) - 1
except Exception:
    pass

def clamp(x, lo, hi):
    if x is None:
        return None
    return max(lo, min(hi, x))

risk_free = 0.04
mkt_prem = 0.055
disc_r = clamp(risk_free + (beta or 1.0) * mkt_prem, 0.06, 0.15)  # 6–15%
grow_g = clamp(rev_cagr_3y if rev_cagr_3y is not None else 0.08, 0.00, 0.25)
term_g = 0.02

def dcf_value(fcf0, growth, discount, terminal_g, years=10):
    base = fcf0 if (fcf0 and fcf0 > 0) else (rev * 0.05 if rev else 1_000_000.0)
    f = base
    pv = 0.0
    for t in range(1, years + 1):
        f *= (1 + growth)
        pv += f / ((1 + discount) ** t)
    tv = (f * (1 + terminal_g)) / max(1e-9, (discount - terminal_g))
    pv += tv / ((1 + discount) ** years)
    return pv

dcf_total = dcf_value(fcf, grow_g, disc_r, term_g, years=10)
dcf_ps = (dcf_total / shares) if shares else None
gap_pct = (dcf_ps / last_px - 1) * 100 if (dcf_ps and last_px) else None

# ================= "Tomorrow" scenario (dynamic & asymmetric) =================
def pct_to_level(px, level):
    if px is None or level is None or px == 0:
        return None
    return (level / px - 1) * 100

atr_pct = (ATR / last_px * 100) if (last_px and ATR) else None
rv_pct = realized_vol_pct(price, 20)
hk_pct = parkinson_vol_pct(price, 20)

candidates = [x for x in [atr_pct, rv_pct, hk_pct] if x is not None]
base_vol = max(candidates) if candidates else 2.0
base_vol = float(np.clip(base_vol, 0.5, 12.0))

UB, MB, LB, BW = bbands(price, 20, 2.0)  # ensure
up_cap = pct_to_level(last_px, UB)
down_cap = pct_to_level(last_px, LB)

up_cap = up_cap if (up_cap is not None and up_cap > 0) else None
down_cap = -down_cap if (down_cap is not None and down_cap < 0) else None

up_pct = min(base_vol, up_cap) if up_cap is not None else base_vol
down_pct = min(base_vol, down_cap) if down_cap is not None else base_vol

# ================= Recommended levels (Day / Swing / Long-term) =================
def clamp_pct(x, lo=0.3, hi=20.0):
    if x is None:
        return None
    return float(np.clip(x, lo, hi))

intraday_pct = clamp_pct(base_vol * 0.5)
intraday_abs = last_px * intraday_pct / 100

day_entry = last_px - intraday_abs * 0.5
day_tp = last_px + intraday_abs * 0.5
day_sl = max(0.01, last_px - intraday_abs)

fallback_pull = last_px * 0.01
swing_entry = max([v for v in [LB, SMA20] if v is not None], default=last_px - fallback_pull)
swing_tp = UB if UB else last_px * (1 + clamp_pct(base_vol, 0.6, 12) / 100)
swing_sl = max(0.01, swing_entry - (ATR if ATR else last_px * 0.01) * 1.5)

lt_comment = None
if dcf_ps and last_px:
    gap = (dcf_ps / last_px - 1) * 100
    if gap >= 20:
        lt_comment = f"Long-term: undervalued vs. intrinsic value (gap {gap:,.1f}%) → favors DCA / scaling in"
    elif gap <= -20:
        lt_comment = f"Long-term: overvalued vs. intrinsic value (gap {gap:,.1f}%) → consider trimming / waiting"
    else:
        lt_comment = f"Long-term: near intrinsic value (gap {gap:,.1f}%) → neutral"

# ================= Trend commentary =================
def slope_dir(val, tol=0.4):
    if val is None:
        return "Neutral"
    if val > tol:
        return "Up"
    if val < -tol:
        return "Down"
    return "Neutral"

trend_lines = []
if all(v is not None for v in [UB, MB, LB, BW]):
    pos = "Upper" if last_px > MB else ("Lower" if last_px < MB else "Middle")
    width = "Widening" if BW >= 0.10 else ("Contracting" if BW <= 0.03 else "Normal")
    trend_lines.append(f"Bollinger: {pos} · Bandwidth {BW*100:,.1f}% ({width})")

s20 = sma_slope_pct(price, 20, 20)
if SMA20 is not None:
    trend_lines.append(f"SMA(20): Price {'above' if last_px >= SMA20 else 'below'} · Slope {slope_dir(s20)}")
if SMA50 is not None:
    trend_lines.append(f"SMA(50): Price {'above' if last_px >= SMA50 else 'below'} · Slope {slope_dir(SLOPE50)} ({'' if SLOPE50 is None else f'{SLOPE50:,.1f}%/20d'})")
if SMA200 is not None:
    trend_lines.append(f"SMA(200): Price {'above' if last_px >= SMA200 else 'below'} · Slope {slope_dir(SLOPE200)} ({'' if SLOPE200 is None else f'{SLOPE200:,.1f}%/20d'})")
if SMA252 is not None:
    trend_lines.append(f"SMA(252): Price {'above' if last_px >= SMA252 else 'below'} · Slope {slope_dir(SLOPE252)} ({'' if SLOPE252 is None else f'{SLOPE252:,.1f}%/20d'})")
if (high_52w is not None) and (low_52w is not None):
    trend_lines.append(f"52w: {(-dist_to_high):,.1f}% below high · {dist_to_low:,.1f}% above low")
if RSI is not None:
    obos = "Overbought(>70)" if RSI >= 70 else ("Oversold(<30)" if RSI <= 30 else "Neutral")
    trend_lines.append(f"RSI(14): {RSI:,.1f} ({obos})")

# ================= Top metrics =================
def fnum(x, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x*100:,.2f}%" if pct else f"{x:,.2f}"

c1, c2, c3, c4 = st.columns([1, 1, 1.2, 1.4])
c1.markdown(f"**Last Price**<br>{last_px:,.2f}", unsafe_allow_html=True)
c2.markdown(f"**Tomorrow Range**<br>▲{up_pct:,.1f}% / ▼{down_pct:,.1f}%", unsafe_allow_html=True)
c3.markdown(
    f"**5D Range (1D)**<br>High {('-' if hi5 is None else f'{hi5:,.2f}')} / Low {('-' if lo5 is None else f'{lo5:,.2f}')}",
    unsafe_allow_html=True,
)
c4.markdown(f"**Sector / Exchange**<br>{sector_disp} / {exchange_name}", unsafe_allow_html=True)

st.divider()

# ================= Recommended levels section =================
st.subheader("🎯 Recommended Entry / Take-Profit / Stop (Day / Swing)")
colA, colB = st.columns(2)

with colA:
    st.markdown("**Day Trading**")
    st.markdown(
        f"- Entry: **{fnum(day_entry)}**\n"
        f"- Take profit: **{fnum(day_tp)}** _(≈ +{fnum((day_tp/last_px-1), True)} vs last)_\n"
        f"- Stop loss: **{fnum(day_sl)}** _(≈ {fnum((day_sl/last_px-1), True)} vs last)_\n"
        f"- Basis: volatility (ATR / realized vol / HL) → range {fnum(intraday_pct, True)}"
    )

with colB:
    st.markdown("**Swing Trading**")
    st.markdown(
        f"- Entry: **{fnum(swing_entry)}** _(pullback near LB / SMA20)_\n"
        f"- Take profit: **{fnum(swing_tp)}** _(upper band or volatility-based)_\n"
        f"- Stop loss: **{fnum(swing_sl)}** _(~1.5×ATR below entry)_"
    )

if lt_comment:
    st.caption("**Long-term view** · " + lt_comment)

# ================= Full report =================
report = []
report.append(f"- **Ticker**: {ticker}")
report.append(f"- **Price / Scenario**: last {last_px:,.2f} · tomorrow ▲{fnum(up_pct)} / ▼{fnum(down_pct)} (percent)")
if (hi5 is not None) and (lo5 is not None) and (chg5_frac is not None):
    report.append(f"- **Last 5 trading days**: high {hi5:,.2f} / low {lo5:,.2f} · 5D return {fnum(chg5_frac, True)}")
report.append(f"- **Day**: entry {fnum(day_entry)}, TP {fnum(day_tp)}, SL {fnum(day_sl)}")
report.append(f"- **Swing**: entry {fnum(swing_entry)}, TP {fnum(swing_tp)}, SL {fnum(swing_sl)}")
report.append("- **Trend summary**:")
for ln in trend_lines:
    report.append(f"  • {ln}")

rev_cagr_txt = (rev_cagr_3y * 100 if rev_cagr_3y is not None else 8.0)
report.append("- **Value / Growth (auto assumptions)**:")
report.append(f"  • Discount rate ≈ {disc_r*100:,.1f}% (risk-free 4% + beta×5.5% clamped)")
report.append(f"  • Growth rate ≈ {rev_cagr_txt:,.1f}% (3Y revenue CAGR; clamped 0–25%)")
report.append(f"  • Terminal growth 2.0% · Intrinsic / share {fnum(dcf_ps)} · Gap {fnum(gap_pct, True)}")

ps_val = (mcap / rev) if (mcap and rev and rev != 0) else None
report.append("- **Valuation**:")
report.append(f"  • P/E(TTM) {fnum(pe_ttm)} · Fwd P/E {fnum(pe_fwd)} · P/B {fnum(pb)} · P/S {fnum(ps_val)}")

st.subheader("📄 Full Report")
st.markdown("\n".join(report))

# ================= TradingView widget =================
def guess_exchange_prefix(info_dict):
    exch = (info_dict.get("exchange") or info_dict.get("fullExchangeName") or "").lower()
    if "nasdaq" in exch or "nms" in exch:
        return "NASDAQ"
    if "nyse" in exch or "nyq" in exch or "nya" in exch:
        return "NYSE"
    if "korea" in exch or "kosdaq" in exch or "kse" in exch or "krx" in exch:
        return "KRX"
    if "amex" in exch or "ase" in exch:
        return "AMEX"
    return "NASDAQ"

ex_prefix = guess_exchange_prefix(info)

st.components.v1.html(
    f"""
    <div class="tradingview-widget-container"><div id="tvchart"></div></div>
    <script src="https://s3.tradingview.com/tv.js"></script>
    <script>
      new TradingView.widget({{
        "container_id": "tvchart",
        "symbol": "{ex_prefix}:{ticker}",
        "interval": "D",
        "timezone": "Etc/UTC",
        "theme": "light",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "hide_top_toolbar": false,
        "allow_symbol_change": true,
        "studies": [],
        "height": 420,
        "width": "100%"
      }});
    </script>
    """,
    height=420,
)

# ================= Indicator help =================
with st.expander("📚 Indicators & How to Use", expanded=False):
    st.markdown(
        """
- **ATR (Average True Range)**: short-term volatility. Often used for day-trading stop sizing (e.g., 1.0–1.5×ATR stop).
- **Bollinger Bands (20, 2)**: mean ± 2σ. Upper band can indicate overextension; lower band can indicate oversold. Bandwidth contraction→expansion can precede trend moves.
- **SMA 20/50/200/252**: trend gauges. Use price vs SMA and slope direction for trend bias. 252 ≈ 52-week line.
- **RSI(14)**: commonly 70+ overbought, 30- oversold.
- **Realized Vol / Parkinson Vol**: recent “felt” volatility; used for short-horizon range estimates.
- **Auto DCF**: discount = 4% + β×5.5% (clamped 6–15%), growth = 3Y revenue CAGR (clamped 0–25%), terminal g = 2%.
        """
    )

# ================= Download report =================
notes = "\n\n*For reference only (not investment advice). Data source: Yahoo Finance (yfinance).*"
st.download_button(
    "📥 Download Full Report (.md)",
    "\n".join(report) + notes,
    file_name=f"{ticker}_full_report.md",
)
