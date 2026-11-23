from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

# ==============================================================================
# 1. LISTA DE ACTIVOS A ESCANEAR EN 15M (YFINANCE)
# ==============================================================================
ASSETS: List[Tuple[str, Optional[str]]] = [
    ("BTCUSD", "BTC-USD"),
    ("ETHUSD", "ETH-USD"),
    ("SOLANA", "SOL-USD"),
    ("EURUSD", "EURUSD=X"),
    ("GBPUSD", "GBPUSD=X"),
    ("XAUUSD", "GC=F"),
    ("XAGUSD", "SI=F"),
    ("SPY", "SPY"),
    ("QQQ", "QQQ"),
    ("DXY", "DX-Y.NYB"),
    ("APPLE", "AAPL"),
    ("TESLA", "TSLA"),
    ("SPACEX", None),        # No cotiza públicamente en bolsa
    ("META", "META"),
    ("NVIDIA", "NVDA"),
    ("GOOGLE", "GOOGL"),
    ("MICROSOFT", "MSFT"),
]

CONFIG = {
    "period": "10d",
    "interval": "15m",
    "entry_depth": 0.50,            # 50% (CE / Midpoint del FVG)
    "max_wait_candles": 20,         # Máximo 20 velas de 15m esperando retesteo
    "rr_ratio": 2.0,                # TP fijo 1:2
    "min_fvg_pct": 0.0006,          # Hueco mínimo: 0.06%
    "min_displacement_ratio": 0.35, # Hueco >= 35% del cuerpo de v2
    "sl_buffer_pct": 0.0001,        # Margen mínimo detrás de v1
}


@dataclass
class ScanHit:
    label: str
    ticker: str
    state: str              # "ENTRY_ACTIVA" o "ESPERANDO_RETESTEO"
    direction: str
    session: str
    fvg_time: pd.Timestamp
    entry_price: float
    sl_price: float
    tp_price: float
    candles_left: int


# ==============================================================================
# 2. LÓGICA DE SESIÓN Y FVG (IGUAL A LA ESTRATEGIA BASE)
# ==============================================================================
def get_session_ny(ts: pd.Timestamp) -> Optional[str]:
    t = ts.time()
    if pd.Timestamp("02:00").time() <= t <= pd.Timestamp("05:00").time():
        return "LONDON"
    if pd.Timestamp("08:30").time() <= t <= pd.Timestamp("11:30").time():
        return "NY"
    return None


def detect_fvg_at_index(df: pd.DataFrame, idx: int, cfg: dict) -> Optional[dict]:
    if idx < 2 or idx >= len(df):
        return None

    session = get_session_ny(df.index[idx])
    if session is None:
        return None

    h1, l1 = float(df["High"].iloc[idx - 2]), float(df["Low"].iloc[idx - 2])
    o2, c2 = float(df["Open"].iloc[idx - 1]), float(df["Close"].iloc[idx - 1])
    h3, l3 = float(df["High"].iloc[idx]), float(df["Low"].iloc[idx])

    body_v2 = abs(c2 - o2)
    if body_v2 == 0:
        return None

    depth = cfg["entry_depth"]
    rr = cfg["rr_ratio"]
    buf = cfg["sl_buffer_pct"]

    if l3 > h1 and c2 > o2:
        gap = l3 - h1
        if (gap / c2 >= cfg["min_fvg_pct"]) and (gap / body_v2 >= cfg["min_displacement_ratio"]):
            entry = l3 - (gap * depth)
            sl = l1 * (1.0 - buf)
            risk = entry - sl
            if risk <= 0:
                return None
            return {
                "direction": "LONG",
                "session": session,
                "fvg_time": df.index[idx],
                "entry": entry,
                "sl": sl,
                "tp": entry + (risk * rr),
            }

    if h3 < l1 and c2 < o2:
        gap = l1 - h3
        if (gap / c2 >= cfg["min_fvg_pct"]) and (gap / body_v2 >= cfg["min_displacement_ratio"]):
            entry = h3 + (gap * depth)
            sl = h1 * (1.0 + buf)
            risk = sl - entry
            if risk <= 0:
                return None
            return {
                "direction": "SHORT",
                "session": session,
                "fvg_time": df.index[idx],
                "entry": entry,
                "sl": sl,
                "tp": entry - (risk * rr),
            }

    return None


def load_15m_data(ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty or len(df) < 3:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df.tz_convert("America/New_York")
        return df[df.index + pd.Timedelta(minutes=15) <= pd.Timestamp.now(tz="America/New_York")][["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return None


def inspect_symbol(label: str, ticker: str, df: pd.DataFrame, cfg: dict) -> Optional[ScanHit]:
    n = len(df)
    last_idx = n - 1
    max_wait = cfg["max_wait_candles"]
    start_search = max(2, last_idx - max_wait)

    active_entry: Optional[ScanHit] = None
    waiting_retest: Optional[ScanHit] = None

    for i in range(start_search, last_idx + 1):
        fvg = detect_fvg_at_index(df, i, cfg)
        if fvg is None:
            continue

        candles_elapsed = last_idx - i
        if candles_elapsed > max_wait:
            continue

        candles_left = max_wait - candles_elapsed
        direction = fvg["direction"]
        entry = fvg["entry"]
        sl = fvg["sl"]
        tp = fvg["tp"]

        filled_at: Optional[int] = None
        invalidated_before_fill = False

        for j in range(i + 1, last_idx + 1):
            lj = float(df["Low"].iloc[j])
            hj = float(df["High"].iloc[j])

            if filled_at is None:
                if direction == "LONG":
                    if lj <= sl:
                        invalidated_before_fill = True
                        break
                    if lj <= entry:
                        filled_at = j
                else:
                    if hj >= sl:
                        invalidated_before_fill = True
                        break
                    if hj >= entry:
                        filled_at = j

            if filled_at is not None:
                if direction == "LONG":
                    hit_sl = lj <= sl
                    hit_tp = hj >= tp
                else:
                    hit_sl = hj >= sl
                    hit_tp = lj <= tp

                if hit_sl or hit_tp:
                    filled_at = -999
                    break

        if invalidated_before_fill or filled_at == -999:
            continue

        if filled_at is not None:
            active_entry = ScanHit(
                label=label,
                ticker=ticker,
                state="ENTRY_ACTIVA",
                direction=direction,
                session=fvg["session"],
                fvg_time=fvg["fvg_time"],
                entry_price=entry,
                sl_price=sl,
                tp_price=tp,
                candles_left=candles_left,
            )
        else:
            waiting_retest = ScanHit(
                label=label,
                ticker=ticker,
                state="ESPERANDO_RETESTEO",
                direction=direction,
                session=fvg["session"],
                fvg_time=fvg["fvg_time"],
                entry_price=entry,
                sl_price=sl,
                tp_price=tp,
                candles_left=candles_left,
            )

    return active_entry if active_entry is not None else waiting_retest


# ==============================================================================
# 3. EJECUCIÓN DEL ESCÁNER MÚLTIPLE
# ==============================================================================
def run_scanner(assets: List[Tuple[str, Optional[str]]], cfg: dict):
    entries_activas: List[ScanHit] = []
    pendientes: List[ScanHit] = []
    sin_setup: List[str] = []
    sin_datos: List[str] = []

    print("=" * 78)
    print(f"🔎 ESCÁNER MULTI-ACTIVO 15M (FVG 50% CE | 1:2 | LONDRES/NY) — {len(assets)} activos")
    print("=" * 78)

    for label, ticker in assets:
        if not ticker:
            sin_datos.append(f"{label} (sin ticker público)")
            continue

        df = load_15m_data(ticker, cfg["period"], cfg["interval"])
        if df is None or len(df) < 3:
            sin_datos.append(f"{label} [{ticker}]")
            continue

        hit = inspect_symbol(label, ticker, df, cfg)
        if hit is None:
            sin_setup.append(f"{label} [{ticker}]")
        elif hit.state == "ENTRY_ACTIVA":
            entries_activas.append(hit)
        else:
            pendientes.append(hit)

    print("\n▶ ACTIVOS CON ENTRY ACTIVA (LISTOS PARA PASAR AL COMMAND CENTER):")
    print("-" * 78)
    if not entries_activas:
        print("  Ninguno tiene ENTRY ACTIVA en este momento.")
    else:
        for h in entries_activas:
            print(
                f"  • {h.label:<10} (ticker: {h.ticker:<10}) | {h.direction:<5} | {h.session:<6} | "
                f"Entry: {h.entry_price:.4f} | SL: {h.sl_price:.4f} | TP: {h.tp_price:.4f}"
            )

    print("\n⏳ ACTIVOS ESPERANDO RETESTEO AL 50% (PENDIENTES):")
    print("-" * 78)
    if not pendientes:
        print("  Ninguno pendiente de retesteo.")
    else:
        for h in pendientes:
            print(
                f"  • {h.label:<10} (ticker: {h.ticker:<10}) | {h.direction:<5} | {h.session:<6} | "
                f"Entry: {h.entry_price:.4f} | SL: {h.sl_price:.4f} | TP: {h.tp_price:.4f} | "
                f"Quedan {h.candles_left} velas"
            )

    print("\n— SIN SETUP ACTIVO:")
    print("  " + (", ".join(sin_setup) if sin_setup else "Ninguno"))

    if sin_datos:
        print("\n⚠️ SIN DATOS / NO DISPONIBLE:")
        print("  " + ", ".join(sin_datos))

    print("=" * 78)


if __name__ == "__main__":
    run_scanner(ASSETS, CONFIG)
