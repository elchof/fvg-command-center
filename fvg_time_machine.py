import glob
import io
import zipfile
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ==============================================================================
# 1. CONFIGURACIÓN BASE (FVG 15M + 50% CE + 1:2 + LONDRES/NY)
# ==============================================================================
CONFIG = {
    "default_ticker": "NQ=F",
    "live_period": "10d",
    "interval_15m": "15m",
    "zip_pattern": "HISTDATA_COM_ASCII_*_M1*.zip",
    "resample_rule": "15min",
    "train_ratio": 0.70,            # 70% Train | 30% Unseen
    "entry_depth": 0.50,            # Entrada al 50% (CE / Midpoint del FVG)
    "max_wait_candles": 20,         # Máximo 20 velas de 15m para retesteo
    "rr_ratio": 2.0,                # TP fijo 1:2
    "min_fvg_pct": 0.0006,          # Hueco mínimo lógico: 0.06%
    "min_displacement_ratio": 0.35, # Hueco >= 35% del cuerpo de v2
    "sl_buffer_pct": 0.0001,        # Margen mínimo detrás de v1
    "cost_per_trade_r": 0.04,       # Coste (comisión + slippage) en R
    "initial_capital": 10000.0,
    "risk_per_trade_pct": 0.01,
    "mc_simulations": 10000,
    "mc_horizon_trades": 100,
}


@dataclass
class Trade:
    symbol: str
    split: str
    direction: str
    fvg_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    sl_price: float
    tp_price: float
    outcome: str
    r_gross: float
    r_net: float
    candles_waited: int
    duration_candles: int


@dataclass
class SetupStatus:
    state: str              # "ENTRY_ACTIVA" o "ESPERANDO_RETESTEO"
    direction: str
    session: str
    fvg_time: pd.Timestamp
    entry_price: float
    sl_price: float
    tp_price: float
    candles_elapsed: int
    candles_left: int
    expires_at: pd.Timestamp
    detail: str


# ==============================================================================
# 2. UTILIDADES DE SESIÓN Y DETECCIÓN DE FVG
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


# ==============================================================================
# 3. DATOS EN VIVO (YFINANCE) Y DETECCIÓN DE ENTRY ACTUAL
# ==============================================================================
def load_live_15m(ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df.tz_convert("America/New_York")
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return None


def check_current_setup(df_live: pd.DataFrame, cfg: dict) -> Optional[SetupStatus]:
    n = len(df_live)
    last_idx = n - 1
    max_wait = cfg["max_wait_candles"]
    bar_delta = df_live.index[-1] - df_live.index[-2] if n >= 2 else pd.Timedelta(minutes=15)

    start_search = max(2, last_idx - max_wait)
    active_entry: Optional[SetupStatus] = None
    waiting_retest: Optional[SetupStatus] = None

    for i in range(start_search, last_idx + 1):
        fvg = detect_fvg_at_index(df_live, i, cfg)
        if fvg is None:
            continue

        candles_elapsed = last_idx - i
        if candles_elapsed > max_wait:
            continue

        candles_left = max_wait - candles_elapsed
        expires_at = df_live.index[i] + (bar_delta * max_wait)
        direction = fvg["direction"]
        entry = fvg["entry"]
        sl = fvg["sl"]
        tp = fvg["tp"]

        filled_at: Optional[int] = None
        invalidated_before_fill = False

        for j in range(i + 1, last_idx + 1):
            lj = float(df_live["Low"].iloc[j])
            hj = float(df_live["High"].iloc[j])

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
            active_entry = SetupStatus(
                state="ENTRY_ACTIVA",
                direction=direction,
                session=fvg["session"],
                fvg_time=fvg["fvg_time"],
                entry_price=round(entry, 2),
                sl_price=round(sl, 2),
                tp_price=round(tp, 2),
                candles_elapsed=candles_elapsed,
                candles_left=candles_left,
                expires_at=expires_at,
                detail="El precio ya tocó el 50% (CE) del FVG y la operación está abierta."
            )
        else:
            waiting_retest = SetupStatus(
                state="ESPERANDO_RETESTEO",
                direction=direction,
                session=fvg["session"],
                fvg_time=fvg["fvg_time"],
                entry_price=round(entry, 2),
                sl_price=round(sl, 2),
                tp_price=round(tp, 2),
                candles_elapsed=candles_elapsed,
                candles_left=candles_left,
                expires_at=expires_at,
                detail=f"Orden límite pendiente en el 50% (CE). Quedan {candles_left} velas de 15m."
            )

    return active_entry if active_entry is not None else waiting_retest


# ==============================================================================
# 4. MÁQUINA DEL TIEMPO (CARGA HISTDATA .ZIP OPCIONAL + BACKTEST 70/30)
# ==============================================================================
def find_matching_zips(symbol: str, pattern: str) -> List[str]:
    all_zips = sorted(glob.glob(pattern))
    if not all_zips:
        return []

    # Limpia prefijos/sufijos típicos (ej. "NQ=F" -> "NQ", "EURUSD=X" -> "EURUSD")
    clean_sym = symbol.upper().replace("=F", "").replace("=X", "").replace("-", "").replace("/", "").strip()
    if clean_sym:
        matched = [z for z in all_zips if clean_sym in z.upper()]
        if matched:
            return matched

    # Si solo hay un grupo de zips en la carpeta para el activo que estás probando ahora, los usa
    return all_zips


def load_histdata_zips_to_15m(zip_files: List[str], rule: str) -> Optional[pd.DataFrame]:
    if not zip_files:
        return None

    frames = []
    for zpath in zip_files:
        with zipfile.ZipFile(zpath, "r") as zf:
            for member in zf.namelist():
                if member.lower().endswith((".csv", ".txt")):
                    text = zf.read(member).decode("utf-8", errors="ignore")
                    if not text.strip():
                        continue
                    sep = ";" if ";" in text.splitlines()[0] else ","
                    ncols = len(text.splitlines()[0].split(sep))
                    df_m1 = pd.read_csv(
                        io.StringIO(text),
                        sep=sep,
                        header=None,
                        names=["DateTimeStr", "Open", "High", "Low", "Close", "Volume"][:ncols],
                    )
                    frames.append(df_m1)

    if not frames:
        return None

    df_all_m1 = pd.concat(frames, ignore_index=True)
    df_all_m1["Datetime"] = pd.to_datetime(
        df_all_m1["DateTimeStr"].astype(str).str.strip(), format="%Y%m%d %H%M%S", errors="coerce"
    )
    for col in ["Open", "High", "Low", "Close"]:
        df_all_m1[col] = pd.to_numeric(df_all_m1[col], errors="coerce")

    df_all_m1 = df_all_m1.dropna(subset=["Datetime", "Open", "High", "Low", "Close"])
    df_all_m1 = df_all_m1.set_index("Datetime").sort_index()
    df_all_m1 = df_all_m1[~df_all_m1.index.duplicated(keep="first")]
    df_all_m1.index = df_all_m1.index.tz_localize("Etc/GMT+5").tz_convert("America/New_York")

    return df_all_m1.resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    }).dropna()


def run_backtest_segment(df: pd.DataFrame, symbol: str, split_label: str, cfg: dict) -> List[Trade]:
    highs = df["High"].values
    lows = df["Low"].values
    opens = df["Open"].values
    closes = df["Close"].values
    times = df.index

    trades: List[Trade] = []
    n = len(df)
    i = 2

    while i < n - 1:
        v3_time = times[i]
        if get_session_ny(v3_time) is None:
            i += 1
            continue

        h1, l1 = highs[i - 2], lows[i - 2]
        o2, c2 = opens[i - 1], closes[i - 1]
        h3, l3 = highs[i], lows[i]
        body_v2 = abs(c2 - o2)

        if body_v2 == 0:
            i += 1
            continue

        direction = None
        entry_price = 0.0
        sl_price = 0.0

        if l3 > h1 and c2 > o2:
            gap = l3 - h1
            if (gap / c2 >= cfg["min_fvg_pct"]) and (gap / body_v2 >= cfg["min_displacement_ratio"]):
                direction = "LONG"
                entry_price = l3 - (gap * cfg["entry_depth"])
                sl_price = l1 * (1.0 - cfg["sl_buffer_pct"])
        elif h3 < l1 and c2 < o2:
            gap = l1 - h3
            if (gap / c2 >= cfg["min_fvg_pct"]) and (gap / body_v2 >= cfg["min_displacement_ratio"]):
                direction = "SHORT"
                entry_price = h3 + (gap * cfg["entry_depth"])
                sl_price = h1 * (1.0 + cfg["sl_buffer_pct"])

        if direction is None:
            i += 1
            continue

        risk = abs(entry_price - sl_price)
        if risk <= 0:
            i += 1
            continue

        tp_price = entry_price + (risk * cfg["rr_ratio"]) if direction == "LONG" else entry_price - (risk * cfg["rr_ratio"])

        filled_idx: Optional[int] = None
        max_search_idx = min(n - 1, i + cfg["max_wait_candles"])

        for j in range(i + 1, max_search_idx + 1):
            if direction == "LONG":
                if lows[j] <= sl_price:
                    break
                if lows[j] <= entry_price:
                    filled_idx = j
                    break
            else:
                if highs[j] >= sl_price:
                    break
                if highs[j] >= entry_price:
                    filled_idx = j
                    break

        if filled_idx is None:
            i += 1
            continue

        outcome = None
        exit_idx = None

        for k in range(filled_idx, n):
            if direction == "LONG":
                hit_sl = lows[k] <= sl_price
                hit_tp = highs[k] >= tp_price
            else:
                hit_sl = highs[k] >= sl_price
                hit_tp = lows[k] <= tp_price

            if hit_sl:
                outcome = "SL"
                exit_idx = k
                break
            elif hit_tp:
                outcome = "TP"
                exit_idx = k
                break

        if outcome is not None and exit_idx is not None:
            r_gross = cfg["rr_ratio"] if outcome == "TP" else -1.0
            r_net = r_gross - cfg["cost_per_trade_r"]

            trades.append(Trade(
                symbol=symbol,
                split=split_label,
                direction=direction,
                fvg_time=v3_time,
                entry_time=times[filled_idx],
                exit_time=times[exit_idx],
                entry_price=round(entry_price, 5),
                sl_price=round(sl_price, 5),
                tp_price=round(tp_price, 5),
                outcome=outcome,
                r_gross=r_gross,
                r_net=round(r_net, 4),
                candles_waited=filled_idx - i,
                duration_candles=exit_idx - filled_idx
            ))
            i = exit_idx + 1
        else:
            i += 1

    return trades


def summarize_segment(name: str, trades: List[Trade]):
    if not trades:
        print(f"{name}: Sin operaciones.")
        return
    df_t = pd.DataFrame([t.__dict__ for t in trades])
    total = len(df_t)
    tp_n = (df_t["outcome"] == "TP").sum()
    sl_n = (df_t["outcome"] == "SL").sum()
    print(f"[{name}] Trades: {total} | TP: {tp_n} ({tp_n/total*100:.2f}%) | SL: {sl_n} ({sl_n/total*100:.2f}%) | Expectativa: {df_t['r_net'].mean():+.3f}R")


def print_command_center_for_active_entry(r_samples: np.ndarray, cfg: dict):
    if len(r_samples) < 10:
        print("⚠️ No hay suficiente histórico de trades para calcular el Command Center.")
        return

    rng = np.random.default_rng(seed=42)
    n_sims = cfg["mc_simulations"]
    horizon = cfg["mc_horizon_trades"]
    init_cap = cfg["initial_capital"]
    risk_pct = cfg["risk_per_trade_pct"]

    sampled = rng.choice(r_samples, size=(n_sims, horizon), replace=True)
    is_tp = (sampled > 0).astype(np.float64)

    tp_pct = is_tp.mean(axis=1) * 100.0
    sl_pct = 100.0 - tp_pct

    eq = init_cap * np.cumprod(1.0 + sampled * risk_pct, axis=1)
    eq = np.hstack([np.full((n_sims, 1), init_cap), eq])
    peak = np.maximum.accumulate(eq, axis=1)
    dd = ((peak - eq) / peak).max(axis=1) * 100.0

    print("-" * 76)
    print(f"🎛️  COMMAND CENTER ({n_sims:,} x {horizon} TRADES)")
    print("-" * 76)
    print(f"Prob. media tocar TP  : {tp_pct.mean():.2f}%  (IC 90%: {np.percentile(tp_pct, 5):.1f}% - {np.percentile(tp_pct, 95):.1f}%)")
    print(f"Prob. media tocar SL  : {sl_pct.mean():.2f}%  (IC 90%: {np.percentile(sl_pct, 5):.1f}% - {np.percentile(sl_pct, 95):.1f}%)")
    print(f"Prob. acabar beneficio: {(eq[:, -1] > init_cap).mean() * 100:.2f}%  |  Prob. DD >= 20%: {(dd >= 20.0).mean() * 100:.2f}%")
    print(f"Pesimista (P5)        : ${np.percentile(eq[:, -1], 5):,.2f}  | Max DD P95: {np.percentile(dd, 95):.2f}%")
    print(f"Mediano   (P50)       : ${np.percentile(eq[:, -1], 50):,.2f}  | Max DD P50: {np.percentile(dd, 50):.2f}%")
    print(f"Optimista (P95)       : ${np.percentile(eq[:, -1], 95):,.2f}  | Max DD P5 : {np.percentile(dd, 5):.2f}%")


# ==============================================================================
# 5. FLUJO PRINCIPAL
# ==============================================================================
def run_all(cfg: dict):
    user_input = input(f"Introduce el símbolo en vivo (Enter para '{cfg['default_ticker']}'): ").strip().upper()
    symbol = user_input if user_input else cfg["default_ticker"]

    # 1) Revisar estado actual con yfinance
    df_live = load_live_15m(symbol, cfg["live_period"], cfg["interval_15m"])
    setup: Optional[SetupStatus] = None

    print("=" * 76)
    if df_live is not None and len(df_live) >= 3:
        last_ts = df_live.index[-1]
        last_close = float(df_live["Close"].iloc[-1])
        setup = check_current_setup(df_live, cfg)

        print(f"📡 ESTADO ACTUAL ({symbol} | 15m) — Vela: {last_ts} | Precio: {last_close:.2f}")
        print("-" * 76)
        if setup is None:
            print("NO HAY ENTRY NI SETUP ACTIVO EN ESTE MOMENTO.")
        elif setup.state == "ENTRY_ACTIVA":
            print(f"▶ ENTRY ACTIVA ({setup.direction} | Sesión {setup.session} | FVG: {setup.fvg_time})")
            print(f"  Entry (50% CE) : {setup.entry_price:.2f}")
            print(f"  Stop Loss      : {setup.sl_price:.2f}")
            print(f"  Take Profit    : {setup.tp_price:.2f} (1:2)")
            print(f"  Nota           : {setup.detail}")
        else:
            print(f"⏳ ESPERANDO RETESTEO ({setup.direction} | Sesión {setup.session} | FVG: {setup.fvg_time})")
            print(f"  Entry (50% CE) : {setup.entry_price:.2f}")
            print(f"  Stop Loss      : {setup.sl_price:.2f}")
            print(f"  Take Profit    : {setup.tp_price:.2f} (1:2)")
            print(f"  Nota           : {setup.detail}")
    else:
        print(f"⚠️ No se pudieron obtener datos en vivo para '{symbol}'.")

    # 2) Máquina del tiempo si hay archivos .zip de HistData presentes
    zip_files = find_matching_zips(symbol, cfg["zip_pattern"])
    df_hist = load_histdata_zips_to_15m(zip_files, cfg["resample_rule"])

    trades_unseen: List[Trade] = []
    if df_hist is not None and len(df_hist) >= 100:
        split_idx = int(len(df_hist) * cfg["train_ratio"])
        df_train = df_hist.iloc[:split_idx]
        df_unseen = df_hist.iloc[split_idx:]

        trades_train = run_backtest_segment(df_train, symbol, "TRAIN", cfg)
        trades_unseen = run_backtest_segment(df_unseen, symbol, "UNSEEN", cfg)

        print("=" * 76)
        print(f"⏱️  MÁQUINA DEL TIEMPO ({len(zip_files)} zip(s) | {len(df_hist):,} velas 15m)")
        print("-" * 76)
        summarize_segment("TRAMO 70% (TRAIN) ", trades_train)
        summarize_segment("TRAMO 30% (UNSEEN)", trades_unseen)

        all_trades = trades_train + trades_unseen
        if all_trades:
            safe_sym = symbol.replace("=", "_").replace("/", "_")
            out_csv = f"backtest_fvg_{safe_sym}_split.csv"
            pd.DataFrame([t.__dict__ for t in all_trades]).to_csv(out_csv, index=False)
            print(f"📁 Guardado: {out_csv}")
    else:
        print("=" * 76)
        print("ℹ️ Sin archivos .zip de HistData en la carpeta: se omite Máquina del Tiempo.")

    # 3) Command Center solo si hay ENTRY ACTIVA
    if setup is not None and setup.state == "ENTRY_ACTIVA":
        if trades_unseen:
            r_samples = np.array([t.r_net for t in trades_unseen], dtype=np.float64)
        elif df_live is not None and len(df_live) >= 50:
            live_trades = run_backtest_segment(df_live, symbol, "LIVE", cfg)
            r_samples = np.array([t.r_net for t in live_trades], dtype=np.float64)
        else:
            r_samples = np.array([], dtype=np.float64)

        print_command_center_for_active_entry(r_samples, cfg)

    print("=" * 76)


if __name__ == "__main__":
    run_all(CONFIG)