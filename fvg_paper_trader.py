import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

# ==============================================================================
# 1. CONFIGURACIÓN DE PAPER TRADING (FVG 15M + 50% CE + 1:2 + LONDRES/NY)
# ==============================================================================
CONFIG = {
    "symbols": [
        "BTC-USD", "ETH-USD", "SOL-USD",
        "EURUSD=X", "GBPUSD=X",
        "GC=F", "SI=F",
        "SPY", "QQQ", "DX-Y.NYB",
        "AAPL", "TSLA", "META", "NVDA", "GOOGL", "MSFT"
    ],
    "period": "5d",
    "interval": "15m",
    "entry_depth": 0.50,            # 50% (CE / Midpoint del FVG)
    "max_wait_candles": 20,         # Máximo 20 velas de 15m esperando retesteo
    "rr_ratio": 2.0,                # TP fijo 1:2
    "min_fvg_pct": 0.0006,          # Hueco mínimo: 0.06%
    "min_displacement_ratio": 0.35, # Hueco >= 35% del cuerpo de v2
    "sl_buffer_pct": 0.0001,        # Margen mínimo detrás de v1
    "initial_capital": 10000.0,     # Saldo inicial paper (USD)
    "risk_per_trade_pct": 0.01,     # 1% de riesgo por operación en paper
    "loop_seconds": 60,             # Revisión cada 60s si se ejecuta en bucle
    "state_file": "paper_state.json",
    "log_file": "paper_trades_log.csv",
}

CRYPTO_SYMBOLS = {"BTC-USD", "ETH-USD", "SOL-USD"}


@dataclass
class PaperPosition:
    trade_id: str
    symbol: str
    direction: str          # "LONG" o "SHORT"
    session: str            # "LONDON", "NY" o "CRYPTO_24_7"
    status: str             # "PENDING" o "OPEN"
    fvg_time: str
    created_at: str
    entry_price: float
    sl_price: float
    tp_price: float
    risk_amount: float
    expires_candle_idx_ref: str
    entry_time: Optional[str] = None


# ==============================================================================
# 2. SESIONES NY Y DETECCIÓN DEL FVG
# ==============================================================================
def get_session_ny(ts: pd.Timestamp) -> Optional[str]:
    t = ts.time()
    if pd.Timestamp("02:00").time() <= t <= pd.Timestamp("05:00").time():
        return "LONDON"
    if pd.Timestamp("08:30").time() <= t <= pd.Timestamp("11:30").time():
        return "NY"
    return None


def detect_fvg_at_index(df: pd.DataFrame, idx: int, cfg: dict, symbol: str = "") -> Optional[dict]:
    if idx < 2 or idx >= len(df):
        return None

    if symbol in CRYPTO_SYMBOLS:
        session = "CRYPTO_24_7"
    else:
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
            if entry <= sl:
                return None
            tp = entry + ((entry - sl) * rr)
            return {
                "direction": "LONG",
                "session": session,
                "fvg_time": str(df.index[idx]),
                "entry": float(entry),
                "sl": float(sl),
                "tp": float(tp),
            }

    if h3 < l1 and c2 < o2:
        gap = l1 - h3
        if (gap / c2 >= cfg["min_fvg_pct"]) and (gap / body_v2 >= cfg["min_displacement_ratio"]):
            entry = h3 + (gap * depth)
            sl = h1 * (1.0 + buf)
            if sl <= entry:
                return None
            tp = entry - ((sl - entry) * rr)
            return {
                "direction": "SHORT",
                "session": session,
                "fvg_time": str(df.index[idx]),
                "entry": float(entry),
                "sl": float(sl),
                "tp": float(tp),
            }

    return None


# ==============================================================================
# 3. PERSISTENCIA DE ESTADO Y LOG DE PAPER TRADES
# ==============================================================================
def load_state(cfg: dict) -> dict:
    path = cfg["state_file"]
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
            data.setdefault("balance", cfg["initial_capital"])
            data.setdefault("positions", {})
            data.setdefault("seen_fvgs", [])
            return data
    return {
        "balance": cfg["initial_capital"],
        "positions": {},
        "seen_fvgs": [],
    }


def save_state(state: dict, cfg: dict):
    with open(cfg["state_file"], "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def append_closed_trade_csv(row: dict, cfg: dict):
    df_row = pd.DataFrame([row])
    path = cfg["log_file"]
    write_header = not os.path.exists(path)
    df_row.to_csv(path, mode="a", index=False, header=write_header)


# ==============================================================================
# 4. CARGA DE DATOS 15M Y SIMULACIÓN DE EJECUCIÓN PAPER
# ==============================================================================
def fetch_15m(symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty or len(df) < 3:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df.tz_convert("America/New_York")
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return None


def step_symbol_paper(symbol: str, df: pd.DataFrame, state: dict, cfg: dict):
    last_idx = len(df) - 1
    last_close = float(df["Close"].iloc[last_idx])
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    pos_dict: Dict[str, dict] = state["positions"]
    pos = pos_dict.get(symbol)

    # 1) Gestionar posición existente: se repasan TODAS las velas desde el FVG
    if pos is not None:
        direction = pos["direction"]
        entry, sl, tp = pos["entry_price"], pos["sl_price"], pos["tp_price"]
        fvg_ts = pd.Timestamp(pos["fvg_time"])

        if pos["status"] == "OPEN":
            mask = df.index >= pd.Timestamp(pos["entry_time"])
        else:
            mask = df.index > fvg_ts

        for ts, h, l in zip(df.index[mask], df["High"][mask], df["Low"][mask]):
            h, l = float(h), float(l)

            if pos["status"] == "PENDING":
                invalidated = (l <= sl) if direction == "LONG" else (h >= sl)
                if invalidated:
                    print(f"  [PAPER] {symbol} ({direction}) orden pendiente cancelada (INVALIDATED_SL).")
                    del pos_dict[symbol]
                    pos = None
                    break
                filled = (l <= entry) if direction == "LONG" else (h >= entry)
                if not filled:
                    continue
                pos["status"] = "OPEN"
                pos["entry_time"] = str(ts)
                print(f"  ▶ [PAPER ENTRY] {symbol} {direction} @ {entry:.4f} | SL: {sl:.4f} | TP: {tp:.4f}")

            hit_sl = (l <= sl) if direction == "LONG" else (h >= sl)
            hit_tp = (h >= tp) if direction == "LONG" else (l <= tp)
            if hit_sl or hit_tp:
                outcome = "SL" if hit_sl else "TP"   # si salta ambos en la misma vela, cuenta SL (conservador)
                r_mult = cfg["rr_ratio"] if outcome == "TP" else -1.0
                pnl_usd = round(pos["risk_amount"] * r_mult, 2)
                state["balance"] = round(state["balance"] + pnl_usd, 2)

                append_closed_trade_csv({
                    "trade_id": pos["trade_id"], "symbol": symbol, "direction": direction,
                    "session": pos["session"], "fvg_time": pos["fvg_time"],
                    "entry_time": pos["entry_time"], "exit_time": str(ts),
                    "closed_at_utc": now_str, "entry_price": entry, "sl_price": sl,
                    "tp_price": tp, "exit_price": tp if outcome == "TP" else sl,
                    "outcome": outcome, "r_multiple": r_mult, "pnl_usd": pnl_usd,
                    "balance_after": state["balance"],
                }, cfg)
                print(f"  {'🎯' if outcome == 'TP' else '🛑'} [PAPER CERRADO {outcome}] {symbol} {direction} "
                      f"| PnL: {pnl_usd:+.2f} USD | Saldo: ${state['balance']:,.2f}")
                del pos_dict[symbol]
                pos = None
                break

        # Caducidad de la orden pendiente
        if pos is not None and pos["status"] == "PENDING":
            if int((df.index > fvg_ts).sum()) > cfg["max_wait_candles"]:
                print(f"  [PAPER] {symbol} ({direction}) orden pendiente cancelada (EXPIRED_20_BARS).")
                del pos_dict[symbol]

    # 2) Si no hay posición en el símbolo, elegir FVG como el scanner: ENTRY activa > pendiente
    if symbol not in pos_dict:
        start_idx = max(2, last_idx - cfg["max_wait_candles"])
        active, waiting = None, None

        for check_idx in range(start_idx, last_idx + 1):
            fvg = detect_fvg_at_index(df, check_idx, cfg, symbol=symbol)
            if fvg is None:
                continue

            fvg_key = f"{symbol}|{fvg['fvg_time']}|{fvg['direction']}"
            if fvg_key in state["seen_fvgs"]:
                continue

            is_long = fvg["direction"] == "LONG"
            entry, sl, tp = fvg["entry"], fvg["sl"], fvg["tp"]
            fill_ts, dead = None, False

            for j in range(check_idx + 1, last_idx + 1):
                h, l = float(df["High"].iloc[j]), float(df["Low"].iloc[j])
                if fill_ts is None:
                    if (l <= sl) if is_long else (h >= sl):
                        dead = True          # invalidado antes de tocar el 50%
                        break
                    if (l <= entry) if is_long else (h >= entry):
                        fill_ts = df.index[j]
                if fill_ts is not None:
                    hit_sl = (l <= sl) if is_long else (h >= sl)
                    hit_tp = (h >= tp) if is_long else (l <= tp)
                    if hit_sl or hit_tp:
                        dead = True          # ya cerrado por SL o TP
                        break

            if dead:
                continue

            cand = (check_idx, fvg_key, fvg, fill_ts)
            if fill_ts is not None:
                active = cand                # el más reciente activo
            else:
                waiting = cand               # el más reciente pendiente

        chosen = active or waiting
        if chosen is not None:
            check_idx, fvg_key, fvg, fill_ts = chosen

            state["seen_fvgs"].append(fvg_key)
            state["seen_fvgs"] = state["seen_fvgs"][-500:]

            risk_usd = round(state["balance"] * cfg["risk_per_trade_pct"], 2)
            trade_id = f"{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

            pos_dict[symbol] = asdict(PaperPosition(
                trade_id=trade_id,
                symbol=symbol,
                direction=fvg["direction"],
                session=fvg["session"],
                status="OPEN" if fill_ts is not None else "PENDING",
                fvg_time=fvg["fvg_time"],
                created_at=now_str,
                entry_price=fvg["entry"],
                sl_price=fvg["sl"],
                tp_price=fvg["tp"],
                risk_amount=risk_usd,
                expires_candle_idx_ref=str(df.index[check_idx]),
                entry_time=str(fill_ts) if fill_ts is not None else None,
            ))
            print(
                f"  ⏳ [PAPER {'ENTRY' if fill_ts is not None else 'ORDEN'}] {symbol} {fvg['direction']} ({fvg['session']}) "
                f"| Entry 50%: {fvg['entry']:.4f} | SL: {fvg['sl']:.4f} | TP: {fvg['tp']:.4f}"
            )

    estado_txt = pos_dict[symbol]["status"] if symbol in pos_dict else "SIN_POSICION"
    print(f"  • {symbol:<10} | Close: {last_close:.4f} | Estado: {estado_txt}")


def run_cycle(cfg: dict):
    state = load_state(cfg)
    print("=" * 78)
    print(f"🤖 PAPER TRADER 15M (50% CE | 1:2) | Saldo Paper: ${state['balance']:,.2f}")
    print("=" * 78)

    for sym in cfg["symbols"]:
        df = fetch_15m(sym, cfg["period"], cfg["interval"])
        if df is None:
            print(f"  ⚠️ {sym:<10} | Sin datos en este momento.")
            continue
        step_symbol_paper(sym, df, state, cfg)

    save_state(state, cfg)
    print("=" * 78)


if __name__ == "__main__":
    mode = input("Modo [1] Una pasada ahora | [2] Bucle automático (Enter = 1): ").strip()
    if mode == "2":
        print(f"Iniciando bucle cada {CONFIG['loop_seconds']}s. Pulsa Ctrl + C para detener.")
        try:
            while True:
                run_cycle(CONFIG)
                time.sleep(CONFIG["loop_seconds"])
        except KeyboardInterrupt:
            print("\nPaper trader detenido por el usuario.")
    else:
        run_cycle(CONFIG)


if __name__ == "__main__":
    mode = input("Modo [1] Una pasada ahora | [2] Bucle automático (Enter = 1): ").strip()
    if mode == "2":
        print(f"Iniciando bucle cada {CONFIG['loop_seconds']}s. Pulsa Ctrl + C para detener.")
        try:
            while True:
                run_cycle(CONFIG)
                time.sleep(CONFIG["loop_seconds"])
        except KeyboardInterrupt:
            print("\nPaper trader detenido por el usuario.")
    else:
        run_cycle(CONFIG)