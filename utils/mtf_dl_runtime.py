from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from utils.mtf_multitask import (
    MTFMultiTaskNet,
    build_mtf_image_from_1m,
    infer_symbol_group,
    normalize_regime,
)

try:
    import torch
except Exception:  # pragma: no cover - runtime fallback when torch is missing
    torch = None


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        out = float(v)
        if not math.isfinite(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _to_float_array(values: Iterable[Any]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.float64)


class MTFDLRuntimeScorer:
    """
    Runtime scorer for the multitask MTF model.

    Output keys:
      - mtf_dl_prob_win
      - mtf_dl_prob_long
      - mtf_dl_prob_short
      - mtf_dl_hold_sec_pred
      - mtf_dl_exit_reason_top
      - mtf_dl_exit_reason_conf
    """

    def __init__(
        self,
        *,
        model_path: str,
        enabled: bool = True,
        device: str = "cpu",
    ) -> None:
        self.model_path = str(model_path or "").strip()
        self.enabled = bool(enabled)
        self.ready = False
        self.error: str | None = None
        self.model = None
        self.device = "cpu"

        self.timeframes: list[int] = [1, 3, 5, 15]
        self.lookback_bars: int = 16
        self.regime_to_id: dict[str, int] = {"unknown": 0}
        self.group_to_id: dict[str, int] = {"other": 0}
        self.exit_to_id: dict[str, int] = {"other": 0}
        self.id_to_exit: dict[int, str] = {0: "other"}
        self.hold_norm_mean: float = 0.0
        self.hold_norm_std: float = 1.0

        if not self.enabled:
            self.error = "disabled"
            return
        if torch is None:
            self.error = "torch_unavailable"
            return
        if not self.model_path:
            self.error = "empty_model_path"
            return
        mp = Path(self.model_path)
        if not mp.exists():
            self.error = f"model_not_found:{self.model_path}"
            return

        try:
            ckpt = torch.load(str(mp), map_location="cpu")
            self.timeframes = [max(1, int(x)) for x in (ckpt.get("timeframes") or self.timeframes)]
            self.timeframes = sorted(set(self.timeframes))
            self.lookback_bars = max(8, int(ckpt.get("lookback_bars", self.lookback_bars)))

            self.regime_to_id = {
                str(k).strip().lower(): int(v)
                for k, v in dict(ckpt.get("regime_to_id") or {"unknown": 0}).items()
            }
            if "unknown" not in self.regime_to_id:
                self.regime_to_id["unknown"] = 0

            self.group_to_id = {
                str(k).strip().lower(): int(v)
                for k, v in dict(ckpt.get("group_to_id") or {"other": 0}).items()
            }
            if "other" not in self.group_to_id:
                self.group_to_id["other"] = 0

            self.exit_to_id = {
                str(k).strip().lower(): int(v)
                for k, v in dict(ckpt.get("exit_to_id") or {"other": 0}).items()
            }
            if "other" not in self.exit_to_id:
                self.exit_to_id["other"] = 0
            self.id_to_exit = {int(v): str(k) for k, v in self.exit_to_id.items()}

            self.hold_norm_mean = _safe_float(ckpt.get("hold_norm_mean"), 0.0)
            self.hold_norm_std = max(1e-6, _safe_float(ckpt.get("hold_norm_std"), 1.0))

            h = int(ckpt.get("height", 12))
            w = int(ckpt.get("width", self.lookback_bars))

            model = MTFMultiTaskNet(
                h=h,
                w=w,
                n_regimes=len(self.regime_to_id),
                n_groups=len(self.group_to_id),
                n_exit_reasons=len(self.exit_to_id),
                embed_dim=8,
                hidden_dim=96,
            )
            model.load_state_dict(dict(ckpt.get("state_dict") or {}), strict=True)
            model.eval()

            self.device = self._resolve_device(device)
            model = model.to(self.device)
            self.model = model
            self.ready = True
        except Exception as e:
            self.error = f"load_failed:{e}"
            self.ready = False

    def _resolve_device(self, req: str) -> str:
        if torch is None:
            return "cpu"
        want = str(req or "cpu").strip().lower()
        if want == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if want == "mps" and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _lookup_id(vocab: dict[str, int], key: str, fallback: str) -> int:
        k = str(key or "").strip().lower()
        if k in vocab:
            return int(vocab[k])
        return int(vocab.get(fallback, 0))

    def predict(
        self,
        *,
        symbol: str,
        regime: str,
        ts_ms: Iterable[Any],
        open_: Iterable[Any],
        high: Iterable[Any],
        low: Iterable[Any],
        close: Iterable[Any],
        volume: Iterable[Any],
        entry_ts_ms: int,
    ) -> dict[str, Any] | None:
        if not self.ready or self.model is None or torch is None:
            return None

        try:
            img = build_mtf_image_from_1m(
                ts_ms=ts_ms,
                open_=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
                entry_ts_ms=int(entry_ts_ms),
                tf_list=list(self.timeframes),
                lookback=int(self.lookback_bars),
            )
            if img is None:
                return None

            reg = normalize_regime(regime)
            grp = infer_symbol_group(symbol)
            reg_id = self._lookup_id(self.regime_to_id, reg, "unknown")
            grp_id = self._lookup_id(self.group_to_id, grp, "other")

            x = torch.from_numpy(np.asarray(img, dtype=np.float32)[None, None, :, :]).to(self.device)
            r = torch.tensor([reg_id], dtype=torch.long, device=self.device)
            g = torch.tensor([grp_id], dtype=torch.long, device=self.device)

            with torch.no_grad():
                out = self.model(x, r, g)
                p_win = float(torch.sigmoid(out["win"]).detach().cpu().numpy()[0])
                p_long = float(torch.sigmoid(out["long"]).detach().cpu().numpy()[0])
                p_short = float(torch.sigmoid(out["short"]).detach().cpu().numpy()[0])
                hold_norm = float(out["hold"].detach().cpu().numpy()[0])
                exit_logits = out["exit"].detach().cpu()
                exit_prob = torch.softmax(exit_logits, dim=1).numpy()[0]

            hold_sec = float(np.expm1(hold_norm * self.hold_norm_std + self.hold_norm_mean))
            hold_sec = float(max(0.0, hold_sec))

            exit_idx = int(np.argmax(exit_prob))
            exit_reason = str(self.id_to_exit.get(exit_idx, "other"))
            exit_conf = float(exit_prob[exit_idx]) if exit_prob.size > 0 else 0.0

            return {
                "mtf_dl_prob_win": float(max(0.0, min(1.0, p_win))),
                "mtf_dl_prob_long": float(max(0.0, min(1.0, p_long))),
                "mtf_dl_prob_short": float(max(0.0, min(1.0, p_short))),
                "mtf_dl_hold_sec_pred": float(hold_sec),
                "mtf_dl_exit_reason_top": exit_reason,
                "mtf_dl_exit_reason_conf": float(max(0.0, min(1.0, exit_conf))),
                "mtf_dl_regime_norm": str(reg),
                "mtf_dl_symbol_group": str(grp),
            }
        except Exception:
            return None

    def predict_from_ohlcv_1m(
        self,
        *,
        symbol: str,
        regime: str,
        close: Iterable[Any],
        open_: Iterable[Any],
        high: Iterable[Any],
        low: Iterable[Any],
        volume: Iterable[Any],
        end_ts_ms: int,
        bar_ms: int = 60_000,
    ) -> dict[str, Any] | None:
        arr_c = _to_float_array(close)
        arr_o = _to_float_array(open_)
        arr_h = _to_float_array(high)
        arr_l = _to_float_array(low)
        arr_v = _to_float_array(volume)
        n = int(min(arr_c.size, arr_o.size, arr_h.size, arr_l.size, arr_v.size))
        if n <= 0:
            return None
        arr_c = arr_c[-n:]
        arr_o = arr_o[-n:]
        arr_h = arr_h[-n:]
        arr_l = arr_l[-n:]
        arr_v = arr_v[-n:]

        end_ts = int(end_ts_ms)
        start_ts = int(end_ts - (n - 1) * int(max(1, bar_ms)))
        ts = np.arange(start_ts, end_ts + int(max(1, bar_ms)), int(max(1, bar_ms)), dtype=np.int64)
        ts = ts[-n:]

        return self.predict(
            symbol=symbol,
            regime=regime,
            ts_ms=ts,
            open_=arr_o,
            high=arr_h,
            low=arr_l,
            close=arr_c,
            volume=arr_v,
            entry_ts_ms=int(end_ts_ms),
        )
