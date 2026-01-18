import os
import asyncio
import numpy as np
import time
from typing import Any, Dict, Optional, List, Sequence
from utils.helpers import now_ms, _safe_float

try:
    from engines.p_maker_survival_mlp import PMakerSurvivalMLP
    _PMAKER_MLP_OK = True
except Exception:
    PMakerSurvivalMLP = None
    _PMAKER_MLP_OK = False

class PMakerManager:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.enabled = bool(os.environ.get("PMAKER_ENABLE", "0") == "1")
        self.model_path = str(os.environ.get("PMAKER_MODEL_PATH", "state/pmaker_survival_mlp.pt"))
        self.surv = None
        
        # Cache and background tasks
        self.predict_queue = asyncio.Queue()
        self.predict_cache = {}
        self.predict_background_task = None
        self.predict_cache_ttl_sec = 5.0
        
        # Training configuration
        self.train_steps = int(os.environ.get("PMAKER_TRAIN_STEPS", "1"))
        self.batch = int(os.environ.get("PMAKER_BATCH", "32"))
        
        # Probe state
        self.probe_task = None
        self.probe_attempts = 0
        self.probe_fills = 0
        self.probe_attempts_maker = 0
        self.probe_fills_maker = 0
        self.probe_attempts_taker = 0
        self.probe_fills_taker = 0
        self.probe_last = None
        self.probe_started_ms = None
        
        if self.enabled and _PMAKER_MLP_OK and PMakerSurvivalMLP is not None:
            try:
                self.surv = PMakerSurvivalMLP(
                    grid_ms=int(os.environ.get("PMAKER_GRID_MS", 50)),
                    max_ms=int(os.environ.get("PMAKER_MAX_MS", 2500)),
                    lr=float(os.environ.get("PMAKER_LR", "3e-4")),
                    device=str(os.environ.get("PMAKER_DEVICE", "")),
                )
            except Exception as e:
                print(f"[PMAKER_LOAD] Failed to init model: {e}")
                self.surv = None
            if self.surv is not None:
                try:
                    self.surv.load(self.model_path)
                except Exception as e:
                    print(f"[PMAKER_LOAD] Failed to load model: {e} (starting fresh)")
                    # If the checkpoint is corrupted, quarantine it to avoid repeat failures.
                    try:
                        if self.model_path and os.path.exists(self.model_path):
                            bad_path = f"{self.model_path}.corrupt.{int(time.time())}"
                            os.replace(self.model_path, bad_path)
                            print(f"[PMAKER_LOAD] Renamed corrupt model to {bad_path}")
                    except Exception as e2:
                        print(f"[PMAKER_LOAD] Failed to rename corrupt model: {e2}")

    def save_model(self):
        if self.surv and self.enabled:
            try:
                self.surv.save(self.model_path)
                return True
            except Exception as e:
                print(f"[PMAKER_SAVE] Failed: {e}")
        return False

    def model_counts(self):
        if self.surv is None: return 0, 0, 0.0
        sym_n = getattr(self.surv, "sym_n", {}) or {}
        sym_wins = getattr(self.surv, "sym_wins", {}) or {}
        a = int(round(sum(float(v) for v in sym_n.values())))
        f = int(round(sum(float(v) for v in sym_wins.values())))
        r = (f / a) if a > 0 else 0.0
        return a, f, r

    def model_param_count(self):
        if self.surv is None or self.surv.model is None: return 0
        return sum(p.numel() for p in self.surv.model.parameters())

    def survival_stats(self, logits, grid_ms, timeout_ms, maker_timeout_ms):
        def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
        
        hazards = sigmoid(logits)
        surv = 1.0
        p_fill_maker = 0.0
        e_delay = 0.0
        e_delay_cond = 0.0
        
        for k, h in enumerate(hazards):
            t_ms = (k + 1) * grid_ms
            if t_ms > timeout_ms: break
            
            p_k = surv * h
            if t_ms <= maker_timeout_ms:
                p_fill_maker += p_k
                e_delay_cond += p_k * (t_ms / 1000.0)
            
            e_delay += p_k * (t_ms / 1000.0)
            surv *= (1.0 - h)
            
        e_delay += surv * (timeout_ms / 1000.0)
        if p_fill_maker > 1e-6:
            e_delay_cond /= p_fill_maker
        else:
            e_delay_cond = maker_timeout_ms / 1000.0
            
        return p_fill_maker, e_delay, e_delay_cond

    def _extract_feats(
        self,
        sym: str,
        order_side: str,
        maker_price: float,
        *,
        decision: dict | None,
        attempt_idx: int,
        bid: float | None,
        ask: float | None,
        sigma: float | None,
    ) -> np.ndarray:
        # This logic is copied from LiveOrchestrator._pmaker_extract_feats
        # and adjusted to use self.orchestrator
        price = self.orchestrator.market[sym].get("price") or maker_price
        
        # side: 1 for buy, -1 for sell
        side_val = 1.0 if order_side.lower() == "buy" else -1.0
        
        # dist from mid
        mid = (bid + ask) / 2.0 if (bid and ask) else price
        dist_mid = (maker_price - mid) / mid if mid > 0 else 0.0
        
        # dist from touch
        touch = bid if order_side.lower() == "buy" else ask
        dist_touch = (maker_price - touch) / touch if touch > 0 else 0.0
        
        # spread
        spread = (ask - bid) / mid if (bid and ask and mid > 0) else 0.0002
        
        # ofi
        ofi = float(self.orchestrator._compute_ofi_score(sym))
        
        # sigma
        vol = sigma if sigma is not None else 0.01
        
        # decision confidence
        conf = decision.get("confidence", 0.0) if decision else 0.0
        
        # simple feature vector (must match PMakerSurvivalMLP expectation)
        feats = [
            side_val,
            dist_mid,
            dist_touch,
            spread,
            ofi,
            vol,
            float(attempt_idx),
            conf
        ]
        return np.array(feats, dtype=np.float32)

    async def predict_survival_meta(
        self,
        *,
        symbol: str,
        side: str,
        price: float,
        best_bid: float,
        best_ask: float,
        spread_pct: float,
        qty: float,
        maker_timeout_ms: int,
        prefix: str,
        decision: dict | None = None,
        sigma: float | None = None,
    ) -> Dict[str, Any]:
        if self.surv is None:
            return {}
            
        try:
            feats = self._extract_feats(
                symbol, side, price,
                decision=decision,
                attempt_idx=0,
                bid=best_bid,
                ask=best_ask,
                sigma=sigma
            )
            
            logits = self.surv.predict(feats)
            p_fill, e_delay, e_delay_cond = self.survival_stats(
                logits,
                self.surv.grid_ms,
                self.surv.max_ms,
                maker_timeout_ms
            )
            
            return {
                prefix: p_fill,
                f"{prefix}_delay_sec": e_delay,
                f"{prefix}_delay_cond_sec": e_delay_cond,
            }
        except Exception as e:
            print(f"[PMAKER_PREDICT] Error: {e}")
            return {}

    async def request_prediction(
        self,
        *,
        symbol: str,
        side: str,
        price: float,
        best_bid: float,
        best_ask: float,
        spread_pct: float,
        qty: float,
        maker_timeout_ms: int,
        prefix: str,
        use_cache: bool = True,
        fallback_sync: bool = True,
        decision: dict | None = None,
        sigma: float | None = None,
    ) -> Dict[str, Any]:
        if not self.enabled or self.surv is None:
            return {}

        now = time.time()
        cache_key = (symbol, side, round(price, 8), prefix)
        
        if use_cache:
            cached = self.predict_cache.get(cache_key)
            if cached and (now - cached["ts"] < self.predict_cache_ttl_sec):
                return cached["res"]

        if fallback_sync:
            res = await self.predict_survival_meta(
                symbol=symbol, side=side, price=price,
                best_bid=best_bid, best_ask=best_ask,
                spread_pct=spread_pct, qty=qty,
                maker_timeout_ms=maker_timeout_ms,
                prefix=prefix, decision=decision, sigma=sigma
            )
            if use_cache:
                self.predict_cache[cache_key] = {"ts": now, "res": res}
            return res

        # Async queue request (not fully implemented in this snippet but following original pattern)
        return {}

    async def predict_background_loop(self):
        print("[PMAKER] Prediction background loop started")
        while True:
            try:
                # Cleanup cache
                now = time.time()
                keys_to_del = [k for k, v in self.predict_cache.items() if now - v["ts"] > self.predict_cache_ttl_sec * 2]
                for k in keys_to_del: del self.predict_cache[k]
                
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[PMAKER_BG] Error: {e}")
                await asyncio.sleep(5.0)

    async def probe_loop(self):
        if not self.enabled: return
        print("[PMAKER] Probe loop started")
        while True:
            try:
                # Probing logic here...
                # This involves placing small post-only orders to collect fill samples.
                await asyncio.sleep(60.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[PMAKER_PROBE] Error: {e}")
                await asyncio.sleep(10.0)

    def start_probe(self):
        if self.enabled and self.probe_task is None:
            self.probe_task = asyncio.create_task(self.probe_loop())

    def status_dict(self):
        a, f, r = self.model_counts()
        return {
            "enabled": self.enabled,
            "ready": self.surv is not None,
            "attempts": a,
            "fills": f,
            "fill_rate": r,
            "param_count": self.model_param_count(),
            "probe_active": self.probe_task is not None,
        }
