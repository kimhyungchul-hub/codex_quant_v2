# engines/engine_hub.py

from engines.dummy_engine import DummyEngine
import time
from engines.mc.constants import MC_VERBOSE_PRINT


class EngineHub:
    """
    실전용 엔진 허브
    - 엔진 안전 로딩
    - EV 중심 통합
    """

    # =========================
    # sanitize (핵심!)
    # =========================
    @staticmethod
    def _sanitize(obj):
        # JAX -> host
        try:
            from jax import device_get  # type: ignore
            obj = device_get(obj)
        except Exception:
            pass

        # dict / list 재귀
        if isinstance(obj, dict):
            return {str(k): EngineHub._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [EngineHub._sanitize(v) for v in obj]

        # numpy / array-like
        try:
            import numpy as np
            arr = np.asarray(obj)

            # scalar
            if arr.ndim == 0:
                item = arr.item()
                if isinstance(item, (int, float, np.floating)):
                    return float(item)
            return arr.tolist()
        except Exception:
            pass

        return obj

    def __init__(self):
        self.engines = []
        self._load_engines()

        if not self.engines:
            self.engines.append(DummyEngine())

    def _load_engines(self):
        self._safe_load(self._load_mc_engine)

    def _safe_load(self, loader_fn):
        try:
            engine = loader_fn()
            self.engines.append(engine)
            print(f"✅ Engine loaded: {engine.name}")
        except Exception as e:
            print(f"⚠️ Engine skipped: {e}")

    def _load_mc_engine(self):
        from engines.mc.monte_carlo_engine import MonteCarloEngine
        return MonteCarloEngine()

    # =========================
    # decision
    # =========================
    def decide(self, ctx: dict) -> dict:
        results = []

        for engine in self.engines:
            try:
                # Debug: log engine name before decide
                if MC_VERBOSE_PRINT:
                    print(f"[PMAKER_DEBUG] EngineHub | calling engine.decide: engine.name={engine.name}")
                
                t_start = time.time()
                res = engine.decide(ctx)
                dt_ms = (time.time() - t_start) * 1000.0
                
                # ✅ PERF LOG: Warn if engine takes > 100ms
                if dt_ms > 100.0:
                    print(f"⚠️ [PERF_WARN] {engine.name}.decide took {dt_ms:.2f}ms for {ctx.get('symbol')}")

                # Debug: log res structure before modifying

                if MC_VERBOSE_PRINT:
                    print(f"[PMAKER_DEBUG] EngineHub | {engine.name} decide: res type={type(res)} res is None={res is None}")
                if res is None:
                    if MC_VERBOSE_PRINT:
                        print(f"[PMAKER_DEBUG] EngineHub | {engine.name} decide: res is None, creating default result")
                    res = {
                        "action": "WAIT",
                        "ev": 0.0,
                        "confidence": 0.0,
                        "reason": f"{engine.name} returned None",
                        "_engine": engine.name,
                        "_weight": engine.weight,
                        "meta": {},
                    }
                elif not isinstance(res, dict):
                    if MC_VERBOSE_PRINT:
                        print(
                            f"[PMAKER_DEBUG] EngineHub | {engine.name} decide: res is not dict, type={type(res)}, creating default result"
                        )
                    res = {
                        "action": "WAIT",
                        "ev": 0.0,
                        "confidence": 0.0,
                        "reason": f"{engine.name} returned {type(res).__name__}",
                        "_engine": engine.name,
                        "_weight": engine.weight,
                        "meta": {},
                    }
                else:
                    if MC_VERBOSE_PRINT:
                        print(
                            f"[PMAKER_DEBUG] EngineHub | {engine.name} decide: res keys={list(res.keys())[:30]} res.get('meta')={type(res.get('meta'))} meta keys={list(res.get('meta', {}).keys())[:30] if isinstance(res.get('meta'), dict) else []}"
                        )
                res["_engine"] = engine.name
                res["_weight"] = engine.weight
                # pass through event-based MC metrics
                meta = res.get("meta") or {}
                # Debug: log meta keys for mc_engine
                if MC_VERBOSE_PRINT:
                    print(
                        f"[PMAKER_DEBUG] EngineHub | {engine.name} decide: meta keys={list(meta.keys())[:30] if isinstance(meta, dict) else []} pmaker_entry={meta.get('pmaker_entry') if isinstance(meta, dict) else None}"
                    )
                for k in (
                    "event_p_tp",
                    "event_p_sl",
                    "event_p_timeout",
                    "event_ev_r",
                    "event_cvar_r",
                    "event_t_median",
                    "event_t_mean",
                ):
                    if k in meta:
                        res[k] = meta[k]
                results.append(res)
            except Exception as e:
                if MC_VERBOSE_PRINT:
                    print(f"[PMAKER_DEBUG] EngineHub | {engine.name} decide: exception={e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "action": "WAIT",
                    "ev": 0.0,
                    "confidence": 0.0,
                    "reason": f"{engine.name} error: {e}",
                    "_engine": engine.name,
                    "_weight": engine.weight,
                })

        # [EV_DEBUG] 각 engine의 EV 값 확인
        if MC_VERBOSE_PRINT:
            print(f"[EV_DEBUG] EngineHub.decide: results count={len(results)}")
        for idx, r in enumerate(results):
            ev_val = r.get("ev", 0.0)
            weight = r.get("_weight", 1.0)
            engine_name = r.get("_engine", "unknown")
            if MC_VERBOSE_PRINT:
                print(
                    f"[EV_DEBUG] EngineHub.decide: result[{idx}] engine={engine_name} ev={ev_val} weight={weight} ev*weight={ev_val * weight}"
                )
        
        # ✅ [EV_DEBUG] 각 result의 ev 값 확인 및 처리
        ev_values = []
        ev_raw_values = []
        for r in results:
            ev_raw_val = r.get("ev_raw", r.get("ev"))
            ev_val = float(r.get("ev", 0.0) or 0.0)
            ev_raw_float = float(ev_raw_val) if ev_raw_val is not None else 0.0
            weight = float(r.get("_weight", 1.0))
            ev_values.append((ev_val, weight, r.get("_engine", "unknown")))
            ev_raw_values.append((ev_raw_float, weight))
            if MC_VERBOSE_PRINT:
                print(
                    f"[EV_DEBUG] EngineHub.decide: result ev={ev_val} ev_raw={ev_raw_float} weight={weight} engine={r.get('_engine', 'unknown')}"
                )
        
        total_ev = sum(ev * w for ev, w, _ in ev_values)
        total_ev_raw = sum(ev_r * w for ev_r, w in ev_raw_values)
        best = max(results, key=lambda r: float(r.get("ev", 0.0) or 0.0))

        # 🔥 [FIX] SCORE_ONLY인 경우 EV가 음수여도 진입 허용 (total_ev > 0 제약 우회)
        is_score_only = any("SCORE_ONLY" in str(r.get("reason", "")) for r in results)
        if is_score_only:
            final_action = best["action"]
        else:
            final_action = best["action"] if total_ev > 0 else "WAIT"

        if MC_VERBOSE_PRINT:
            print(
                f"[EV_DEBUG] EngineHub.decide: total_ev={total_ev} total_ev_raw={total_ev_raw} best_action={best.get('action')} best_ev={best.get('ev')} final_action={final_action}"
            )

        final = {
            "action": final_action,
            "ev": total_ev,
            "ev_raw": total_ev_raw,
            "confidence": max((float(r.get("confidence", 0.0) or 0.0) for r in results), default=0.0),
            "reason": " | ".join(r.get("reason", "") for r in results),
            "details": results,
        }

        # 🔥 최종 경계에서 무조건 sanitize
        if MC_VERBOSE_PRINT:
            print(
                f"[EV_DEBUG] EngineHub.decide: BEFORE sanitize: final ev={final.get('ev')} (type={type(final.get('ev'))}) action={final.get('action')}"
            )
        sanitized = EngineHub._sanitize(final)
        if MC_VERBOSE_PRINT:
            print(
                f"[EV_DEBUG] EngineHub.decide: AFTER sanitize: final ev={sanitized.get('ev')} (type={type(sanitized.get('ev'))}) action={sanitized.get('action')}"
            )
        return sanitized

    def decide_batch(self, ctx_list: list[dict]) -> list[dict]:
        """
        GLOBAL BATCHING: 모든 심볼에 대해 한 번에 의사결정을 수행한다.
        """
        num_ctx = len(ctx_list)
        if num_ctx == 0:
            return []

        # MC 엔진 찾기
        mc_engine = next(
            (e for e in self.engines if getattr(e, "name", "") == "mc_barrier"), None
        )

        if mc_engine and hasattr(mc_engine, "decide_batch"):
            try:
                # MC 엔진의 배치 버전 호출 (로그/타이밍 추가)
                import time, os
                env_flag = str(os.environ.get("MC_VERBOSE_PRINT", "0")).strip().lower() in ("1", "true", "yes")
                # Always log batch call timing
                print(f"[ENGINEHUB_BATCH] calling mc_engine.decide_batch for {len(ctx_list)} ctxs")
                t0 = time.perf_counter()
                batch_results = mc_engine.decide_batch(ctx_list)
                t1 = time.perf_counter()
                print(f"[ENGINEHUB_BATCH] mc_engine.decide_batch done in {(t1-t0):.3f}s")
                # 각 결과를 개별적으로 sanitize
                return [self._sanitize(res) for res in batch_results]
            except Exception as e:
                import traceback
                print(f"⚠️ [decide_batch] mc_engine error: {e}")
                traceback.print_exc()
                # Fallback to sequential
        
        # Fallback: 순차 처리
        return [self.decide(ctx) for ctx in ctx_list]
