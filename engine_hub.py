# engines/engine_hub.py

class DummyEngine:
    """Fallback engine when no other engine loads successfully."""
    name = "DummyEngine"
    weight = 0.0

    def decide(self, ctx: dict) -> dict:
        return {
            "action": "WAIT",
            "ev": 0.0,
            "confidence": 0.0,
            "reason": "No engine available (fallback)",
            "meta": {},
        }


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
            # Lazily initialize JAX and use module-level jax if available
            import engines.mc.jax_backend as jax_backend
            jax_backend.ensure_jax()
            if getattr(jax_backend, "jax", None) is not None:
                obj = jax_backend.jax.device_get(obj)
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
                return str(item)

            # vector/matrix -> flat list
            return [
                float(x) if isinstance(x, (int, float, np.floating)) else str(x)
                for x in arr.reshape(-1).tolist()
            ]
        except Exception:
            pass

        # primitive
        if isinstance(obj, (int, float, bool)) or obj is None:
            return obj

        # fallback
        return str(obj)

    # =========================
    # lifecycle
    # =========================
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
        from mc_engine import MonteCarloEngine
        return MonteCarloEngine()

    # =========================
    # decision (batch)
    # =========================
    def decide_batch(self, ctx_list: list) -> list:
        """
        [OPTIMIZATION] Global Batching for GPU
        여러 심볼의 context를 받아 한 번의 처리로 결과를 반환합니다.
        """
        if not ctx_list:
            return []

        # MC 엔진이 decide_batch를 지원하면 사용
        for engine in self.engines:
            if hasattr(engine, "decide_batch"):
                try:
                    batch_results = engine.decide_batch(ctx_list)
                    # sanitize 및 메타 처리
                    final_results = []
                    for i, res in enumerate(batch_results):
                        res["_engine"] = engine.name
                        res["_weight"] = engine.weight
                        meta = res.get("meta") or {}
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
                        final_results.append(EngineHub._sanitize(res))
                    return final_results
                except Exception as e:
                    print(f"⚠️ decide_batch failed: {e}, falling back to sequential")
                    break

        # Fallback: 순차 실행
        return [self.decide(ctx) for ctx in ctx_list]

    # =========================
    # decision
    # =========================
    def decide(self, ctx: dict) -> dict:
        results = []

        for engine in self.engines:
            try:
                res = engine.decide(ctx)
                res["_engine"] = engine.name
                res["_weight"] = engine.weight
                # pass through event-based MC metrics
                meta = res.get("meta") or {}
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
                results.append({
                    "action": "WAIT",
                    "ev": 0.0,
                    "confidence": 0.0,
                    "reason": f"{engine.name} error: {e}",
                    "_engine": engine.name,
                    "_weight": engine.weight,
                })

        total_ev = sum(r["ev"] * r["_weight"] for r in results)
        best = max(results, key=lambda r: r["ev"])

        final_action = best["action"] if total_ev > 0 else "WAIT"

        final = {
            "action": final_action,
            "ev": total_ev,
            "confidence": max(r.get("confidence", 0.0) for r in results),
            "reason": " | ".join(r.get("reason", "") for r in results),
            "details": results,
        }

        # 🔥 최종 경계에서 무조건 sanitize
        return EngineHub._sanitize(final)
