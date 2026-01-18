from __future__ import annotations

import os
import math
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def _pick_device(pref: str = "") -> torch.device:
    pref = (pref or "").strip().lower()
    if pref in ("cuda", "gpu") and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class HazardMLP(nn.Module):
    """
    MLP that outputs per-time-step hazard logits for discrete-time survival.
    Given features x -> logits[t] for t=0..T-1.
    hazard[t] = sigmoid(logit[t]) in (0,1)
    """

    def __init__(self, in_dim: int, T: int, hidden: int = 64, depth: int = 2, dropout: float = 0.05):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(max(1, int(depth))):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, T))  # output logits per step
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, T)


@dataclass
class PMakerSurvivalMLP:
    """
    Online + replay-trained survival model for maker fill time.

    Discrete-time grid model:
      - time step = grid_ms
      - max horizon = max_ms
      - T = ceil(max_ms / grid_ms)

    Labels per attempt:
      - first_fill_delay_ms: if filled observed => event at idx = ceil(t/grid_ms)-1
      - else censored at timeout => no event within N steps

    Training uses negative log-likelihood for discrete-time survival with censoring.
    """

    grid_ms: int = 50
    max_ms: int = 2500
    lr: float = 3e-4
    weight_decay: float = 1e-4
    hidden: int = 96
    depth: int = 2
    dropout: float = 0.05
    device: str = ""
    clip_feat: float = 8.0

    # replay buffer
    replay_cap: int = 20000
    replay: List[Dict[str, Any]] = field(default_factory=list)

    # symbol smoothing (wins can be fractional via fill fraction)
    sym_wins: Dict[str, float] = field(default_factory=dict)
    sym_n: Dict[str, float] = field(default_factory=dict)
    # ✅ 매우 보수적인 베이지안 prior: alpha0=0.5, beta0=4.0 → 기본 fill rate 11% (매우 conservative)
    alpha0: float = 0.5
    beta0: float = 4.0
    # ✅ blend_lambda를 매우 낮춰서 모델 예측에 거의 의존 (심볼 통계 의존도 최소화)
    blend_lambda: float = 0.10

    def __post_init__(self):
        self._dev = _pick_device(self.device)
        self.device_str = str(self._dev)
        self.T = int(math.ceil(self.max_ms / max(1, int(self.grid_ms))))
        self.in_dim = 7  # fixed featurize dim below
        self.model = HazardMLP(self.in_dim, self.T, hidden=self.hidden, depth=self.depth, dropout=self.dropout).to(self._dev)
        self.opt = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self._step = 0

    # -------- features --------
    def featurize(self, feats: Dict[str, Any]) -> torch.Tensor:
        """
        Stable features (small set):
          spread_pct, sigma, |ofi|, |mom|, liq_score, attempt_idx, rel_px
        """

        def cz(v: Optional[float]) -> float:
            if v is None:
                return 0.0
            v = float(v)
            return max(-self.clip_feat, min(self.clip_feat, v))

        spread = feats.get("spread_pct")
        sigma = feats.get("sigma")
        ofi = feats.get("ofi_z")
        mom = feats.get("momentum_z")
        liq = feats.get("liq_score")
        attempt = feats.get("attempt_idx", 0)
        rel_px = feats.get("rel_px", 0.0)

        # scaling to similar magnitudes
        x = [
            cz(spread) * 10.0,
            cz(sigma) * 1000.0,
            abs(cz(ofi)),
            abs(cz(mom)),
            cz(liq) * 0.5,
            cz(attempt) * 0.25,
            cz(rel_px) * 10.0,
        ]
        return torch.tensor(x, dtype=torch.float32)

    def _n_steps(self, timeout_ms: int) -> int:
        # Clamp to model horizon T; callers may pass a longer timeout than max_ms.
        n = int(math.ceil(int(timeout_ms) / max(1, int(self.grid_ms))))
        return int(max(1, min(self.T, n)))

    def _event_index(self, first_fill_delay_ms: Optional[int], timeout_ms: int) -> Optional[int]:
        if first_fill_delay_ms is None:
            return None
        t = int(first_fill_delay_ms)
        if t <= 0:
            return 0
        # Clamp the event time to both the observation window and the model horizon.
        t = min(t, int(timeout_ms), int(self.max_ms))
        idx = int(math.ceil(t / max(1, int(self.grid_ms)))) - 1
        idx = max(0, min(self._n_steps(timeout_ms) - 1, idx, self.T - 1))
        return idx

    # -------- likelihood --------
    @staticmethod
    def _safe_log(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=eps, max=1.0))

    def _nll_one(self, logits: torch.Tensor, event_idx: Optional[int], n_steps: int) -> torch.Tensor:
        """
        logits: (T,)
        event_idx: int if event occurred, else None (censored)
        n_steps: number of steps within timeout (<=T)

        Discrete-time survival:
          hazard h_t = sigmoid(logit_t)
          P(event at k) = prod_{t<k}(1-h_t) * h_k
          P(censored) = prod_{t<n_steps}(1-h_t)
        """
        n_steps = int(max(1, min(int(n_steps), int(self.T), int(logits.shape[0]))))
        logits = logits[:n_steps]
        h = torch.sigmoid(logits)
        # log(1-h) for survival
        log_surv = self._safe_log(1.0 - h)

        if event_idx is None:
            # censored at timeout: sum log(1-h_t)
            return -torch.sum(log_surv)
        k = int(event_idx)
        k = max(0, min(n_steps - 1, k))
        # -[ sum_{t<k} log(1-h_t) + log(h_k) ]
        return -(torch.sum(log_surv[:k]) + self._safe_log(h[k]))

    # -------- predict --------
    @torch.no_grad()
    def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Return logits for a single input x (feat_dim,).
        Returns numpy array (T,).
        """
        self.model.eval()
        
        # Convert numpy to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
            
        # Ensure input is on correct device and has batch dim if needed
        x_in = x.to(self._dev)
        if x_in.dim() == 1:
            x_in = x_in.unsqueeze(0)
        
        logits = self.model(x_in).squeeze(0) # (T,)
        return logits.cpu().numpy()

    @torch.no_grad()
    def predict_F_Emin(self, x: torch.Tensor, timeout_ms: int) -> Tuple[float, float]:
        """
        Exact grid evaluation (no midpoint): E[min(T,timeout)] and P(fill<=timeout).
        """
        self.model.eval()
        n_steps = self._n_steps(timeout_ms)
        logits = self.model(x.to(self._dev).unsqueeze(0)).squeeze(0)  # (T,)
        logits = logits[:n_steps]
        h = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)

        S = 1.0
        e_ms = 0.0
        for i in range(n_steps):
            hi = float(h[i].item())
            p_i = S * hi
            t_i = (i + 1) * self.grid_ms
            e_ms += p_i * t_i
            S *= (1.0 - hi)
        e_ms += S * float(timeout_ms)
        p_fill = 1.0 - S
        return float(p_fill), float(e_ms / 1000.0)

    def predict_retry_comp(self, xs: List[torch.Tensor], timeout_ms: int) -> Dict[str, float]:
        """
        Sequential attempts composition (exact on grid):
          e_total = sum_{attempt i} P(reach i) * E[min(T_i, tau)]
          p_total = 1 - prod_i (1 - p_i)
        xs: list of x per attempt (attempt-specific features allowed)
        """
        p_fail = 1.0
        e_total = 0.0
        for x in xs:
            p1, e1 = self.predict_F_Emin(x, timeout_ms)
            e_total += p_fail * e1
            p_fail *= (1.0 - p1)
        p_total_raw = 1.0 - p_fail
        return {"p_total": float(p_total_raw), "p_total_raw": float(p_total_raw), "e_total_sec": float(e_total)}

    # -------- update / replay --------
    def update_one_attempt(
        self,
        sym: str,
        x: torch.Tensor,
        timeout_ms: int,
        first_fill_delay_ms: Optional[int],
        *,
        qty_attempt: Optional[float] = None,
        qty_filled: Optional[float] = None,
    ) -> None:
        """
        Add one attempt to replay buffer (and optionally do one SGD step right away).

        qty_attempt/qty_filled (optional):
          Used to update symbol smoothing with fill fraction for partial fills.
        """
        ev_idx = self._event_index(first_fill_delay_ms, timeout_ms)
        rec = {
            "sym": sym,
            "x": x.detach().cpu(),
            "timeout_ms": int(timeout_ms),
            "event_idx": ev_idx,
            "qty_attempt": float(qty_attempt) if qty_attempt is not None else None,
            "qty_filled": float(qty_filled) if qty_filled is not None else None,
        }
        self.replay.append(rec)
        if len(self.replay) > int(self.replay_cap):
            self.replay = self.replay[-int(self.replay_cap) :]

        # update symbol smoothing for P(fill<=timeout), optionally fractional by fill fraction
        if qty_attempt is not None and qty_attempt > 0 and qty_filled is not None:
            y_any = float(max(0.0, min(1.0, float(qty_filled) / float(qty_attempt))))
        else:
            y_any = 1.0 if (first_fill_delay_ms is not None and first_fill_delay_ms <= timeout_ms) else 0.0
        self.sym_n[sym] = self.sym_n.get(sym, 0.0) + 1.0
        self.sym_wins[sym] = self.sym_wins.get(sym, 0.0) + float(y_any)

        # optional immediate tiny step (keeps it learning even with small replay)
        self._train_batch([rec])
        # 타임스탬프 업데이트 (디버깅용)
        import time
        self._last_update_time_ms = int(time.time() * 1000)

    def _train_batch(self, batch: List[Dict[str, Any]]) -> float:
        self.model.train()
        xs = torch.stack([b["x"] for b in batch], dim=0).to(self._dev)  # (B, in_dim)
        logits = self.model(xs)  # (B, T)
        losses = []
        for i, b in enumerate(batch):
            n_steps = self._n_steps(int(b["timeout_ms"]))
            ev = b["event_idx"]
            losses.append(self._nll_one(logits[i], ev, n_steps))
        loss = torch.stack(losses).mean()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        self._step += 1
        return float(loss.item())

    def train_from_replay(self, steps: int = 2, batch_size: int = 64) -> None:
        if not self.replay:
            return
        bs = max(1, int(batch_size))
        for _ in range(max(0, int(steps))):
            batch = random.sample(self.replay, k=min(bs, len(self.replay)))
            self._train_batch(batch)
        # 타임스탬프 업데이트 (디버깅용)
        import time
        self._last_train_time_ms = int(time.time() * 1000)

    # -------- symbol blending --------
    def sym_fill_mean(self, sym: str) -> float:
        # ✅ 필터 없이 모든 데이터 사용 (A/F 결과가 누적된 데이터)
        n = self.sym_n.get(sym, 0.0)
        w = self.sym_wins.get(sym, 0.0)
        return (w + self.alpha0) / (n + self.alpha0 + self.beta0)

    def blend(self, sym: str, p: float, conservative: bool = True) -> float:
        """
        Blend global p with symbol prior. Conservative uses a lower-ish bound.
        """
        p = float(p)
        n = self.sym_n.get(sym, 0.0) + self.alpha0 + self.beta0
        p_sym = self.sym_fill_mean(sym)

        # ✅ 필터 없이 실제 데이터 사용 (A/F 결과가 누적되어 S로 계산됨)
        # conservative 파라미터는 유지하되, 실제 데이터를 그대로 사용
        if conservative:
            # 표준편차 기반 보정 (통계적으로 합리적)
            var = p_sym * (1 - p_sym) / max(1.0, n)
            p_sym = max(0.0, p_sym - 1.0 * math.sqrt(var))

        lam = float(self.blend_lambda)
        # 샘플이 적을 때는 심볼 통계에 더 의존
        if n < 50:
            lam = min(0.40, lam + 0.20 * (1.0 - n / 50.0))

        out = (1.0 - lam) * p + lam * p_sym
        return max(0.001, min(0.999, out))

    # -------- persistence --------
    def save(self, path: str) -> None:
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        model_state = self.model.state_dict()
        opt_state = self.opt.state_dict()
        replay_tail = self.replay[-2000:]  # keep last 2k to avoid huge file
        # New explicit checkpoint format (v2) + legacy keys for backward compatibility.
        payload = {
            "format_version": 2,
            "meta": {
                "grid_ms": self.grid_ms,
                "max_ms": self.max_ms,
                "T": self.T,
                "in_dim": self.in_dim,
            },
            "model_state": model_state,
            "opt_state": opt_state,
            "stats": {
                "sym_wins": dict(self.sym_wins),  # ✅ dict()로 복사하여 참조 문제 방지
                "sym_n": dict(self.sym_n),  # ✅ dict()로 복사하여 참조 문제 방지
                "alpha0": self.alpha0,
                "beta0": self.beta0,
                "blend_lambda": self.blend_lambda,
                "_trimmed_200": getattr(self, "_trimmed_200", False),  # ✅ 한 번만 제거했는지 플래그 저장
            },
            "replay": replay_tail,
            # legacy (v1) keys
            "grid_ms": self.grid_ms,
            "max_ms": self.max_ms,
            "T": self.T,
            "in_dim": self.in_dim,
            "model": model_state,
            "opt": opt_state,
            "sym_wins": dict(self.sym_wins),  # ✅ dict()로 복사하여 참조 문제 방지
            "sym_n": dict(self.sym_n),  # ✅ dict()로 복사하여 참조 문제 방지
            "alpha0": self.alpha0,
            "beta0": self.beta0,
            "blend_lambda": self.blend_lambda,
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        if not path or not os.path.exists(path):
            return
        payload = torch.load(path, map_location="cpu")
        # accept if compatible
        sd = payload.get("model_state")
        if sd is None:
            sd = payload.get("model")
        if sd:
            self.model.load_state_dict(sd, strict=False)
        od = payload.get("opt_state")
        if od is None:
            od = payload.get("opt")
        if od:
            try:
                self.opt.load_state_dict(od)
            except Exception:
                pass
        stats = payload.get("stats") or {}
        # ✅ 필터 없이 모든 데이터 로드 (dict()로 복사하여 참조 문제 방지)
        sym_wins_loaded = dict(stats.get("sym_wins") or payload.get("sym_wins") or {})
        sym_n_loaded = dict(stats.get("sym_n") or payload.get("sym_n") or {})
        
        # ✅ 디버깅: 로드된 원본 데이터 확인
        total_n_loaded_raw = int(round(sum(float(v) for v in sym_n_loaded.values())))
        total_w_loaded_raw = int(round(sum(float(v) for v in sym_wins_loaded.values())))
        print(f"[PMAKER_LOAD] Raw loaded from file: sym_n total={total_n_loaded_raw} sym_wins total={total_w_loaded_raw} sym_count={len(sym_n_loaded)}")
        
        # ✅ 디버깅: 로드된 데이터 확인
        total_n_loaded = int(round(sum(float(v) for v in sym_n_loaded.values())))
        total_w_loaded = int(round(sum(float(v) for v in sym_wins_loaded.values())))
        print(f"[PMAKER_LOAD] Loaded from file: sym_n total={total_n_loaded} sym_wins total={total_w_loaded} sym_count={len(sym_n_loaded)}")
        
        # ✅ 이번 한 번만 앞의 200개 제거 (플래그 확인)
        already_trimmed = payload.get("_trimmed_200", False) or stats.get("_trimmed_200", False)
        
        if not already_trimmed:
            # 전체 attempts 합계 계산
            total_attempts_loaded = int(round(sum(float(v) for v in sym_n_loaded.values())))
            total_fills_loaded = int(round(sum(float(v) for v in sym_wins_loaded.values())))
            
            # ✅ 앞의 200개 정도 제거 (비율 유지하면서 제거)
            # 단, 전체 데이터가 200개 이하면 제거하지 않음 (데이터 손실 방지)
            remove_count = min(200, total_attempts_loaded) if total_attempts_loaded > 200 else 0
            if remove_count > 0 and total_attempts_loaded > 200:
                # 전체 fill rate 계산
                overall_fill_rate = float(total_fills_loaded) / float(total_attempts_loaded) if total_attempts_loaded > 0 else 0.0
                
                # 각 심볼별로 비율에 맞게 제거
                self.sym_n = {}
                self.sym_wins = {}
                
                for sym in sym_n_loaded:
                    sym_attempts = float(sym_n_loaded.get(sym, 0.0))
                    sym_fills = float(sym_wins_loaded.get(sym, 0.0))
                    
                    if total_attempts_loaded > 0:
                        # 심볼별 비율 계산
                        sym_ratio = sym_attempts / float(total_attempts_loaded)
                        # 제거할 양 계산
                        remove_attempts = sym_ratio * remove_count
                        remove_fills = sym_ratio * remove_count * overall_fill_rate
                        
                        # 제거 후 남은 값
                        new_attempts = max(0.0, sym_attempts - remove_attempts)
                        new_fills = max(0.0, sym_fills - remove_fills)
                        
                        if new_attempts > 0:
                            self.sym_n[sym] = new_attempts
                            self.sym_wins[sym] = new_fills
                
                # ✅ 플래그 설정하여 다음 로드부터는 제거하지 않음
                self._trimmed_200 = True
                # 즉시 저장하여 플래그를 영구적으로 기록
                try:
                    self.save(path)
                except Exception:
                    pass
            else:
                # 제거할 것이 없으면 그대로 사용
                self.sym_wins = sym_wins_loaded
                self.sym_n = sym_n_loaded
                self._trimmed_200 = True
        else:
            # 이미 제거했으면 그대로 사용
            self.sym_wins = sym_wins_loaded
            self.sym_n = sym_n_loaded
        
        # ✅ 최종 로드된 데이터 확인
        total_n_final = int(round(sum(float(v) for v in self.sym_n.values())))
        total_w_final = int(round(sum(float(v) for v in self.sym_wins.values())))
        print(f"[PMAKER_LOAD] Final after processing: sym_n total={total_n_final} sym_wins total={total_w_final} sym_count={len(self.sym_n)}")
        if "alpha0" in stats or "alpha0" in payload:
            try:
                self.alpha0 = float(stats.get("alpha0", payload.get("alpha0", self.alpha0)))
            except Exception:
                pass
        if "beta0" in stats or "beta0" in payload:
            try:
                self.beta0 = float(stats.get("beta0", payload.get("beta0", self.beta0)))
            except Exception:
                pass
        if "blend_lambda" in stats or "blend_lambda" in payload:
            try:
                self.blend_lambda = float(stats.get("blend_lambda", payload.get("blend_lambda", self.blend_lambda)))
            except Exception:
                pass
        self.replay = list(payload.get("replay") or [])
