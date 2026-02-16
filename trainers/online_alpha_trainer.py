"""
OnlineAlphaTrainer - Alpha Hit ML Module

Horizon별 TP/SL hit 확률을 예측하는 온라인 학습 MLP.
거래 결과 데이터를 수집하여 실시간으로 모델을 업데이트합니다.

Usage:
    from trainers.online_alpha_trainer import OnlineAlphaTrainer, AlphaTrainerConfig
    
    cfg = AlphaTrainerConfig(
        horizons_sec=[60, 300, 600, 1800, 3600],
        n_features=20,
        device="mps",
    )
    trainer = OnlineAlphaTrainer(cfg)
    
    # Prediction
    pred = trainer.predict(features_np)  # returns dict with p_tp_long, p_sl_long, etc.
    
    # Training (online)
    trainer.add_sample(x=features, y=labels, ts_ms=timestamp, symbol="BTCUSDT")
    stats = trainer.train_tick()
"""

from __future__ import annotations

import logging
import math
import os
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# PyTorch import
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except ImportError:
    torch = None
    nn = None
    F = None
    _TORCH_OK = False
    logger.warning("[ALPHA_TRAINER] PyTorch not available")


@dataclass
class AlphaTrainerConfig:
    """OnlineAlphaTrainer 설정"""
    horizons_sec: List[int] = field(default_factory=lambda: [60, 300, 600, 1800, 3600])
    n_features: int = 20
    hidden_dim: int = 256
    hidden_dim2: int = 128
    device: str = "cpu"
    lr: float = 2e-4
    lr_min: float = 1e-5  # minimum learning rate
    batch_size: int = 256
    steps_per_tick: int = 2
    max_buffer: int = 200000
    min_buffer: int = 1024
    data_half_life_sec: float = 3600.0
    ckpt_path: str = "state/alpha_hit_mlp.pt"
    enable: bool = True
    warmup_samples: int = 512  # warm up 전까지는 MC 결과만 사용
    
    # Advanced features
    label_smoothing: float = 0.05  # Label smoothing factor
    feature_normalize: bool = True  # Enable running mean/std normalization
    lr_warmup_steps: int = 500  # LR warmup steps
    weight_decay: float = 0.01  # AdamW weight decay
    grad_accum_steps: int = 1  # Gradient accumulation steps
    validation_frac: float = 0.1  # Fraction of buffer for validation
    replay_path: str = "state/alpha_hit_replay.npz"
    replay_save_every: int = 2000  # Save replay every N samples (0 to disable)


if _TORCH_OK:
    class RunningNormalizer:
        """Running mean/std normalizer for stable feature normalization"""
        
        def __init__(self, n_features: int, momentum: float = 0.01, device: str = "cpu"):
            self.n_features = n_features
            self.momentum = momentum
            self.device = torch.device(device)
            
            self.running_mean = torch.zeros(n_features, device=self.device)
            self.running_var = torch.ones(n_features, device=self.device)
            self._count = 0
        
        def update(self, x: torch.Tensor):
            """Update running statistics with new batch"""
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            
            if self._count == 0:
                self.running_mean = batch_mean
                self.running_var = batch_var
            else:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            self._count += x.shape[0]
        
        def normalize(self, x: torch.Tensor) -> torch.Tensor:
            """Normalize input with running statistics"""
            eps = 1e-6
            return (x - self.running_mean) / (torch.sqrt(self.running_var) + eps)
        
        def state_dict(self) -> Dict:
            return {
                "running_mean": self.running_mean.cpu().numpy(),
                "running_var": self.running_var.cpu().numpy(),
                "count": self._count,
            }
        
        def load_state_dict(self, state: Dict):
            self.running_mean = torch.from_numpy(state["running_mean"]).to(self.device)
            self.running_var = torch.from_numpy(state["running_var"]).to(self.device)
            self._count = state.get("count", 0)


    class AlphaHitMLP(nn.Module):
        """
        Multi-head MLP for TP/SL hit probability prediction.
        
        Output heads (각 horizon별):
        - p_tp_long: Long 포지션의 TP 도달 확률
        - p_sl_long: Long 포지션의 SL 도달 확률
        - p_tp_short: Short 포지션의 TP 도달 확률
        - p_sl_short: Short 포지션의 SL 도달 확률
        """
        
        def __init__(self, cfg: AlphaTrainerConfig):
            super().__init__()
            self.cfg = cfg
            n_horizons = len(cfg.horizons_sec)
            
            # Shared backbone with residual connection
            self.input_proj = nn.Linear(cfg.n_features, cfg.hidden_dim)
            self.backbone = nn.Sequential(
                nn.LayerNorm(cfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.LayerNorm(cfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            
            # Projection to hidden2
            self.hidden_proj = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim2),
                nn.LayerNorm(cfg.hidden_dim2),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            
            # Output heads (4 heads × n_horizons outputs each)
            self.head_tp_long = nn.Linear(cfg.hidden_dim2, n_horizons)
            self.head_sl_long = nn.Linear(cfg.hidden_dim2, n_horizons)
            self.head_tp_short = nn.Linear(cfg.hidden_dim2, n_horizons)
            self.head_sl_short = nn.Linear(cfg.hidden_dim2, n_horizons)
            
            self._init_weights()
        
        def _init_weights(self):
            """Xavier initialization for stable training"""
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            Forward pass.
            
            Args:
                x: [batch, n_features] input tensor
                
            Returns:
                Dict with keys: p_tp_long, p_sl_long, p_tp_short, p_sl_short
                Each value is [batch, n_horizons] tensor with probabilities in [0, 1]
            """
            # Input projection
            h = self.input_proj(x)
            
            # Backbone with residual
            h = h + self.backbone(h)
            
            # Project to output dimension
            h = self.hidden_proj(h)
            
            return {
                "p_tp_long": torch.sigmoid(self.head_tp_long(h)),
                "p_sl_long": torch.sigmoid(self.head_sl_long(h)),
                "p_tp_short": torch.sigmoid(self.head_tp_short(h)),
                "p_sl_short": torch.sigmoid(self.head_sl_short(h)),
            }
        
        def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            """Inference mode forward"""
            self.eval()
            with torch.no_grad():
                return self.forward(x)


class OnlineAlphaTrainer:
    """
    온라인 학습 및 예측을 위한 Trainer.
    
    핵심 기능:
    - Experience replay buffer로 최근 거래 데이터 저장
    - Exponential decay로 오래된 샘플 가중치 감소
    - train_tick()으로 점진적 학습
    
    고급 기능:
    - Running mean/std 기반 feature normalization
    - Learning rate warmup + cosine decay
    - Label smoothing for better calibration
    - Symbol-specific performance tracking
    - Calibration metrics (Brier score)
    """
    
    def __init__(self, cfg: AlphaTrainerConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device) if _TORCH_OK else None
        
        # Model
        if _TORCH_OK:
            self.model = AlphaHitMLP(cfg)
            self.model.to(self.device)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=cfg.lr,
                weight_decay=cfg.weight_decay
            )
            
            # Feature normalizer
            if cfg.feature_normalize:
                self.normalizer = RunningNormalizer(cfg.n_features, device=cfg.device)
            else:
                self.normalizer = None
        else:
            self.model = None
            self.optimizer = None
            self.normalizer = None
        
        # Experience buffer
        self._buffer_x: List[np.ndarray] = []
        self._buffer_y: List[Dict[str, np.ndarray]] = []
        self._buffer_ts: List[int] = []
        self._buffer_sym: List[str] = []
        
        # Stats
        self._total_samples = 0
        self._total_train_steps = 0
        self._last_loss = 0.0
        self._warmed_up = False
        
        # EMA loss tracking
        self._ema_loss = 0.0
        self._ema_alpha = 0.1
        
        # Symbol-specific statistics
        self._symbol_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            "samples": 0,
            "tp_long_rate": 0.0,
            "sl_long_rate": 0.0,
            "tp_short_rate": 0.0,
            "sl_short_rate": 0.0,
        })
        
        # Calibration metrics
        self._brier_scores: List[float] = []
        
        # Gradient accumulation
        self._grad_accum_counter = 0
        self._last_replay_save_n = 0
        self._train_lock = threading.Lock()
        
        # Load checkpoint if exists
        self._load_checkpoint()
        # Load replay buffer if available
        self._load_replay()
        
        logger.info(
            f"[ALPHA_TRAINER] Initialized | device={cfg.device} horizons={cfg.horizons_sec} "
            f"n_features={cfg.n_features} buffer_max={cfg.max_buffer} "
            f"label_smoothing={cfg.label_smoothing} feature_normalize={cfg.feature_normalize}"
        )

    def _coerce_horizon_vec(self, arr: Any) -> np.ndarray:
        """Normalize a label vector to current horizon length."""
        n_h = int(len(self.cfg.horizons_sec))
        try:
            out = np.asarray(arr, dtype=np.float32).reshape(-1)
        except Exception:
            out = np.zeros(n_h, dtype=np.float32)
        if out.size < n_h:
            pad = np.zeros(n_h - out.size, dtype=np.float32)
            out = np.concatenate([out, pad], axis=0)
        elif out.size > n_h:
            out = out[:n_h]
        return out.astype(np.float32, copy=False)

    def _coerce_feature_vec(self, x: Any) -> np.ndarray:
        """Normalize a feature vector to configured input dimension."""
        n_f = int(self.cfg.n_features)
        try:
            out = np.asarray(x, dtype=np.float32).reshape(-1)
        except Exception:
            out = np.zeros(n_f, dtype=np.float32)
        if out.size < n_f:
            pad = np.zeros(n_f - out.size, dtype=np.float32)
            out = np.concatenate([out, pad], axis=0)
        elif out.size > n_f:
            out = out[:n_f]
        return out.astype(np.float32, copy=False)

    def _is_mps_runtime_error(self, err_msg: str) -> bool:
        """Detect unstable MPS autograd/runtime failures and trigger CPU fallback."""
        e = str(err_msg or "").lower()
        mps_markers = (
            "mps",
            "inplace operation",
            "version",
            "autograd",
            "not implemented for 'mps'",
        )
        return any(m in e for m in mps_markers)

    def _switch_device(self, device: str) -> bool:
        """Move model to a safer device and reset optimizer/normalizer state."""
        if not _TORCH_OK or self.model is None:
            return False
        try:
            new_device = torch.device(device)
            if self.device is not None and self.device.type == new_device.type:
                return True
            self.device = new_device
            self.cfg.device = str(device)
            self.model.to(self.device)
            # Re-create optimizer on new device for stable continued training.
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self._get_lr(),
                weight_decay=self.cfg.weight_decay,
            )
            if self.cfg.feature_normalize:
                self.normalizer = RunningNormalizer(self.cfg.n_features, device=str(device))
            logger.warning(f"[ALPHA_TRAINER] Switched training device to {device} due to runtime instability")
            return True
        except Exception as e:
            logger.warning(f"[ALPHA_TRAINER] Failed to switch device to {device}: {e}")
            return False
    
    def _get_lr(self) -> float:
        """Get current learning rate with warmup and decay"""
        if self._total_train_steps < self.cfg.lr_warmup_steps:
            # Linear warmup
            return self.cfg.lr * (self._total_train_steps + 1) / self.cfg.lr_warmup_steps
        else:
            # Cosine decay
            progress = (self._total_train_steps - self.cfg.lr_warmup_steps) / max(1, 10000)
            return self.cfg.lr_min + 0.5 * (self.cfg.lr - self.cfg.lr_min) * (1 + math.cos(math.pi * min(progress, 1.0)))
    
    def _update_lr(self):
        """Update optimizer learning rate"""
        if self.optimizer is None:
            return
        new_lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
    
    def _smooth_labels(self, y: torch.Tensor) -> torch.Tensor:
        """Apply label smoothing"""
        eps = self.cfg.label_smoothing
        return y * (1 - eps) + 0.5 * eps
    
    def predict(self, features: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Predict TP/SL probabilities for given features.
        
        Args:
            features: [1, n_features] or [n_features] numpy array
            
        Returns:
            Dict with p_tp_long, p_sl_long, p_tp_short, p_sl_short tensors
        """
        if not _TORCH_OK or self.model is None:
            return self._fallback_prediction()
        
        try:
            x = np.asarray(features, dtype=np.float32)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            
            # Pad/truncate to n_features
            if x.shape[1] < self.cfg.n_features:
                pad = np.zeros((x.shape[0], self.cfg.n_features - x.shape[1]), dtype=np.float32)
                x = np.concatenate([x, pad], axis=1)
            elif x.shape[1] > self.cfg.n_features:
                x = x[:, :self.cfg.n_features]
            
            x_t = torch.from_numpy(x).to(self.device)
            
            # Apply normalization
            if self.normalizer is not None and self.normalizer._count > 0:
                x_t = self.normalizer.normalize(x_t)
            
            return self.model.predict(x_t)
        except Exception as e:
            logger.warning(f"[ALPHA_TRAINER] Prediction failed: {e}")
            return self._fallback_prediction()
    
    def _fallback_prediction(self) -> Dict[str, torch.Tensor]:
        """Uniform 0.5 fallback when model unavailable"""
        n_h = len(self.cfg.horizons_sec)
        dummy = torch.ones(1, n_h) * 0.5
        return {
            "p_tp_long": dummy,
            "p_sl_long": dummy,
            "p_tp_short": dummy,
            "p_sl_short": dummy,
        }
    
    def add_sample(
        self,
        x: np.ndarray,
        y: Dict[str, np.ndarray],
        ts_ms: int,
        symbol: str,
    ):
        """
        Add training sample to buffer.
        
        Args:
            x: [n_features] feature vector
            y: Dict with tp_long, sl_long, tp_short, sl_short arrays (each [n_horizons])
            ts_ms: Timestamp in milliseconds
            symbol: Trading pair symbol
        """
        try:
            x_arr = self._coerce_feature_vec(x)
            
            # Validate y
            y_clean = {}
            for key in ["tp_long", "sl_long", "tp_short", "sl_short"]:
                y_clean[key] = self._coerce_horizon_vec(y.get(key))
            
            self._buffer_x.append(x_arr)
            self._buffer_y.append(y_clean)
            self._buffer_ts.append(int(ts_ms))
            self._buffer_sym.append(str(symbol))
            self._total_samples += 1
            
            # Update symbol statistics
            sym_stats = self._symbol_stats[symbol]
            sym_stats["samples"] += 1
            # EMA update for rates
            alpha = 0.01
            sym_stats["tp_long_rate"] = (1 - alpha) * sym_stats["tp_long_rate"] + alpha * float(y_clean["tp_long"].max())
            sym_stats["sl_long_rate"] = (1 - alpha) * sym_stats["sl_long_rate"] + alpha * float(y_clean["sl_long"].max())
            sym_stats["tp_short_rate"] = (1 - alpha) * sym_stats["tp_short_rate"] + alpha * float(y_clean["tp_short"].max())
            sym_stats["sl_short_rate"] = (1 - alpha) * sym_stats["sl_short_rate"] + alpha * float(y_clean["sl_short"].max())
            
            # Trim buffer if needed
            while len(self._buffer_x) > self.cfg.max_buffer:
                self._buffer_x.pop(0)
                self._buffer_y.pop(0)
                self._buffer_ts.pop(0)
                self._buffer_sym.pop(0)
            
            # Check warmup
            if not self._warmed_up and len(self._buffer_x) >= self.cfg.warmup_samples:
                self._warmed_up = True
                logger.info(f"[ALPHA_TRAINER] Warmup complete with {len(self._buffer_x)} samples")

            # Periodic replay save
            self._maybe_save_replay()
                
        except Exception as e:
            logger.warning(f"[ALPHA_TRAINER] Failed to add sample: {e}")

    def _maybe_save_replay(self):
        if int(self.cfg.replay_save_every or 0) <= 0:
            return
        n = len(self._buffer_x)
        if n <= 0:
            return
        if n - self._last_replay_save_n < int(self.cfg.replay_save_every):
            return
        self._last_replay_save_n = n
        self._save_replay()
    
    def train_tick(self, _retry_on_device_fail: bool = True) -> Dict[str, float]:
        """
        Perform training step(s).
        
        Returns:
            Dict with training stats (loss, n_samples, etc.)
        """
        if not _TORCH_OK or self.model is None:
            return {"loss": 0.0, "n_samples": 0, "skipped": True}
        
        n = len(self._buffer_x)
        if n < self.cfg.min_buffer:
            return {"loss": 0.0, "n_samples": n, "skipped": True, "reason": "insufficient_samples"}

        # Guard against concurrent optimizer updates from async engine loops.
        if not self._train_lock.acquire(blocking=False):
            return {"loss": self._last_loss, "n_samples": n, "skipped": True, "reason": "trainer_busy"}
        
        try:
            self.model.train()
            total_loss = 0.0
            total_brier = 0.0
            
            for step in range(self.cfg.steps_per_tick):
                # Update learning rate
                self._update_lr()
                
                # Sample batch with recency weighting
                indices = self._sample_indices_with_decay(self.cfg.batch_size)
                
                # Build batch
                x_batch = np.stack([self._buffer_x[i] for i in indices], axis=0)
                
                # Pad/truncate features
                if x_batch.shape[1] < self.cfg.n_features:
                    pad = np.zeros((x_batch.shape[0], self.cfg.n_features - x_batch.shape[1]), dtype=np.float32)
                    x_batch = np.concatenate([x_batch, pad], axis=1)
                elif x_batch.shape[1] > self.cfg.n_features:
                    x_batch = x_batch[:, :self.cfg.n_features]
                
                y_tp_long = np.stack([self._buffer_y[i]["tp_long"] for i in indices], axis=0)
                y_sl_long = np.stack([self._buffer_y[i]["sl_long"] for i in indices], axis=0)
                y_tp_short = np.stack([self._buffer_y[i]["tp_short"] for i in indices], axis=0)
                y_sl_short = np.stack([self._buffer_y[i]["sl_short"] for i in indices], axis=0)
                
                # To tensors
                x_t = torch.from_numpy(x_batch).to(self.device)
                y_tp_long_t = torch.from_numpy(y_tp_long).to(self.device)
                y_sl_long_t = torch.from_numpy(y_sl_long).to(self.device)
                y_tp_short_t = torch.from_numpy(y_tp_short).to(self.device)
                y_sl_short_t = torch.from_numpy(y_sl_short).to(self.device)
                
                # Update normalizer and normalize
                if self.normalizer is not None:
                    self.normalizer.update(x_t)
                    x_t = self.normalizer.normalize(x_t)
                
                # Apply label smoothing
                y_tp_long_t = self._smooth_labels(y_tp_long_t)
                y_sl_long_t = self._smooth_labels(y_sl_long_t)
                y_tp_short_t = self._smooth_labels(y_tp_short_t)
                y_sl_short_t = self._smooth_labels(y_sl_short_t)
                
                # Forward
                pred = self.model(x_t)
                
                # BCE loss
                loss = (
                    F.binary_cross_entropy(pred["p_tp_long"], y_tp_long_t) +
                    F.binary_cross_entropy(pred["p_sl_long"], y_sl_long_t) +
                    F.binary_cross_entropy(pred["p_tp_short"], y_tp_short_t) +
                    F.binary_cross_entropy(pred["p_sl_short"], y_sl_short_t)
                ) / 4.0
                
                # Gradient accumulation
                loss = loss / self.cfg.grad_accum_steps
                loss.backward()
                
                self._grad_accum_counter += 1
                if self._grad_accum_counter >= self.cfg.grad_accum_steps:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self._grad_accum_counter = 0
                
                total_loss += float(loss.item()) * self.cfg.grad_accum_steps
                self._total_train_steps += 1
                
                # Calculate Brier score (calibration metric)
                with torch.no_grad():
                    # Use original labels for Brier score
                    y_orig = torch.from_numpy(np.stack([self._buffer_y[i]["tp_long"] for i in indices], axis=0)).to(self.device)
                    brier = ((pred["p_tp_long"] - y_orig) ** 2).mean().item()
                    total_brier += brier
            
            avg_loss = total_loss / self.cfg.steps_per_tick
            avg_brier = total_brier / self.cfg.steps_per_tick
            self._last_loss = avg_loss
            
            # Update EMA loss
            self._ema_loss = (1 - self._ema_alpha) * self._ema_loss + self._ema_alpha * avg_loss
            
            # Track Brier scores
            self._brier_scores.append(avg_brier)
            if len(self._brier_scores) > 100:
                self._brier_scores.pop(0)
            
            # Periodic checkpoint
            if self._total_train_steps % 1000 == 0:
                self._save_checkpoint()
                logger.info(
                    f"[ALPHA_TRAINER] Step {self._total_train_steps} | "
                    f"loss={avg_loss:.4f} ema_loss={self._ema_loss:.4f} "
                    f"brier={avg_brier:.4f} lr={self._get_lr():.2e}"
                )
            
            return {
                "loss": avg_loss,
                "ema_loss": self._ema_loss,
                "brier_score": avg_brier,
                "lr": self._get_lr(),
                "n_samples": n,
                "train_steps": self._total_train_steps,
                "skipped": False,
            }
            
        except Exception as e:
            err = str(e)
            if (
                _retry_on_device_fail
                and self.device is not None
                and self.device.type == "mps"
                and self._is_mps_runtime_error(err)
                and self._switch_device("cpu")
            ):
                logger.warning("[ALPHA_TRAINER] CPU fallback armed; training will resume on next tick")
                return {"loss": self._last_loss, "n_samples": n, "skipped": True, "reason": "device_fallback_switched"}
            logger.warning(f"[ALPHA_TRAINER] Training failed: {err}")
            return {"loss": 0.0, "n_samples": n, "skipped": True, "error": err}
        finally:
            try:
                self._train_lock.release()
            except RuntimeError:
                pass
    
    def _sample_indices_with_decay(self, batch_size: int) -> List[int]:
        """Sample indices with exponential recency weighting"""
        n = len(self._buffer_x)
        if n == 0:
            return []
        
        now_ms = int(time.time() * 1000)
        half_life_ms = self.cfg.data_half_life_sec * 1000
        
        # Calculate weights
        weights = np.zeros(n, dtype=np.float64)
        for i in range(n):
            age_ms = now_ms - self._buffer_ts[i]
            decay = math.exp(-0.693 * age_ms / half_life_ms)  # ln(2) ≈ 0.693
            weights[i] = max(decay, 0.01)  # minimum weight
        
        weights /= weights.sum()
        
        # Sample
        indices = np.random.choice(n, size=min(batch_size, n), replace=False, p=weights)
        return indices.tolist()
    
    def get_symbol_stats(self, symbol: str) -> Dict[str, float]:
        """Get statistics for a specific symbol"""
        return dict(self._symbol_stats.get(symbol, {}))
    
    def get_calibration_stats(self) -> Dict[str, float]:
        """Get calibration statistics"""
        if not self._brier_scores:
            return {"mean_brier": 0.0, "min_brier": 0.0, "max_brier": 0.0}
        
        return {
            "mean_brier": float(np.mean(self._brier_scores)),
            "min_brier": float(np.min(self._brier_scores)),
            "max_brier": float(np.max(self._brier_scores)),
            "recent_brier": float(self._brier_scores[-1]) if self._brier_scores else 0.0,
        }
    
    def _save_checkpoint(self):
        """Save model checkpoint"""
        if not _TORCH_OK or self.model is None:
            return
        
        try:
            ckpt_dir = os.path.dirname(self.cfg.ckpt_path)
            if ckpt_dir and not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir, exist_ok=True)
            
            ckpt = {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "total_samples": self._total_samples,
                "total_train_steps": self._total_train_steps,
                "ema_loss": self._ema_loss,
                "config": {
                    "horizons_sec": self.cfg.horizons_sec,
                    "n_features": self.cfg.n_features,
                },
            }
            
            # Save normalizer state
            if self.normalizer is not None:
                ckpt["normalizer_state"] = self.normalizer.state_dict()
            
            torch.save(ckpt, self.cfg.ckpt_path)
            logger.info(f"[ALPHA_TRAINER] Checkpoint saved: {self.cfg.ckpt_path}")
        except Exception as e:
            logger.warning(f"[ALPHA_TRAINER] Failed to save checkpoint: {e}")

    def _save_replay(self):
        """Persist replay buffer for warm start across runs."""
        try:
            rp = str(self.cfg.replay_path or "").strip()
            if not rp:
                return
            rp_dir = os.path.dirname(rp)
            if rp_dir and not os.path.exists(rp_dir):
                os.makedirs(rp_dir, exist_ok=True)

            if not self._buffer_x:
                return
            x = np.asarray(self._buffer_x, dtype=np.float32)
            y_tp_long = np.stack([y["tp_long"] for y in self._buffer_y], axis=0)
            y_sl_long = np.stack([y["sl_long"] for y in self._buffer_y], axis=0)
            y_tp_short = np.stack([y["tp_short"] for y in self._buffer_y], axis=0)
            y_sl_short = np.stack([y["sl_short"] for y in self._buffer_y], axis=0)
            ts = np.asarray(self._buffer_ts, dtype=np.int64)
            sym = np.asarray(self._buffer_sym, dtype="U")

            np.savez_compressed(
                rp,
                x=x,
                y_tp_long=y_tp_long,
                y_sl_long=y_sl_long,
                y_tp_short=y_tp_short,
                y_sl_short=y_sl_short,
                ts=ts,
                sym=sym,
                total_samples=int(self._total_samples),
            )
            logger.info(f"[ALPHA_TRAINER] Replay saved: {rp} (n={len(self._buffer_x)})")
        except Exception as e:
            logger.warning(f"[ALPHA_TRAINER] Failed to save replay: {e}")

    def _load_replay(self):
        """Load replay buffer if present."""
        try:
            rp = str(self.cfg.replay_path or "").strip()
            if not rp or (not os.path.exists(rp)):
                return
            data = np.load(rp, allow_pickle=False)
            x = data.get("x")
            if x is None:
                return
            y_tp_long = data.get("y_tp_long")
            y_sl_long = data.get("y_sl_long")
            y_tp_short = data.get("y_tp_short")
            y_sl_short = data.get("y_sl_short")
            ts = data.get("ts")
            sym = data.get("sym")
            if any(v is None for v in (y_tp_long, y_sl_long, y_tp_short, y_sl_short, ts, sym)):
                return

            self._buffer_x = [self._coerce_feature_vec(v) for v in x]
            self._buffer_y = []
            for i in range(len(self._buffer_x)):
                self._buffer_y.append(
                    {
                        "tp_long": self._coerce_horizon_vec(y_tp_long[i]),
                        "sl_long": self._coerce_horizon_vec(y_sl_long[i]),
                        "tp_short": self._coerce_horizon_vec(y_tp_short[i]),
                        "sl_short": self._coerce_horizon_vec(y_sl_short[i]),
                    }
                )
            self._buffer_ts = [int(v) for v in ts.tolist()]
            self._buffer_sym = [str(v) for v in sym.tolist()]
            self._total_samples = int(data.get("total_samples") or len(self._buffer_x))
            self._last_replay_save_n = len(self._buffer_x)
            logger.info(f"[ALPHA_TRAINER] Replay loaded: {rp} (n={len(self._buffer_x)})")
        except Exception as e:
            logger.warning(f"[ALPHA_TRAINER] Failed to load replay: {e}")
    
    def _load_checkpoint(self):
        """Load model checkpoint if exists"""
        if not _TORCH_OK or self.model is None:
            return
        
        if not os.path.exists(self.cfg.ckpt_path):
            logger.info(f"[ALPHA_TRAINER] No checkpoint found at {self.cfg.ckpt_path}")
            return
        
        try:
            ckpt = torch.load(self.cfg.ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state"])
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self._total_samples = ckpt.get("total_samples", 0)
            self._total_train_steps = ckpt.get("total_train_steps", 0)
            self._ema_loss = ckpt.get("ema_loss", 0.0)
            
            # Load normalizer state
            if self.normalizer is not None and "normalizer_state" in ckpt:
                self.normalizer.load_state_dict(ckpt["normalizer_state"])
            
            logger.info(
                f"[ALPHA_TRAINER] Checkpoint loaded: {self.cfg.ckpt_path} "
                f"(samples={self._total_samples}, steps={self._total_train_steps})"
            )
        except Exception as e:
            logger.warning(f"[ALPHA_TRAINER] Failed to load checkpoint: {e}")
    
    @property
    def is_warmed_up(self) -> bool:
        """Whether trainer has enough samples for reliable predictions"""
        return self._warmed_up
    
    @property
    def buffer_size(self) -> int:
        """Current buffer size"""
        return len(self._buffer_x)
    
    @property
    def last_loss(self) -> float:
        """Last training loss"""
        return self._last_loss
    
    @property
    def ema_loss(self) -> float:
        """Exponential moving average of loss"""
        return self._ema_loss
