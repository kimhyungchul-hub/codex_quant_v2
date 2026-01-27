# mc_plus.py - PyTorch-first with NumPy fallback

import bootstrap  # ensure environment vars are set

import numpy as np
from typing import Dict, List, Tuple

# Import PyTorch-first backend
from engines.mc.jax_backend import _TORCH_OK, torch, to_torch, to_numpy, get_device

class KalmanFilter1D:
    def __init__(self, R=0.01, Q=1e-5):
        self.R = R
        self.Q = Q
        self.P = 1.0
        self.x = None
        
    def update(self, measurement):
        if self.x is None:
            self.x = measurement
            return self.x
        x_pred = self.x
        P_pred = self.P + self.Q
        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred
        return self.x

class OUProcess:
    def __init__(self, window=20):
        self.window = window
    def get_z_score(self, prices):
        if len(prices) < self.window: return 0.0
        log_prices = np.log(prices[-self.window:])
        mu = np.mean(log_prices)
        sigma = np.std(log_prices)
        if sigma < 1e-9: return 0.0
        return (log_prices[-1] - mu) / sigma

class LSMModel:
    @staticmethod
    def calculate_values(paths, entry_price, direction, leverage, discount=0.9999):
        """
        Least Squares Monte Carlo valuation.
        Uses PyTorch for GPU acceleration when available, NumPy fallback otherwise.
        """
        # Try PyTorch first
        if _TORCH_OK:
            try:
                device = get_device()
                
                # Convert to PyTorch tensor
                if not isinstance(paths, torch.Tensor):
                    paths_tensor = to_torch(paths, device=device)
                else:
                    paths_tensor = paths
                
                current_price = torch.mean(paths_tensor[:, 0])
                exercise_value = (current_price - entry_price) / entry_price * direction * leverage
                
                future_prices = paths_tensor[:, 1:]
                future_pnl = (future_prices - entry_price) / entry_price * direction * leverage
                
                n_steps = future_pnl.shape[1]
                discount_factors = torch.pow(
                    torch.tensor(discount, device=device, dtype=torch.float32),
                    torch.arange(1, n_steps + 1, device=device, dtype=torch.float32)
                )
                discounted_pnl = future_pnl * discount_factors[None, :]
                
                continuation_value = torch.mean(torch.sum(discounted_pnl, dim=1))
                
                # Return as Python floats
                return float(exercise_value.cpu()), float(continuation_value.cpu())
            
            except Exception:
                # Fall through to NumPy
                pass
        
        # NumPy fallback
        paths_np = np.asarray(paths)
        current_price = np.mean(paths_np[:, 0])
        exercise_value = (current_price - entry_price) / entry_price * direction * leverage
        
        future_prices = paths_np[:, 1:]
        future_pnl = (future_prices - entry_price) / entry_price * direction * leverage
        
        n_steps = future_pnl.shape[1]
        discount_factors = np.power(discount, np.arange(1, n_steps + 1))
        discounted_pnl = future_pnl * discount_factors[None, :]
        
        continuation_value = np.mean(np.sum(discounted_pnl, axis=1))
        return exercise_value, continuation_value

class LeverageOptimizer:
    def __init__(self, max_leverage=10.0, kelly_fraction=0.5):
        self.max_leverage = max_leverage
        self.kelly_fraction = kelly_fraction
        
    def calculate_optimal_leverage(self, win_rate, avg_win, avg_loss, z_score, volatility):
        if avg_loss < 1e-9: avg_loss = 1e-9
        b_ratio = avg_win / abs(avg_loss)
        kelly_pct = win_rate - ((1 - win_rate) / b_ratio)
        if kelly_pct <= 0: return 1.0
        
        safe_kelly = kelly_pct * self.kelly_fraction
        target_vol = 0.02
        vol_leverage = target_vol / (volatility + 1e-9)
        
        raw_leverage = min(safe_kelly * 10, vol_leverage)
        
        ou_penalty = 1.0
        if abs(z_score) > 1.0:
            ou_penalty = max(0.0, 1.0 - (abs(z_score) - 1.0) * 0.5)
            
        final_leverage = np.clip(raw_leverage * ou_penalty, 1.0, self.max_leverage)
        return float(round(final_leverage * 2) / 2)

class QuantDecisionEngine:
    def __init__(self):
        self.kalman = KalmanFilter1D()
        self.ou = OUProcess(window=20)
        self.lsm = LSMModel()
        
    def decide(self, mc_engine, symbol, current_price, position, historical_prices, market_data, win_probability=0.5):
        # 1. 절대 손절 (-1.5%)
        pnl_pct = (current_price - position['entry_price']) / position['entry_price'] * position['direction'] * position['leverage']
        if pnl_pct < -0.015: return "CLOSE", f"🛑 Stop Loss (-1.5%)"
        
        # 2. 최소 보유 시간 (3분)
        import time
        if time.time() - position['entry_time'] < 180:
             if pnl_pct > -0.005:
                 return "HOLD", "⏳ Min Hold (3m)"

        # 3. LSMC 평가 (15분 horizon)
        mc_paths = mc_engine.generate_raw_paths(
            symbol=symbol,
            current_price=current_price,
            mu=market_data['predicted_mu'],
            sigma=market_data['volatility'],
            n_steps=15,
            dt=1/525600,
            n_paths=5000
        )
        
        exercise_val, continue_val = self.lsm.calculate_values(
            mc_paths, position['entry_price'], position['direction'], position['leverage']
        )
        
        # 판결
        score_close = 0
        if exercise_val > continue_val * 1.01:
            score_close += 50
            
        z_score = self.ou.get_z_score(np.array(historical_prices))
        if abs(z_score) > 2.5:
            score_close += 30
            
        if score_close >= 50:
            return "CLOSE", f"📉 Optimal Exit (Val:{exercise_val:.4f} > Fut:{continue_val:.4f})"
            
        return "HOLD", f"💎 Holding (Upside Left)"