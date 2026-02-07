"""
Portfolio-level Joint Monte Carlo Simulation Engine

Key features:
- Shared market factor path across symbols (correlated simulation)
- Block bootstrap with score-tilted sampling for regime consistency
- Market-wide + idiosyncratic jump modeling
- Portfolio risk metrics: VaR, CVaR, liquidation probability
- Multi-symbol coordination via beta decomposition

Author: Optimized for portfolio-level decision making
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# JAX Metal support - use lazy initialization from jax_backend
import engines.mc.jax_backend as jax_backend


@jax_backend.lazy_jit()
def _jnp_bps_to_mult(bps: float):
    return 1.0 + bps / 10000.0


@jax_backend.lazy_jit()
def _jnp_first_hit_index(mask):
    """Finds first True index per row using argmax."""
    jnp = jax_backend.jnp
    sims, days = mask.shape
    idx = jnp.argmax(mask, axis=1)
    has = jnp.any(mask, axis=1)
    return jnp.where(has, idx, days + 999)


@jax_backend.lazy_jit()
def _symbol_realized_pnl_jax(
    price0: float,
    paths,               # (sims, days)
    tp: float,
    sl: float,
    liq_price: float,
    jump_mask_any=None,       # (sims, days) bool
    slippage_bps: float=0.0,
    jump_slippage_mult: float=1.0,
    liq_penalty_bps: float=0.0,
):
    """Vectorized PnL calculation on GPU."""
    jnp = jax_backend.jnp
    sims, days = paths.shape
    
    tp_mask = paths >= tp
    sl_mask = paths <= sl
    liq_mask = paths <= liq_price
    
    tp_i = _jnp_first_hit_index(tp_mask)
    sl_i = _jnp_first_hit_index(sl_mask)
    liq_i = _jnp_first_hit_index(liq_mask)
    
    first = jnp.minimum(tp_i, jnp.minimum(sl_i, liq_i))
    
    hit_tp = (tp_i == first) & (tp_i < days)
    hit_sl = (sl_i == first) & (sl_i < days) & ~hit_tp
    hit_liq = (liq_i == first) & (liq_i < days) & ~hit_tp & ~hit_sl
    hit_none = ~(hit_tp | hit_sl | hit_liq)
    
    slip = _jnp_bps_to_mult(slippage_bps)
    liq_pen = _jnp_bps_to_mult(liq_penalty_bps)
    
    # We need indices for gathering jump info
    batch_idx = jnp.arange(sims)
    
    # TP PnL
    tp_days = jnp.clip(tp_i, 0, days - 1)
    tp_jump_extra = jnp.where(jump_mask_any[batch_idx, tp_days], jump_slippage_mult, 1.0)
    tp_fill = tp / (slip * tp_jump_extra)
    pnl_tp = (tp_fill - price0) / price0
    
    # SL PnL
    sl_days = jnp.clip(sl_i, 0, days - 1)
    sl_jump_extra = jnp.where(jump_mask_any[batch_idx, sl_days], jump_slippage_mult, 1.0)
    sl_fill = sl * (slip * sl_jump_extra)
    pnl_sl = (sl_fill - price0) / price0
    
    # LIQ PnL
    liq_days = jnp.clip(liq_i, 0, days - 1)
    liq_jump_extra = jnp.where(jump_mask_any[batch_idx, liq_days], jump_slippage_mult, 1.0)
    liq_fill = liq_price * (slip * liq_jump_extra) * liq_pen
    pnl_liq = (liq_fill - price0) / price0
    
    # NONE PnL (last day)
    pnl_none = ((paths[:, -1] / slip) - price0) / price0
    
    # Combine
    final_pnl = jnp.zeros(sims)
    final_pnl = jnp.where(hit_tp, pnl_tp, final_pnl)
    final_pnl = jnp.where(hit_sl, pnl_sl, final_pnl)
    final_pnl = jnp.where(hit_liq, pnl_liq, final_pnl)
    final_pnl = jnp.where(hit_none, pnl_none, final_pnl)
    
    return final_pnl, hit_tp, hit_sl, hit_liq

# -----------------------------
# Config
# -----------------------------

@dataclass
class PortfolioConfig:
    """Configuration for portfolio joint simulation"""
    
    # Monte Carlo
    days: int = 3
    simulations: int = 30000
    batch_size: int = 6000  # controls memory
    block_size: int = 12
    min_history: int = 180
    seed: Optional[int] = 42

    # AI score -> drift calibration (daily log-return)
    # drift_per_day = clip(score, -1, 1) * drift_k * recent_vol
    drift_k: float = 0.35
    score_clip: float = 1.0

    # Score tilting for block selection (directional regime-consistent sampling)
    # Reduced to prevent overfitting - still references downside scenarios even with high score
    tilt_strength: float = 0.6  # conservative (was 1.5)
    tilt_clip: float = 2.0

    # Jump settings (specified in PRICE DROP fraction range; converted to log)
    use_jumps: bool = True

    # Market common jump (same day shock, shared across all symbols in joint simulation)
    p_jump_market: float = 0.005
    jump_market_drop_lo: float = 0.07
    jump_market_drop_hi: float = 0.20

    # Idiosyncratic jump (symbol-specific)
    p_jump_idio: float = 0.007
    jump_idio_drop_lo: float = 0.05
    jump_idio_drop_hi: float = 0.18

    # Execution / slippage
    slippage_bps: float = 8.0
    jump_slippage_mult: float = 2.5
    liquidation_extra_penalty_bps: float = 30.0

    # Trading logic / liquidation
    leverage: float = 10.0
    maintenance_buffer: float = 0.025  # earlier liquidation (conservative; increased for high leverage)
    maintenance_buffer_high_lev: float = 0.035  # for leverage >= 20x

    # Portfolio build
    target_leverage: float = 10.0
    individual_cap: float = 3.0
    min_vol_floor: float = 0.005
    risk_aversion: float = 0.5  # penalize liquidation prob in scoring
    # Correlation strength controls
    market_factor_scale: float = 1.0
    residual_scale: float = 1.0

    # Portfolio-level "account liquidation proxy"
    # If portfolio pnl <= -(1/target_leverage) - buffer => account liquidated proxy
    portfolio_maintenance_buffer: float = 0.010

    # Risk report
    var_alpha: float = 0.05  # 5% VaR/CVaR

    # Rebalance policy simulation (pre-MC)
    rebalance_sim_enabled: bool = False
    rebalance_interval: int = 1  # steps between rebalances
    rebalance_fee_bps: float = 6.0
    rebalance_slippage_bps: float = 4.0
    rebalance_score_noise: float = 0.0  # score perturbation per rebalance
    rebalance_min_score: float = 0.0  # scores below -> zero weight


# -----------------------------
# Helpers
# -----------------------------

def _extract_closes(ohlcv: List[Tuple]) -> np.ndarray:
    """Extract close prices from OHLCV tuples"""
    if not ohlcv:
        return np.array([], dtype=np.float64)
    row = ohlcv[0]
    n = len(row)
    if n >= 6:      # (t,o,h,l,c,v)
        close_idx = 4
    elif n == 5:    # (o,h,l,c,v)
        close_idx = 3
    else:
        close_idx = n - 1
    return np.array([float(r[close_idx]) for r in ohlcv], dtype=np.float64)


def _safe_log_returns(closes: np.ndarray) -> np.ndarray:
    """Compute log returns with safety guards"""
    closes = np.asarray(closes, dtype=np.float64)
    closes = np.maximum(closes, 1e-12)
    return np.diff(np.log(closes))


def _ols_beta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute OLS beta: beta = cov(x,y) / var(x)"""
    if _JAX_OK:
        try:
            xj = jnp.asarray(x)
            yj = jnp.asarray(y)
            # Covariance matrix for [x, y]
            mat = jnp.stack([xj, yj], axis=1)
            cov_mat = jax_covariance(mat)
            vx = cov_mat[0, 0]
            if vx < 1e-12:
                return 0.0
            return float(cov_mat[0, 1] / vx)
        except Exception:
            pass
            
    vx = np.var(x)
    if vx < 1e-12:
        return 0.0
    return float(np.cov(x, y, ddof=0)[0, 1] / vx)


def _bps_to_mult(bps: float) -> float:
    """Convert basis points to multiplier"""
    return 1.0 + bps / 10000.0


def _price_drop_to_logshock(drop_frac: np.ndarray) -> np.ndarray:
    """Convert price drop fraction to log shock"""
    drop_frac = np.clip(drop_frac, 1e-9, 0.999999)
    return np.log(1.0 - drop_frac)


def _first_hit_index(mask: np.ndarray) -> np.ndarray:
    """Find first True index per row; returns large number if never hit"""
    sims, days = mask.shape
    idx = np.argmax(mask, axis=1)
    has = mask.max(axis=1)
    return np.where(has, idx, days + 999)


def _force_fill_with_caps(raw: np.ndarray, target: float, cap: float) -> np.ndarray:
    """Force-fill weights to target with individual caps via iterative scaling"""
    n = raw.size
    if n == 0 or np.all(raw <= 0):
        return np.zeros(n, dtype=np.float64)

    w = raw / raw.sum() * target
    w = np.minimum(w, cap)

    for _ in range(60):
        total = w.sum()
        if abs(total - target) < 1e-12:
            break
        if total > target:
            w *= target / total
            w = np.minimum(w, cap)
            continue

        leftover = target - total
        room = cap - w
        eligible = room > 1e-12
        if not np.any(eligible):
            break

        raw_elig = raw.copy()
        raw_elig[~eligible] = 0.0
        s = raw_elig.sum()
        if s <= 0:
            add = leftover / eligible.sum()
            w[eligible] += np.minimum(room[eligible], add)
        else:
            add = leftover * (raw_elig / s)
            w += np.minimum(room, add)

        w = np.minimum(w, cap)

    return w


# -----------------------------
# Engine (Joint Simulation)
# -----------------------------

class PortfolioJointSimEngine:
    """
    Joint simulation engine for portfolio-level Monte Carlo evaluation.
    
    Features:
      - One shared market factor path per simulation (bootstrapped + market jump)
      - Per-symbol residual paths (bootstrapped + idio jump)
      - Per-symbol TP/SL/liquidation with slippage and jump-day worse execution
      - Portfolio PnL distribution computed from weights * per-symbol realized PnL per sim
    """

    def __init__(self, ohlcv_map: Dict[str, List[Tuple]], ai_scores: Dict[str, float], cfg: PortfolioConfig):
        self.ohlcv_map = ohlcv_map
        self.ai_scores = ai_scores
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        
        # JAX state
        self._jax_ok = _JAX_OK
        if self._jax_ok:
            self.jax_key = jrand.PRNGKey(cfg.seed if cfg.seed is not None else 42)
            logger.info(f"🚀 [PORTFOLIO_MC] JAX Metal acceleration enabled for joint simulation")

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._prep_market_factor()

    # ---------- data prep ----------

    def _prep_market_factor(self) -> None:
        """Prepare market factor from cross-sectional average of all symbols"""
        rets = []
        for sym, rows in self.ohlcv_map.items():
            closes = _extract_closes(rows)
            if closes.size < self.cfg.min_history:
                continue
            r = _safe_log_returns(closes)
            if r.size >= self.cfg.min_history - 1:
                rets.append(r)
        if not rets:
            self.market_ret = np.array([], dtype=np.float64)
            return
        T = min(r.size for r in rets)
        mat = np.stack([r[-T:] for r in rets], axis=0)
        self.market_ret = np.mean(mat, axis=0).astype(np.float64)

    def _get_symbol_data(self, symbol: str) -> Dict[str, Any]:
        """Get or compute symbol-specific data (beta, residual, vol, etc)"""
        if symbol in self._cache:
            return self._cache[symbol]

        rows = self.ohlcv_map.get(symbol, [])
        closes = _extract_closes(rows)
        if closes.size < self.cfg.min_history:
            raise ValueError(f"Not enough history for {symbol}: {closes.size} < {self.cfg.min_history}")

        r = _safe_log_returns(closes)
        if self.market_ret.size == 0:
            m = np.zeros_like(r)
        else:
            T = min(r.size, self.market_ret.size)
            r = r[-T:]
            m = self.market_ret[-T:]

        # Beta fallback for insufficient data
        min_data_for_beta = 30
        if len(r) < min_data_for_beta or len(m) < min_data_for_beta:
            beta = 1.0  # Market-following fallback for new/illiquid symbols
        else:
            beta = _ols_beta(m, r)
        resid = r - beta * m

        recent = r[-min(60, r.size):]
        recent_vol = float(np.std(recent, ddof=0))
        if recent_vol < 1e-12:
            recent_vol = float(np.std(r, ddof=0)) + 1e-6

        out = {
            "closes": closes,
            "ret": r,
            "mkt": m,
            "beta": beta,
            "resid": resid,
            "recent_vol": recent_vol,
            "last_price": float(closes[-1]),
        }
        self._cache[symbol] = out
        return out

    # ---------- sampling ----------

    def _score_to_drift(self, score: float, recent_vol: float) -> float:
        """Convert AI score to daily drift"""
        s = float(np.clip(score, -self.cfg.score_clip, self.cfg.score_clip))
        return s * self.cfg.drift_k * recent_vol

    def _block_start_probs(self, series: np.ndarray, score: float, block: int) -> np.ndarray:
        """Compute tilted block start probabilities based on score"""
        n = series.size
        max_start = max(1, n - block)
        starts = np.arange(max_start)

        cs = np.concatenate([[0.0], np.cumsum(series)])
        block_sum = cs[starts + block] - cs[starts]

        sgn = np.sign(score)
        align = sgn * np.sign(block_sum)
        strength = np.clip(abs(score) * self.cfg.tilt_strength, 0.0, self.cfg.tilt_clip)
        w = np.exp(strength * align)
        w = w / w.sum()
        return w

    def _sample_block_bootstrap_jax(self, series: np.ndarray, score: float, days: int, block: int, sims: int, key) -> jnp.ndarray:
        """JAX-accelerated block bootstrap sampling."""
        series_j = jnp.asarray(series, dtype=jnp.float32)
        n = series_j.size
        if n < block + 2:
            idx = jrand.randint(key, shape=(sims, days), minval=0, maxval=n)
            return series_j[idx]

        # Reuse existing tilted probability logic (it's fast on CPU anyway, or JAX convert)
        # For simplicity and to avoid Tracer issues with dynamic block sizes, 
        # we compute probabilities using numpy and pass to JAX.
        probs = self._block_start_probs(series, score, block)
        max_start = probs.size
        num_blocks = int(np.ceil(days / block))

        starts = jrand.choice(key, jnp.arange(max_start), shape=(sims, num_blocks), replace=True, p=jnp.asarray(probs))
        
        # Build paths (vectorized)
        def _get_block(start):
            return lax.dynamic_slice_in_dim(series_j, start, block)
        
        # We can use vmap to extract blocks
        get_blocks_vmap = jax.vmap(jax.vmap(_get_block))
        out = get_blocks_vmap(starts) # (sims, num_blocks, block)
        
        # Reshape and trim to days
        out = out.reshape((sims, -1))[:, :days]
        return out

    def _sample_block_bootstrap(self, series: np.ndarray, score: float, days: int, block: int, sims: int) -> np.ndarray:
        """Sample block bootstrap with score-tilted regime consistency (Numpy version)"""
        n = series.size
        if n < block + 2:
            idx = self.rng.integers(0, n, size=(sims, days))
            return series[idx]

        probs = self._block_start_probs(series, score, block)
        max_start = probs.size
        num_blocks = int(np.ceil(days / block))

        starts = self.rng.choice(np.arange(max_start), size=(sims, num_blocks), replace=True, p=probs)

        base = np.arange(block)
        segs = []
        for j in range(num_blocks):
            idx = starts[:, j:j+1] + base
            segs.append(series[idx])
        out = np.concatenate(segs, axis=1)[:, :days]
        return out

    # ---------- per-symbol path -> PnL (vectorized) ----------

    def _symbol_realized_pnl(
        self,
        price0: float,
        paths: np.ndarray,               # (b, days)
        tp: float,
        sl: float,
        liq_price: float,
        jump_mask_any: np.ndarray,       # (b, days) bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute realized PnL for a symbol with TP/SL/liquidation.
        
        Returns:
          pnl (b,)
          hit_tp (b,) bool
          hit_sl (b,) bool
          hit_liq (b,) bool
        """
        cfg = self.cfg
        b, days = paths.shape

        tp_mask = paths >= tp
        sl_mask = paths <= sl
        liq_mask = paths <= liq_price

        tp_i = _first_hit_index(tp_mask)
        sl_i = _first_hit_index(sl_mask)
        liq_i = _first_hit_index(liq_mask)
        first = np.minimum(tp_i, np.minimum(sl_i, liq_i))

        slip = _bps_to_mult(cfg.slippage_bps)
        liq_pen = _bps_to_mult(cfg.liquidation_extra_penalty_bps)

        pnl = np.zeros(b, dtype=np.float64)

        hit_tp = (tp_i == first)
        hit_sl = (sl_i == first) & ~hit_tp
        hit_liq = (liq_i == first) & ~hit_tp & ~hit_sl
        hit_none = ~(hit_tp | hit_sl | hit_liq)

        # TP fill worse for long: receive slightly lower
        if np.any(hit_tp):
            day = tp_i[hit_tp].clip(0, days - 1)
            extra = np.where(jump_mask_any[hit_tp, day], cfg.jump_slippage_mult, 1.0)
            fill = tp / (slip * extra)
            pnl[hit_tp] = (fill - price0) / price0

        # SL fill worse: sell lower
        if np.any(hit_sl):
            day = sl_i[hit_sl].clip(0, days - 1)
            extra = np.where(jump_mask_any[hit_sl, day], cfg.jump_slippage_mult, 1.0)
            fill = sl * (slip * extra)
            pnl[hit_sl] = (fill - price0) / price0

        # LIQ fill even worse
        if np.any(hit_liq):
            day = liq_i[hit_liq].clip(0, days - 1)
            extra = np.where(jump_mask_any[hit_liq, day], cfg.jump_slippage_mult, 1.0)
            fill = liq_price * (slip * extra) * liq_pen
            pnl[hit_liq] = (fill - price0) / price0

        # None: exit at last with mild slippage
        if np.any(hit_none):
            last = paths[hit_none, -1]
            pnl[hit_none] = ((last / slip) - price0) / price0

        return pnl, hit_tp, hit_sl, hit_liq

    # ---------- public: per-symbol evaluation (single-asset simulation) ----------

    def evaluate_symbol(self, symbol: str, tp_mult: float = 4.0, sl_mult: float = 2.0) -> Dict[str, float]:
        """Evaluate single symbol with Monte Carlo simulation"""
        cfg = self.cfg
        d = self._get_symbol_data(symbol)

        price0 = d["last_price"]
        vol = max(cfg.min_vol_floor, d["recent_vol"])
        tp = price0 * (1.0 + vol * tp_mult)
        sl = price0 * (1.0 - vol * sl_mult)

        # liquidation barrier (leverage-linked buffer)
        buffer = cfg.maintenance_buffer_high_lev if cfg.leverage >= 20.0 else cfg.maintenance_buffer
        liq_simple = -(1.0 / max(cfg.leverage, 1e-9)) - buffer
        liq_price = price0 * float(np.exp(liq_simple))

        score = float(self.ai_scores.get(symbol, 0.0))
        drift = self._score_to_drift(score, d["recent_vol"])

        days = cfg.days
        block = max(3, min(cfg.block_size, max(3, days * 4)))

        total = cfg.simulations
        done = 0

        sum_pnl = 0.0
        c_tp = 0
        c_sl = 0
        c_liq = 0

        while done < total:
            b = int(min(cfg.batch_size, total - done))

            if self._jax_ok:
                # ✅ PRNGKey Split (Goal 2)
                self.jax_key, k_bs1, k_bs2, k_j1, k_j2, k_j3, k_j4 = jrand.split(self.jax_key, 7)
                
                # JAX Bootstrap sampling
                mkt_path = self._sample_block_bootstrap_jax(d["mkt"], score, days, block, b, k_bs1)
                res_path = self._sample_block_bootstrap_jax(d["resid"], score, days, block, b, k_bs2)
                mkt_path = mkt_path * float(cfg.market_factor_scale)
                res_path = res_path * float(cfg.residual_scale)
                sim_r = d["beta"] * mkt_path + res_path + drift

                jump_any = jnp.zeros((b, days), dtype=bool)
                if cfg.use_jumps:
                    # Market jump
                    m_mask = jrand.uniform(k_j1, shape=(b, days)) < cfg.p_jump_market
                    m_drop = jrand.uniform(k_j2, shape=(b, days), minval=cfg.jump_market_drop_lo, maxval=cfg.jump_market_drop_hi)
                    sim_r += jnp.log(1.0 - jnp.clip(m_drop, 0.0, 0.999)) * m_mask
                    jump_any |= m_mask
                    
                    # Idio jump
                    i_mask = jrand.uniform(k_j3, shape=(b, days)) < cfg.p_jump_idio
                    i_drop = jrand.uniform(k_j4, shape=(b, days), minval=cfg.jump_idio_drop_lo, maxval=cfg.jump_idio_drop_hi)
                    sim_r += jnp.log(1.0 - jnp.clip(i_drop, 0.0, 0.999)) * i_mask
                    jump_any |= i_mask

                paths = price0 * jnp.exp(jnp.cumsum(sim_r, axis=1))

                # GPU Kernel call
                pnl_j, hit_tp_j, hit_sl_j, hit_liq_j = _symbol_realized_pnl_jax(
                    price0=price0, paths=paths, tp=tp, sl=sl, liq_price=liq_price, 
                    jump_mask_any=jump_any,
                    slippage_bps=cfg.slippage_bps,
                    jump_slippage_mult=cfg.jump_slippage_mult,
                    liq_penalty_bps=cfg.liquidation_extra_penalty_bps
                )
                
                # Transfer back if needed (or keep accumulation on JAX if we want)
                sum_pnl += float(jnp.sum(pnl_j))
                c_tp += int(jnp.sum(hit_tp_j))
                c_sl += int(jnp.sum(hit_sl_j))
                c_liq += int(jnp.sum(hit_liq_j))
            else:
                # Numpy Fallback
                mkt_path = self._sample_block_bootstrap(d["mkt"], score, days, block, b)
                res_path = self._sample_block_bootstrap(d["resid"], score, days, block, b)
                mkt_path = mkt_path * float(cfg.market_factor_scale)
                res_path = res_path * float(cfg.residual_scale)
                sim_r = d["beta"] * mkt_path + res_path + drift

                jump_any = np.zeros((b, days), dtype=bool)
                if cfg.use_jumps:
                    u = self.rng.random((b, days))
                    m_mask = u < cfg.p_jump_market
                    drop = self.rng.uniform(cfg.jump_market_drop_lo, cfg.jump_market_drop_hi, size=(b, days))
                    sim_r += _price_drop_to_logshock(drop) * m_mask
                    jump_any |= m_mask

                    u2 = self.rng.random((b, days))
                    i_mask = u2 < cfg.p_jump_idio
                    drop2 = self.rng.uniform(cfg.jump_idio_drop_lo, cfg.jump_idio_drop_hi, size=(b, days))
                    sim_r += _price_drop_to_logshock(drop2) * i_mask
                    jump_any |= i_mask

                paths = price0 * np.exp(np.cumsum(sim_r, axis=1))
                pnl, hit_tp, hit_sl, hit_liq = self._symbol_realized_pnl(
                    price0=price0, paths=paths, tp=tp, sl=sl, liq_price=liq_price, jump_mask_any=jump_any
                )

                sum_pnl += float(np.sum(pnl))
                c_tp += int(np.sum(hit_tp))
                c_sl += int(np.sum(hit_sl))
                c_liq += int(np.sum(hit_liq))
            
            done += b

        return {
            "expected_pnl": float(sum_pnl / total),
            "prob_hit_tp": float(c_tp / total),
            "prob_hit_sl": float(c_sl / total),
            "prob_liquidated": float(c_liq / total),
            "entry_price": float(price0),
            "tp": float(tp),
            "sl": float(sl),
            "liq_price": float(liq_price),
            "drift_per_day": float(drift),
            "beta_to_market": float(d["beta"]),
            "recent_vol": float(d["recent_vol"]),
        }

    # ---------- portfolio build + JOINT simulation ----------

    def build_portfolio(self, symbols: List[str], tp_mult: float = 4.0, sl_mult: float = 2.0) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Build portfolio with joint simulation.
        
        Step 1) per-symbol evaluate (independent) for scoring
        Step 2) force-fill weights with caps to target leverage
        Step 3) JOINT simulate portfolio PnL distribution using shared market path & jumps
        """
        cfg = self.cfg
        per_sym: Dict[str, Dict[str, float]] = {}
        valid: List[str] = []
        raw_scores: List[float] = []

        for sym in symbols:
            try:
                m = self.evaluate_symbol(sym, tp_mult=tp_mult, sl_mult=sl_mult)
            except Exception:
                continue
            per_sym[sym] = m
            valid.append(sym)

            mu = m["expected_pnl"]
            vol = max(cfg.min_vol_floor, m["recent_vol"])
            if mu <= 0:
                s = 0.0
            else:
                s = mu / vol
                if cfg.risk_aversion > 0:
                    s *= max(0.0, 1.0 - cfg.risk_aversion * m["prob_liquidated"])
            raw_scores.append(s)

        raw = np.array(raw_scores, dtype=np.float64)
        w = _force_fill_with_caps(raw, cfg.target_leverage, cfg.individual_cap)
        weights = {sym: float(w[i]) for i, sym in enumerate(valid) if w[i] > 0}

        # Joint portfolio simulation with FINAL weights
        port_metrics = self.simulate_portfolio_joint(weights, tp_mult=tp_mult, sl_mult=sl_mult)
        rebalance_metrics = None
        if cfg.rebalance_sim_enabled:
            try:
                rebalance_metrics = self.simulate_portfolio_rebalance(valid, raw_scores, weights=weights)
            except Exception as e:
                logger.warning(f"[PORTFOLIO_REBAL_SIM] failed: {e}")
        
        # ✅ FINAL LOG: Show scores to user clearly
        logger.info(
            f"💰 [PORTFOLIO_RESULT] Allocated Symbols: {list(weights.keys())} | "
            f"Exp_PnL: {port_metrics['expected_portfolio_pnl']*100:.4f}% | "
            f"CVaR: {port_metrics['cvar']*100:.4f}% | "
            f"Total_Lev: {port_metrics['total_leverage_allocated']:.2f}x"
        )

        report = {
            "weights": weights,
            "n_valid_symbols": int(len(valid)),
            "per_symbol_metrics": per_sym,
            **port_metrics,
        }
        if rebalance_metrics:
            report.update({f"rebalance_{k}": v for k, v in rebalance_metrics.items()})
        return weights, report

    def simulate_portfolio_joint(self, weights: Dict[str, float], tp_mult: float = 4.0, sl_mult: float = 2.0) -> Dict[str, Any]:
        """
        True joint simulation:
          - One market path per sim
          - Market jump mask shared across ALL symbols
          - Each symbol has its own residual bootstrap & idio jump
          - Portfolio PnL per sim = sum_i weight_i * pnl_i(sim)
        
        Outputs: expected pnl, VaR/CVaR, prob_any_liq, prob_account_liq_proxy, etc.
        """
        cfg = self.cfg
        if not weights:
            return {
                "expected_portfolio_pnl": 0.0,
                "var": 0.0,
                "cvar": 0.0,
                "prob_any_position_liquidated": 0.0,
                "prob_account_liquidation_proxy": 0.0,
                "total_leverage_allocated": 0.0,
                "portfolio_pnl_samples_head": [],
            }

        syms = list(weights.keys())
        w = np.array([weights[s] for s in syms], dtype=np.float64)
        total_lev = float(w.sum())

        # Prepare per-symbol constants
        sym_data = [self._get_symbol_data(s) for s in syms]
        price0 = np.array([d["last_price"] for d in sym_data], dtype=np.float64)
        beta = np.array([d["beta"] for d in sym_data], dtype=np.float64)
        vol = np.array([max(cfg.min_vol_floor, d["recent_vol"]) for d in sym_data], dtype=np.float64)

        tp = price0 * (1.0 + vol * tp_mult)
        sl = price0 * (1.0 - vol * sl_mult)

        # liquidation barrier per symbol (leverage-linked buffer)
        buffer = cfg.maintenance_buffer_high_lev if cfg.leverage >= 20.0 else cfg.maintenance_buffer
        liq_simple = -(1.0 / max(cfg.leverage, 1e-9)) - buffer
        liq_price = price0 * float(np.exp(liq_simple))

        scores = np.array([float(self.ai_scores.get(s, 0.0)) for s in syms], dtype=np.float64)
        drifts = np.array([self._score_to_drift(scores[i], vol[i]) for i in range(len(syms))], dtype=np.float64)

        days = cfg.days
        block = max(3, min(cfg.block_size, max(3, days * 4)))

        # Market sampling score: weighted average score
        if total_lev > 1e-12:
            market_score = float(np.clip(np.dot(w, scores) / total_lev, -cfg.score_clip, cfg.score_clip))
        else:
            market_score = 0.0

        # Market series for bootstrap
        if self.market_ret.size == 0:
            market_series = np.zeros(500, dtype=np.float64)
        else:
            market_series = self.market_ret

        total = cfg.simulations
        done = 0

        # Accumulate portfolio pnl samples
        port_pnls = np.empty(total, dtype=np.float64)
        any_liq = np.zeros(total, dtype=bool)

        # Account liquidation proxy threshold
        acc_liq_thresh = -(1.0 / max(cfg.target_leverage, 1e-9)) - cfg.portfolio_maintenance_buffer

        while done < total:
            b = int(min(cfg.batch_size, total - done))

            if self._jax_ok:
                # ✅ PRNGKey Split for shared factors (Goal 2)
                self.jax_key, k_bs_m, k_j_m, k_j_m2 = jrand.split(self.jax_key, 4)
                
                # Shared market path (JAX)
                mkt_path = self._sample_block_bootstrap_jax(market_series, market_score, days, block, b, k_bs_m)
                mkt_path = mkt_path * float(cfg.market_factor_scale)
                jump_mkt_mask = jnp.zeros((b, days), dtype=bool)
                if cfg.use_jumps:
                    jump_mkt_mask = jrand.uniform(k_j_m, shape=(b, days)) < cfg.p_jump_market
                    drop = jrand.uniform(k_j_m2, shape=(b, days), minval=cfg.jump_market_drop_lo, maxval=cfg.jump_market_drop_hi)
                    mkt_path = mkt_path + jnp.log(1.0 - jnp.clip(drop, 0.0, 0.999)) * jump_mkt_mask

                batch_port_pnl = jnp.zeros(b, dtype=jnp.float32)
                batch_any_liq = jnp.zeros(b, dtype=bool)

                for i, s in enumerate(syms):
                    # ✅ PRNGKey Split per symbol residuals (Goal 2)
                    self.jax_key, k_bs_r, k_j_i, k_j_i2 = jrand.split(self.jax_key, 4)
                    
                    d = sym_data[i]
                    res_path = self._sample_block_bootstrap_jax(d["resid"], float(scores[i]), days, block, b, k_bs_r)
                    res_path = res_path * float(cfg.residual_scale)
                    sim_r = beta[i] * mkt_path + res_path + drifts[i]

                    jump_any = jump_mkt_mask.copy()
                    if cfg.use_jumps:
                        jump_idio_mask = jrand.uniform(k_j_i, shape=(b, days)) < cfg.p_jump_idio
                        drop2 = jrand.uniform(k_j_i2, shape=(b, days), minval=cfg.jump_idio_drop_lo, maxval=cfg.jump_idio_drop_hi)
                        sim_r = sim_r + jnp.log(1.0 - jnp.clip(drop2, 0.0, 0.999)) * jump_idio_mask
                        jump_any |= jump_idio_mask

                    paths = price0[i] * jnp.exp(jnp.cumsum(sim_r, axis=1))

                    pnl_i, _, _, hit_liq = _symbol_realized_pnl_jax(
                        price0=float(price0[i]), paths=paths, tp=float(tp[i]), sl=float(sl[i]),
                        liq_price=float(liq_price[i]), jump_mask_any=jump_any,
                        slippage_bps=cfg.slippage_bps, jump_slippage_mult=cfg.jump_slippage_mult, 
                        liq_penalty_bps=cfg.liquidation_extra_penalty_bps
                    )
                    batch_port_pnl += w[i] * pnl_i
                    batch_any_liq |= hit_liq

                port_pnls[done:done + b] = np.asarray(batch_port_pnl)
                any_liq[done:done + b] = np.asarray(batch_any_liq)
            else:
                mkt_path = self._sample_block_bootstrap(market_series, market_score, days, block, b)
                mkt_path = mkt_path * float(cfg.market_factor_scale)
                jump_mkt_mask = np.zeros((b, days), dtype=bool)
                if cfg.use_jumps:
                    u = self.rng.random((b, days))
                    jump_mkt_mask = u < cfg.p_jump_market
                    drop = self.rng.uniform(cfg.jump_market_drop_lo, cfg.jump_market_drop_hi, size=(b, days))
                    mkt_path = mkt_path + _price_drop_to_logshock(drop) * jump_mkt_mask

                batch_port_pnl = np.zeros(b, dtype=np.float64)
                batch_any_liq = np.zeros(b, dtype=bool)

                for i, s in enumerate(syms):
                    d = sym_data[i]
                    res_path = self._sample_block_bootstrap(d["resid"], float(scores[i]), days, block, b)
                    res_path = res_path * float(cfg.residual_scale)
                    sim_r = beta[i] * mkt_path + res_path + drifts[i]
                    jump_any = jump_mkt_mask.copy()
                    if cfg.use_jumps:
                        u2 = self.rng.random((b, days))
                        jump_idio_mask = u2 < cfg.p_jump_idio
                        drop2 = self.rng.uniform(cfg.jump_idio_drop_lo, cfg.jump_idio_drop_hi, size=(b, days))
                        sim_r = sim_r + _price_drop_to_logshock(drop2) * jump_idio_mask
                        jump_any |= jump_idio_mask
                    paths = price0[i] * np.exp(np.cumsum(sim_r, axis=1))
                    pnl_i, _, _, hit_liq = self._symbol_realized_pnl(
                        price0=float(price0[i]), paths=paths, tp=float(tp[i]), sl=float(sl[i]),
                        liq_price=float(liq_price[i]), jump_mask_any=jump_any,
                    )
                    batch_port_pnl += w[i] * pnl_i
                    batch_any_liq |= hit_liq

                port_pnls[done:done + b] = batch_port_pnl
                any_liq[done:done + b] = batch_any_liq
            
            done += b

        # Portfolio risk stats
        exp_pnl = float(np.mean(port_pnls))
        alpha = float(cfg.var_alpha)
        q = float(np.quantile(port_pnls, alpha))
        cvar = float(np.mean(port_pnls[port_pnls <= q])) if np.any(port_pnls <= q) else q

        prob_any_liq = float(np.mean(any_liq))
        prob_acc_liq = float(np.mean(port_pnls <= acc_liq_thresh))

        # small head sample for quick sanity check
        head = port_pnls[: min(20, port_pnls.size)].tolist()

        return {
            "expected_portfolio_pnl": exp_pnl,
            "var": q,
            "cvar": cvar,
            "prob_any_position_liquidated": prob_any_liq,
            "prob_account_liquidation_proxy": prob_acc_liq,
            "account_liq_threshold_proxy": float(acc_liq_thresh),
            "total_leverage_allocated": float(total_lev),
            "portfolio_pnl_samples_head": head,
        }

    def _weights_from_scores(self, scores: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        s = np.asarray(scores, dtype=np.float64)
        if cfg.rebalance_min_score > 0:
            s = np.where(s >= float(cfg.rebalance_min_score), s, 0.0)
        s = np.clip(s, 0.0, None)
        if s.size == 0 or float(s.sum()) <= 0:
            return np.zeros_like(s)
        return _force_fill_with_caps(s, cfg.target_leverage, cfg.individual_cap)

    def simulate_portfolio_rebalance(
        self,
        symbols: List[str],
        scores: List[float],
        *,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Pre-MC portfolio simulation with explicit rebalancing (partial/zero exits).
        Uses joint market path + residual bootstraps, then rebalances to target weights
        at a fixed interval with cost and turnover tracking.
        """
        cfg = self.cfg
        if not symbols:
            return {
                "expected_portfolio_pnl": 0.0,
                "turnover_mean": 0.0,
                "turnover_p95": 0.0,
                "cost_mean": 0.0,
                "prob_account_liq_proxy": 0.0,
                "steps": int(cfg.days),
            }

        syms = list(symbols)
        base_scores = np.asarray(scores, dtype=np.float64)
        if weights is None:
            w_target = self._weights_from_scores(base_scores)
        else:
            w_target = np.array([float(weights.get(s, 0.0)) for s in syms], dtype=np.float64)

        total_lev = float(np.sum(np.abs(w_target)))
        if total_lev <= 1e-12:
            return {
                "expected_portfolio_pnl": 0.0,
                "turnover_mean": 0.0,
                "turnover_p95": 0.0,
                "cost_mean": 0.0,
                "prob_account_liq_proxy": 0.0,
                "steps": int(cfg.days),
            }

        sym_data = [self._get_symbol_data(s) for s in syms]
        price0 = np.array([d["last_price"] for d in sym_data], dtype=np.float64)
        beta = np.array([d["beta"] for d in sym_data], dtype=np.float64)
        vol = np.array([max(cfg.min_vol_floor, d["recent_vol"]) for d in sym_data], dtype=np.float64)
        drifts = np.array([self._score_to_drift(float(base_scores[i]), vol[i]) for i in range(len(syms))], dtype=np.float64)

        days = int(cfg.days)
        if days <= 0:
            days = 1
        block = max(3, min(cfg.block_size, max(3, days * 4)))

        # Market series for bootstrap
        if self.market_ret.size == 0:
            market_series = np.zeros(500, dtype=np.float64)
        else:
            market_series = self.market_ret

        total = int(cfg.simulations)
        done = 0
        port_pnls = np.empty(total, dtype=np.float64)
        turnovers = np.empty(total, dtype=np.float64)
        costs = np.empty(total, dtype=np.float64)

        acc_liq_thresh = -(1.0 / max(cfg.target_leverage, 1e-9)) - cfg.portfolio_maintenance_buffer
        cost_rate = float(cfg.rebalance_fee_bps + cfg.rebalance_slippage_bps) / 10000.0
        reb_interval = max(1, int(cfg.rebalance_interval))

        while done < total:
            b = int(min(cfg.batch_size, total - done))
            # Sample market returns
            mkt_path = self._sample_block_bootstrap(market_series, 0.0, days, block, b)
            mkt_path = mkt_path * float(cfg.market_factor_scale)
            if cfg.use_jumps:
                u = self.rng.random((b, days))
                jump_mkt_mask = u < cfg.p_jump_market
                drop = self.rng.uniform(cfg.jump_market_drop_lo, cfg.jump_market_drop_hi, size=(b, days))
                mkt_path = mkt_path + _price_drop_to_logshock(drop) * jump_mkt_mask

            # Build per-symbol step returns: (b, days, S)
            step_ret = np.zeros((b, days, len(syms)), dtype=np.float64)
            for i, s in enumerate(syms):
                d = sym_data[i]
                res_path = self._sample_block_bootstrap(d["resid"], float(base_scores[i]), days, block, b)
                res_path = res_path * float(cfg.residual_scale)
                sim_r = beta[i] * mkt_path + res_path + drifts[i]
                if cfg.use_jumps:
                    u2 = self.rng.random((b, days))
                    jump_idio_mask = u2 < cfg.p_jump_idio
                    drop2 = self.rng.uniform(cfg.jump_idio_drop_lo, cfg.jump_idio_drop_hi, size=(b, days))
                    sim_r = sim_r + _price_drop_to_logshock(drop2) * jump_idio_mask
                step_ret[:, :, i] = np.exp(sim_r) - 1.0

            equity = np.ones(b, dtype=np.float64)
            pos_notional = equity[:, None] * w_target[None, :]
            turnover = np.zeros(b, dtype=np.float64)
            cost_acc = np.zeros(b, dtype=np.float64)
            alive = np.ones(b, dtype=bool)

            for t in range(days):
                r_t = step_ret[:, t, :]
                pnl = np.sum(pos_notional * r_t, axis=1)
                equity = equity + pnl
                pos_notional = pos_notional * (1.0 + r_t)

                # liquidation proxy
                liq_mask = (equity <= (1.0 + acc_liq_thresh)) & alive
                if np.any(liq_mask):
                    alive[liq_mask] = False
                    pos_notional[liq_mask] = 0.0

                if not np.any(alive):
                    break

                # Rebalance at interval
                if ((t + 1) % reb_interval) == 0:
                    if cfg.rebalance_score_noise > 0:
                        noise = self.rng.normal(0.0, float(cfg.rebalance_score_noise), size=base_scores.shape)
                        scores_eff = base_scores + noise
                    else:
                        scores_eff = base_scores
                    w_target = self._weights_from_scores(scores_eff)
                    target_notional = equity[:, None] * w_target[None, :]
                    delta = target_notional - pos_notional
                    turnover_step = np.sum(np.abs(delta), axis=1)
                    cost_step = turnover_step * cost_rate
                    equity = equity - cost_step
                    cost_acc = cost_acc + cost_step
                    turnover = turnover + turnover_step
                    pos_notional = target_notional

            port_pnls[done:done + b] = equity - 1.0
            turnovers[done:done + b] = turnover
            costs[done:done + b] = cost_acc
            done += b

        exp_pnl = float(np.mean(port_pnls))
        turn_mean = float(np.mean(turnovers))
        turn_p95 = float(np.quantile(turnovers, 0.95))
        cost_mean = float(np.mean(costs))
        prob_acc_liq = float(np.mean(port_pnls <= acc_liq_thresh))

        return {
            "expected_portfolio_pnl": exp_pnl,
            "turnover_mean": turn_mean,
            "turnover_p95": turn_p95,
            "cost_mean": cost_mean,
            "prob_account_liq_proxy": prob_acc_liq,
            "steps": int(days),
            "total_leverage_allocated": float(total_lev),
        }
