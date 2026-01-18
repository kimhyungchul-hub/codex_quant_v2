"""
Economic NAPV Decision Engine V2

Mathematical Foundation:
    NAPV = ∫[0,T*] (Yield - ρ) * e^(-ρt) dt - Cost

Where:
    - Yield: Expected return rate (directional EV vector)
    - ρ: Opportunity cost (system per-second return rate)
    - Cost: Transaction costs (asymmetric: exit_only vs full)

Key Improvements:
    1. Trapezoidal integration for precision
    2. Direct EV vector injection (no side multiplication)
    3. Per-second ρ for correct time discounting
    4. Returns optimal T* for exit protection
"""
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class EconomicBrain:
    """
    Economic NAPV-based 4-way decision engine.
    
    Evaluates: HOLD, REVERSE, SWITCH, CASH
    """
    
    def __init__(
        self, 
        entry_cost: float = 0.0004,
        exit_cost: float = 0.0004, 
        slippage: float = 0.0002,
        switch_buffer: float = 0.0002
    ):
        """
        Initialize cost structure.
        
        Args:
            entry_cost: Entry fee (bp)
            exit_cost: Exit fee (bp)
            slippage: Slippage cost (bp)
            switch_buffer: Hysteresis buffer for switching (bp)
        """
        self.EXIT_COST = exit_cost
        self.FULL_COST = entry_cost + exit_cost + slippage
        self.BUFFER = switch_buffer
        
    def calculate_napv_precise(
        self,
        horizons: np.ndarray,
        yield_vector: np.ndarray,
        rho: float,
        r_f: float,
        cost_mode: str
    ) -> Dict[str, Any]:
        """
        Calculate Net Added Present Value with time discounting.
        
        Mathematical Formula:
            NAPV = ∫[0,T] (Y(t) - ρ) * e^(-ρt) dt - C
        
        Args:
            horizons: Time array in seconds (e.g., [300, 600, 1800, 3600])
            yield_vector: Expected return rates (already directional)
            rho: System opportunity cost (per-second rate)
            r_f: Risk-free rate (per-second rate)
            cost_mode: 'exit_only' or 'full'
            
        Returns:
            {
                'napv': Maximum NAPV value
                't_star': Optimal holding time (seconds)
                'optimal_idx': Index of optimal horizon
                'napv_curve': Full NPV curve for debugging
            }
        """
        # 1. Excess return: Yield - Opportunity Cost
        excess_yield = yield_vector - rho
        
        # 2. Time value discounting: e^(-ρt)
        discount_factors = np.exp(-rho * horizons)
        discounted_stream = excess_yield * discount_factors
        
        # 3. Numerical integration (Trapezoidal approximation)
        # dt[0] = horizons[0] (area from t=0 to first horizon)
        dt = np.diff(horizons, prepend=0)
        cumulative_napv = np.cumsum(discounted_stream * dt)
        
        # 4. Cost application
        cost = self.EXIT_COST if cost_mode == 'exit_only' else self.FULL_COST
        net_napv_curve = cumulative_napv - cost
        
        # 5. Find optimal T*
        best_idx = np.argmax(net_napv_curve)
        max_napv = net_napv_curve[best_idx]
        t_star = horizons[best_idx]
        
        return {
            'napv': float(max_napv),
            't_star': int(t_star),
            'optimal_idx': int(best_idx),
            'napv_curve': net_napv_curve.tolist()
        }
    
    def evaluate_slot_action(
        self,
        current_slot: Dict[str, Any],
        global_best_candidate: Dict[str, Any],
        horizons: np.ndarray,
        rho: float,
        r_f: float
    ) -> Dict[str, Any]:
        """
        4-Way decision logic: HOLD vs REVERSE vs SWITCH vs CASH.
        
        Args:
            current_slot: {
                'symbol': str,
                'side': int (+1 or -1),
                'ev_vector': np.array (current direction EV),
                'reverse_ev_vector': np.array (opposite direction EV)
            }
            global_best_candidate: {
                'symbol': str,
                'best_ev_vector': np.array (best market opportunity)
            }
            horizons: Time horizons array
            rho: System opportunity cost (per-second)
            r_f: Risk-free rate (per-second)
            
        Returns:
            {
                'action': 'HOLD' | 'REVERSE' | 'SWITCH' | 'CASH',
                'score': Winning NAPV value,
                't_star': Optimal holding time,
                'all_scores': Dict of all 4 scores
            }
        """
        results = {}
        t_stars = {}
        
        # --- Option 1: HOLD (Keep current position) ---
        res_hold = self.calculate_napv_precise(
            horizons,
            current_slot['ev_vector'],
            rho,
            r_f,
            cost_mode='exit_only'
        )
        results['HOLD'] = res_hold['napv']
        t_stars['HOLD'] = res_hold['t_star']
        
        # --- Option 2: REVERSE (Flip direction) ---
        res_reverse = self.calculate_napv_precise(
            horizons,
            current_slot['reverse_ev_vector'],
            rho,
            r_f,
            cost_mode='full'
        )
        results['REVERSE'] = res_reverse['napv']
        t_stars['REVERSE'] = res_reverse['t_star']
        
        # --- Option 3: SWITCH (Replace with global best) ---
        if global_best_candidate['symbol'] == current_slot['symbol']:
            # Already holding the best - switching is meaningless
            results['SWITCH'] = -np.inf
            t_stars['SWITCH'] = 0
        else:
            res_switch = self.calculate_napv_precise(
                horizons,
                global_best_candidate['best_ev_vector'],
                rho,
                r_f,
                cost_mode='full'
            )
            results['SWITCH'] = res_switch['napv']
            t_stars['SWITCH'] = res_switch['t_star']
        
        # --- Option 4: CASH (Liquidate to cash) ---
        cash_vector = np.full_like(horizons, r_f, dtype=float)
        res_cash = self.calculate_napv_precise(
            horizons,
            cash_vector,
            rho,
            r_f,
            cost_mode='exit_only'
        )
        results['CASH'] = res_cash['napv']
        t_stars['CASH'] = res_cash['t_star']
        
        # --- Winner selection with hysteresis ---
        best_action = 'HOLD'
        winning_score = results['HOLD']
        winning_t_star = t_stars['HOLD']
        
        # Aggressive actions need to beat HOLD by buffer amount
        if results['REVERSE'] > winning_score + self.BUFFER:
            best_action = 'REVERSE'
            winning_score = results['REVERSE']
            winning_t_star = t_stars['REVERSE']
        
        if results['SWITCH'] > winning_score + self.BUFFER:
            best_action = 'SWITCH'
            winning_score = results['SWITCH']
            winning_t_star = t_stars['SWITCH']
        
        # Cash is safety net - no buffer required
        if results['CASH'] > winning_score:
            best_action = 'CASH'
            winning_score = results['CASH']
            winning_t_star = t_stars['CASH']
        
        return {
            'action': best_action,
            'score': winning_score,
            't_star': winning_t_star,
            'all_scores': results
        }
