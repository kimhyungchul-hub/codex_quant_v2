"""
Continuous Opportunity Replacement Manager

Checks at every decision point whether existing G1 positions should be
replaced with better opportunities based on optimal horizon scoring.
"""

from __future__ import annotations
from typing import Dict, Set, Tuple, Optional, Any
import time


class ContinuousOpportunityChecker:
    """
    Manages continuous opportunity replacement for TOP10 positions.
    
    Checks Score_B > Score_A + Transaction_Cost to decide replacements.
    """
    
    def __init__(self, orchestrator):
        self.orch = orchestrator
        # Switching_Buffer: The integral score improvement needed to justify a switch.
        # Since scores are area (avg_return * duration), a buffer of 0.5 (e.g. 0.05% * 1000s) is reasonable.
        self.buffer = 0.5
    
    async def check_and_replace_if_better(
        self,
        sym: str,
        detail: Optional[Dict[str, Any]],
        ts_ms: int
    ) -> bool:
        """
        Check if a new candidate should replace an existing G1 position.
        
        Args:
            sym: Symbol being evaluated
            detail: Decision detail for the symbol
            ts_ms: Current timestamp
        
        Returns:
            True if replacement occurred, False otherwise
        """
        # Only consider TOP10 candidates
        group_info = self.orch._group_info.get(sym, {})
        if group_info.get("group") != "TOP10":
            return False
        
        # Get current TOP10 positions
        current_top10 = set()
        for s, pos in self.orch.positions.items():
            entry_group = pos.get("entry_group")
            if entry_group == "TOP10":
                current_top10.add(s)
        
        # If we have fewer than max positions, don't replace
        max_positions = getattr(self.orch, '_max_positions', 10)
        if len(current_top10) < max_positions:
            return False
        
        # If this symbol is already in TOP10, don't check replacement
        if sym in current_top10:
            return False
        
        # Calculate score for new candidate (sym)
        score_new = self.orch._get_symbol_expected_value(sym, detail, use_optimal_horizon=True)
        
        # Find the worst existing TOP10 position
        worst_sym = None
        worst_score = float('inf')
        
        for existing_sym in current_top10:
            cached = self.orch._decision_cache.get(existing_sym) or {}
            existing_decision = cached.get("decision")
            existing_score = self.orch._get_symbol_expected_value(
                existing_sym,
                existing_decision if isinstance(existing_decision, dict) else {},
                use_optimal_horizon=True,
            )
            
            if existing_score < worst_score:
                worst_score = existing_score
                worst_sym = existing_sym
        
        # Switching buffer is in Score_A units (area), not bps.
        switching_cost = float(self.buffer)
        
        # Decision: Replace if Score_new > Score_worst + Switching_Cost
        gain = score_new - worst_score
        if gain > switching_cost:
            # Replace worst with new
            self.orch._log(
                f"🔄 [OPPORTUNITY_REPLACE] {sym} (Score={score_new:.6f}) "
                f"replaces {worst_sym} (Score={worst_score:.6f}) | "
                f"Gain={gain:.6f} > Cost={switching_cost:.6f}"
            )
            
            # Close worst position
            worst_pos = self.orch.positions.get(worst_sym)
            if worst_pos:
                mark_price = worst_pos.get("mark_price")
                if not mark_price:
                    mark_price = worst_pos.get("current") or worst_pos.get("price") or worst_pos.get("entry_price")
                if mark_price:
                    await self.orch._close_position(
                        sym=worst_sym,
                        exit_price=float(mark_price),
                        ts_ms=ts_ms,
                        reason=f"Opportunity replacement: {sym} has better score (+{score_new - worst_score:.3f}%)"
                    )
                    return True
        
        return False
