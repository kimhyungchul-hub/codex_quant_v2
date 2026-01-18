    def _recalculate_groups(self, force: bool = False) -> None:
        """
        Recalculate G1/G2 group assignments based on current scores.
        
        G1: Top 5 symbols by score
        G2: Top 6-20 symbols by score
        OTHER: Rest
        
        Updates:
        - self._group_info: {symbol: {group, boost, cap}}
        - self._latest_rankings: [sym1, sym2, ...] ordered by score
        """
        try:
            now = now_ms()
            # Update at most every 2 seconds unless forced
            if not force and (now - self._last_group_update_ts) < 2000:
                return
            
            self._last_group_update_ts = now
            
            # Get all symbols with scores
            scored_symbols = [(sym, self._symbol_scores.get(sym, 0.0)) for sym in self.symbols]
            # Sort by score descending
            scored_symbols.sort(key=lambda x: x[1], reverse=True)
            
            # Update rankings
            self._latest_rankings = [sym for sym, score in scored_symbols]
            
            # Clear and rebuild group_info
            self._group_info.clear()
            
            for i, (sym, score) in enumerate(scored_symbols):
                rank = i + 1
                
                if rank <= 5:
                    # G1: Top 5
                    self._group_info[sym] = {
                        "group": "G1",
                        "rank": rank,
                        "score": score,
                        "boost": 1.0,
                        "cap": 0.3,  # 30% max per G1 position
                    }
                elif rank <= 20:
                    # G2: Top 6-20
                    self._group_info[sym] = {
                        "group": "G2",
                        "rank": rank,
                        "score": score,
                        "boost": 0.8,
                        "cap": 0.15,  # 15% max per G2 position
                    }
                else:
                    # OTHER: Rest
                    self._group_info[sym] = {
                        "group": "OTHER",
                        "rank": rank,
                        "score": score,
                        "boost": 0.5,
                        "cap": 0.05,
                    }
        except Exception as e:
            self._log(f"[ERR] _recalculate_groups: {e}")
    
    def _get_top_k_symbols(self, k: int) -> list[str]:
        """Get top K symbols by current score."""
        return self._latest_rankings[:k] if len(self._latest_rankings) >= k else self._latest_rankings
