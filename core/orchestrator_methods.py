    def _get_current_rank(self, sym: str) -> Optional[int]:
        """
        Get current rank of symbol in latest rankings.
        Returns 1-indexed rank or None if not in rankings.
        """
        try:
            if sym in self._latest_rankings:
                return self._latest_rankings.index(sym) + 1
            return None
        except Exception:
            return None
    
    def _get_validated_entry_group(self, sym: str) -> Optional[str]:
        """
        Get validated entry group for symbol at the moment of entry.
        """
        info = self._group_info.get(sym)
        if info:
            return info.get("group")
        return None
