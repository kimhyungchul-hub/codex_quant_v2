    def evaluate_portfolio_joint(
        self,
        symbols: Optional[List[str]] = None,
        tp_mult: float = 4.0,
        sl_mult: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Evaluate portfolio using joint Monte Carlo simulation.
        
        Uses shared market factor and correlated paths across symbols.
        Returns portfolio weights and risk metrics (VaR, CVaR, liquidation prob).
        
        Args:
            symbols: List of symbols to evaluate (defaults to self.symbols)
            tp_mult: Take profit multiplier (relative to volatility)
            sl_mult: Stop loss multiplier (relative to volatility)
            
        Returns:
            Dict with keys:
                - weights: {symbol: leverage_weight}
                - expected_portfolio_pnl: float
                - var: float (Value at Risk)
                - cvar: float (Conditional VaR)
                - prob_any_position_liquidated: float
                - prob_account_liquidation_proxy: float
                - per_symbol_metrics: {symbol: {...}}
        """
        from engines.mc.portfolio_joint_sim import PortfolioJointSimEngine, PortfolioConfig
        from engines.mc.config import config as mc_config
        
        if symbols is None:
            symbols = self.symbols
        
        # Prepare OHLCV map from buffer
        ohlcv_map = {}
        for sym in symbols:
            candles = self.data.ohlcv_buffer.get(sym, [])
            if candles:
                # Convert to tuples: (o, h, l, c, v)
                ohlcv_map[sym] = [
                    (float(c['open']), float(c['high']), float(c['low']), float(c['close']), float(c['volume']))
                    for c in candles
                ]
        
        # Extract AI scores
        ai_scores = {sym: float(self._symbol_scores.get(sym, 0.0)) for sym in symbols}
        
        # Build config from MC config
        cfg = PortfolioConfig(
            days=mc_config.portfolio_days,
            simulations=mc_config.portfolio_simulations,
            batch_size=mc_config.portfolio_batch_size,
            block_size=mc_config.portfolio_block_size,
            drift_k=mc_config.portfolio_drift_k,
            score_clip=mc_config.portfolio_score_clip,
            tilt_strength=mc_config.portfolio_tilt_strength,
            use_jumps=mc_config.portfolio_use_jumps,
            p_jump_market=mc_config.portfolio_p_jump_market,
            p_jump_idio=mc_config.portfolio_p_jump_idio,
            target_leverage=mc_config.portfolio_target_leverage,
            individual_cap=mc_config.portfolio_individual_cap,
            risk_aversion=mc_config.portfolio_risk_aversion,
            var_alpha=mc_config.portfolio_var_alpha,
            leverage=float(self.max_leverage),
            seed=None,  # Use random seed for each evaluation
        )
        
        # Run joint simulation
        engine = PortfolioJointSimEngine(ohlcv_map, ai_scores, cfg)
        weights, report = engine.build_portfolio(symbols, tp_mult=tp_mult, sl_mult=sl_mult)
        
        self._log(
            f"📊 [PORTFOLIO_JOINT] Weights: {weights} | "
            f"E[PnL]={report['expected_portfolio_pnl']:.4f} | "
            f"VaR(5%)={report['var']:.4f} | CVaR={report['cvar']:.4f} | "
            f"P(liq)={report['prob_any_position_liquidated']:.2%}"
        )
        
        return report
