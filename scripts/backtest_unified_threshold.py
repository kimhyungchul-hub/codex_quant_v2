#!/usr/bin/env python3
"""
UnifiedScore 백테스팅 및 최적 Threshold 탐색

사용법:
  python scripts/backtest_unified_threshold.py [--db trading_bot.db]

기능:
  - SQLite 거래 히스토리에서 UnifiedScore와 realized_r 관계 분석
  - 여러 threshold 후보에 대해 승률/수익률 시뮬레이션
  - 최적 threshold 자동 추천
"""

import argparse
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

def load_trade_history(db_path: Path) -> pd.DataFrame:
    """SQLite에서 거래 히스토리 로드"""
    try:
        conn = sqlite3.connect(db_path)
        
        # trades 테이블에서 진입/청산 쌍 추출
        query = """
        SELECT 
            symbol,
            timestamp_ms,
            action,
            side,
            price,
            quantity,
            pnl,
            metadata
        FROM trades
        WHERE mode = 'paper'
        ORDER BY timestamp_ms ASC
        """
        
        df = conn.execute(query).fetchall()
        conn.close()
        
        if not df:
            print("[WARN] 거래 데이터가 없습니다.")
            return pd.DataFrame()
        
        # DataFrame 변환
        df = pd.DataFrame(df, columns=[
            "symbol", "timestamp_ms", "action", "side", 
            "price", "quantity", "pnl", "metadata"
        ])
        
        return df
    
    except Exception as e:
        print(f"[ERROR] DB 로드 실패: {e}")
        return pd.DataFrame()

def extract_unified_scores(df: pd.DataFrame) -> List[Tuple[float, float]]:
    """진입 시점 UnifiedScore와 realized_r 추출"""
    import json
    
    score_return_pairs = []
    
    # ENTER 액션만 필터링
    entries = df[df["action"] == "ENTER"].copy()
    
    for idx, row in entries.iterrows():
        try:
            metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else {}
            
            # UnifiedScore 추출 (metadata에 저장되어 있다고 가정)
            unified_score = metadata.get("unified_score") or metadata.get("ev")
            
            # 해당 포지션의 청산 기록 찾기
            symbol = row["symbol"]
            entry_ts = row["timestamp_ms"]
            
            exit_row = df[
                (df["symbol"] == symbol) &
                (df["action"] == "EXIT") &
                (df["timestamp_ms"] > entry_ts)
            ].iloc[0] if len(df[
                (df["symbol"] == symbol) &
                (df["action"] == "EXIT") &
                (df["timestamp_ms"] > entry_ts)
            ]) > 0 else None
            
            if exit_row is not None:
                exit_metadata = json.loads(exit_row["metadata"]) if isinstance(exit_row["metadata"], str) else {}
                realized_r = exit_metadata.get("realized_r") or (exit_row["pnl"] / row["price"] / row["quantity"])
                
                if unified_score is not None and realized_r is not None:
                    score_return_pairs.append((float(unified_score), float(realized_r)))
        
        except Exception as e:
            continue
    
    return score_return_pairs

def simulate_thresholds(pairs: List[Tuple[float, float]], thresholds: List[float]):
    """각 threshold에 대해 승률/수익률 시뮬레이션"""
    results = []
    
    for thresh in thresholds:
        # threshold 이상인 신호만 필터링
        filtered = [(score, ret) for score, ret in pairs if score >= thresh]
        
        if not filtered:
            results.append({
                "threshold": thresh,
                "n_trades": 0,
                "win_rate": 0.0,
                "mean_return": 0.0,
                "sharpe": 0.0,
            })
            continue
        
        scores, returns = zip(*filtered)
        returns_arr = np.array(returns)
        
        n_trades = len(filtered)
        win_rate = (returns_arr > 0).mean()
        mean_return = returns_arr.mean()
        std_return = returns_arr.std()
        sharpe = (mean_return / std_return) if std_return > 0 else 0.0
        
        results.append({
            "threshold": thresh,
            "n_trades": n_trades,
            "win_rate": win_rate,
            "mean_return": mean_return,
            "sharpe": sharpe,
        })
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="UnifiedScore Threshold 백테스팅")
    parser.add_argument("--db", type=str, default="./state/paper/trading_bot.db", help="SQLite DB 경로")
    args = parser.parse_args()
    
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[ERROR] DB 파일을 찾을 수 없습니다: {db_path}")
        print("  → 실 거래 데이터가 아직 없거나 경로가 잘못되었습니다.")
        return
    
    print("="*70)
    print("📊 UnifiedScore Threshold 백테스팅")
    print("="*70)
    print(f"DB: {db_path}\n")
    
    # 1. 거래 히스토리 로드
    df = load_trade_history(db_path)
    if df.empty:
        print("[ERROR] 거래 데이터를 로드할 수 없습니다.")
        return
    
    print(f"✅ 총 {len(df)} 건의 거래 기록 로드\n")
    
    # 2. UnifiedScore와 realized_r 쌍 추출
    pairs = extract_unified_scores(df)
    if not pairs:
        print("[ERROR] UnifiedScore 데이터를 추출할 수 없습니다.")
        print("  → metadata에 unified_score가 저장되어 있는지 확인하세요.")
        return
    
    print(f"✅ {len(pairs)} 건의 진입-청산 쌍 추출\n")
    
    scores, returns = zip(*pairs)
    scores_arr = np.array(scores)
    returns_arr = np.array(returns)
    
    # 3. 현재 분포 확인
    print("📈 UnifiedScore 분포")
    print("="*70)
    print(f"  Mean      : {scores_arr.mean():.6f}")
    print(f"  Median    : {np.median(scores_arr):.6f}")
    print(f"  Std       : {scores_arr.std():.6f}")
    print(f"  Min       : {scores_arr.min():.6f}")
    print(f"  Max       : {scores_arr.max():.6f}")
    print(f"  P25       : {np.percentile(scores_arr, 25):.6f}")
    print(f"  P50       : {np.percentile(scores_arr, 50):.6f}")
    print(f"  P75       : {np.percentile(scores_arr, 75):.6f}\n")
    
    # 4. Threshold 후보 생성
    thresholds = [
        scores_arr.min(),  # 모든 신호 허용
        np.percentile(scores_arr, 10),
        np.percentile(scores_arr, 25),
        np.percentile(scores_arr, 50),
        scores_arr.mean(),
        np.percentile(scores_arr, 75),
        np.percentile(scores_arr, 90),
    ]
    
    # 5. 시뮬레이션
    print("🔍 Threshold 시뮬레이션")
    print("="*70)
    results_df = simulate_thresholds(pairs, thresholds)
    
    print(results_df.to_string(index=False))
    
    # 6. 최적 threshold 추천
    print("\n" + "="*70)
    print("💡 최적 Threshold 추천")
    print("="*70)
    
    # Sharpe 최대화
    best_by_sharpe = results_df.loc[results_df["sharpe"].idxmax()]
    print(f"\n1. Sharpe Ratio 최대화:")
    print(f"   UNIFIED_ENTRY_FLOOR={best_by_sharpe['threshold']:.6f}")
    print(f"     거래 수: {int(best_by_sharpe['n_trades'])}")
    print(f"     승률   : {best_by_sharpe['win_rate']*100:.1f}%")
    print(f"     평균 수익: {best_by_sharpe['mean_return']*100:.2f}%")
    print(f"     Sharpe : {best_by_sharpe['sharpe']:.2f}")
    
    # 승률 최대화
    best_by_winrate = results_df.loc[results_df["win_rate"].idxmax()]
    print(f"\n2. 승률 최대화:")
    print(f"   UNIFIED_ENTRY_FLOOR={best_by_winrate['threshold']:.6f}")
    print(f"     거래 수: {int(best_by_winrate['n_trades'])}")
    print(f"     승률   : {best_by_winrate['win_rate']*100:.1f}%")
    print(f"     평균 수익: {best_by_winrate['mean_return']*100:.2f}%")
    
    # 평균 수익률 최대화
    best_by_return = results_df.loc[results_df["mean_return"].idxmax()]
    print(f"\n3. 평균 수익률 최대화:")
    print(f"   UNIFIED_ENTRY_FLOOR={best_by_return['threshold']:.6f}")
    print(f"     거래 수: {int(best_by_return['n_trades'])}")
    print(f"     승률   : {best_by_return['win_rate']*100:.1f}%")
    print(f"     평균 수익: {best_by_return['mean_return']*100:.2f}%")
    
    print("\n" + "="*70)
    print("✅ 분석 완료")
    print("="*70)

if __name__ == "__main__":
    main()
