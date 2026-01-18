#!/usr/bin/env python3
"""
JSON → SQLite 마이그레이션 스크립트

기존 state/ 디렉토리의 JSON 파일들을 SQLite 데이터베이스로 마이그레이션합니다.

Usage:
    python scripts/migrate_json_to_sqlite.py
    python scripts/migrate_json_to_sqlite.py --dry-run  # 실제 저장 없이 테스트
    python scripts/migrate_json_to_sqlite.py --backup   # 마이그레이션 후 JSON 백업

Author: codex_quant
Date: 2026-01-19
"""

import json
import sys
import time
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.database_manager import get_db, TradingMode


def load_json_safe(filepath: Path) -> any:
    """JSON 파일을 안전하게 로드합니다."""
    if not filepath.exists():
        print(f"  ⚠️  파일 없음: {filepath}")
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                print(f"  ⚠️  빈 파일: {filepath}")
                return None
            return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON 파싱 오류: {filepath} - {e}")
        return None


def migrate_positions(db, state_dir: Path, dry_run: bool = False) -> int:
    """paper_positions.json → positions 테이블"""
    print("\n📦 포지션 마이그레이션...")
    filepath = state_dir / "paper_positions.json"
    data = load_json_safe(filepath)
    
    if not data:
        return 0
    
    count = 0
    if isinstance(data, dict):
        for symbol, pos in data.items():
            if dry_run:
                print(f"  [DRY-RUN] 포지션: {symbol}")
            else:
                db.save_position(symbol, pos, mode=TradingMode.PAPER)
                db.log_position_event(symbol, "MIGRATE", pos, mode=TradingMode.PAPER)
            count += 1
    
    print(f"  ✅ 포지션 {count}개 마이그레이션 완료")
    return count


def migrate_balance(db, state_dir: Path, dry_run: bool = False) -> bool:
    """paper_balance.json → bot_state (key: balance_paper)"""
    print("\n💰 잔고 마이그레이션...")
    filepath = state_dir / "paper_balance.json"
    data = load_json_safe(filepath)
    
    if data is None:
        return False
    
    # 숫자만 있는 경우
    if isinstance(data, (int, float)):
        balance = float(data)
    elif isinstance(data, dict):
        balance = data.get('balance') or data.get('total_equity') or 10000.0
    else:
        print(f"  ⚠️  알 수 없는 형식: {type(data)}")
        return False
    
    if dry_run:
        print(f"  [DRY-RUN] 잔고: {balance}")
    else:
        db.save_balance(balance, mode=TradingMode.PAPER)
    
    print(f"  ✅ 잔고 마이그레이션 완료: ${balance:,.2f}")
    return True


def migrate_trade_tape(db, state_dir: Path, dry_run: bool = False) -> int:
    """paper_trade_tape.json → trades 테이블"""
    print("\n📝 거래 기록 마이그레이션...")
    
    count = 0
    for filename in ["paper_trade_tape.json", "trades.json"]:
        filepath = state_dir / filename
        data = load_json_safe(filepath)
        
        if not data:
            continue
        
        if isinstance(data, list):
            for trade in data:
                if dry_run:
                    print(f"  [DRY-RUN] 거래: {trade.get('symbol', 'N/A')}")
                else:
                    # 기존 필드 매핑
                    trade_data = {
                        'symbol': trade.get('symbol'),
                        'side': trade.get('side'),
                        'action': trade.get('action', 'OPEN'),
                        'target_price': trade.get('target_price') or trade.get('price'),
                        'fill_price': trade.get('fill_price') or trade.get('price'),
                        'qty': trade.get('qty') or trade.get('quantity') or trade.get('size'),
                        'notional': trade.get('notional'),
                        'fee': trade.get('fee', 0),
                        'exec_type': trade.get('exec_type'),
                        'order_id': trade.get('order_id', ''),
                        'timestamp_ms': trade.get('timestamp_ms') or trade.get('timestamp', 0) * 1000 if trade.get('timestamp') else int(time.time() * 1000),
                        'entry_reason': trade.get('reason'),
                        'realized_pnl': trade.get('realized_pnl') or trade.get('pnl'),
                        'roe': trade.get('roe'),
                    }
                    db.log_trade(trade_data, mode=TradingMode.PAPER)
                count += 1
    
    print(f"  ✅ 거래 기록 {count}개 마이그레이션 완료")
    return count


def migrate_equity_history(db, state_dir: Path, dry_run: bool = False) -> int:
    """paper_equity_history.json → equity_history 테이블"""
    print("\n📈 Equity 히스토리 마이그레이션...")
    filepath = state_dir / "paper_equity_history.json"
    data = load_json_safe(filepath)
    
    if not data:
        return 0
    
    count = 0
    if isinstance(data, list):
        for entry in data:
            if dry_run:
                print(f"  [DRY-RUN] Equity: {entry.get('total_equity', 'N/A')}")
            else:
                equity_data = {
                    'timestamp_ms': entry.get('timestamp_ms') or entry.get('time', 0) * 1000 if entry.get('time') else int(time.time() * 1000),
                    'total_equity': entry.get('total_equity') or entry.get('equity'),
                    'wallet_balance': entry.get('wallet_balance') or entry.get('cash'),
                    'available_balance': entry.get('available_balance'),
                    'unrealized_pnl': entry.get('unrealized_pnl') or entry.get('unreal'),
                }
                db.log_equity(equity_data, mode=TradingMode.PAPER)
            count += 1
    
    print(f"  ✅ Equity 히스토리 {count}개 마이그레이션 완료")
    return count


def migrate_evph_history(db, state_dir: Path, dry_run: bool = False) -> int:
    """evph_history.json, score_history.json → evph_history 테이블"""
    print("\n📊 EVPH/Score 히스토리 마이그레이션...")
    
    count = 0
    for filename in ["evph_history.json", "score_history.json"]:
        filepath = state_dir / filename
        data = load_json_safe(filepath)
        
        if not data:
            continue
        
        if isinstance(data, list):
            for entry in data:
                if dry_run:
                    print(f"  [DRY-RUN] EVPH: {entry.get('symbol', 'N/A')}")
                else:
                    symbol = entry.get('symbol', 'UNKNOWN')
                    evph_data = {
                        'timestamp_ms': entry.get('timestamp_ms') or entry.get('time', 0) * 1000 if entry.get('time') else int(time.time() * 1000),
                        'ev_per_hour': entry.get('ev_per_hour') or entry.get('evph'),
                        'ev_score': entry.get('ev_score'),
                        'confidence': entry.get('confidence') or entry.get('conf'),
                        'kelly': entry.get('kelly'),
                        'regime': entry.get('regime'),
                        'details': entry,
                    }
                    db.log_evph(symbol, evph_data)
                count += 1
        elif isinstance(data, dict):
            # 심볼별로 저장된 경우
            for symbol, entries in data.items():
                if isinstance(entries, list):
                    for entry in entries:
                        if dry_run:
                            print(f"  [DRY-RUN] EVPH: {symbol}")
                        else:
                            evph_data = {
                                'timestamp_ms': entry.get('timestamp_ms') or int(time.time() * 1000),
                                'ev_per_hour': entry.get('ev_per_hour') or entry.get('evph'),
                                'ev_score': entry.get('ev_score'),
                                'confidence': entry.get('confidence'),
                                'kelly': entry.get('kelly'),
                                'regime': entry.get('regime'),
                                'details': entry,
                            }
                            db.log_evph(symbol, evph_data)
                        count += 1
    
    print(f"  ✅ EVPH/Score 히스토리 {count}개 마이그레이션 완료")
    return count


def backup_json_files(state_dir: Path):
    """마이그레이션 전 JSON 파일 백업"""
    backup_dir = state_dir / f"backup_pre_sqlite_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(exist_ok=True)
    
    json_files = [
        "paper_positions.json",
        "paper_balance.json",
        "paper_trade_tape.json",
        "paper_equity_history.json",
        "evph_history.json",
        "score_history.json",
        "trades.json",
    ]
    
    for filename in json_files:
        src = state_dir / filename
        if src.exists():
            shutil.copy2(src, backup_dir / filename)
            print(f"  📁 백업: {filename}")
    
    print(f"\n✅ 백업 완료: {backup_dir}")
    return backup_dir


def main():
    parser = argparse.ArgumentParser(description="JSON → SQLite 마이그레이션")
    parser.add_argument("--dry-run", action="store_true", help="실제 저장 없이 테스트")
    parser.add_argument("--backup", action="store_true", help="마이그레이션 전 JSON 백업")
    parser.add_argument("--db-path", default="state/bot_data.db", help="SQLite DB 경로")
    args = parser.parse_args()
    
    state_dir = PROJECT_ROOT / "state"
    
    print("=" * 60)
    print("🚀 JSON → SQLite 마이그레이션 시작")
    print("=" * 60)
    print(f"상태 디렉토리: {state_dir}")
    print(f"DB 경로: {args.db_path}")
    print(f"Dry-run: {args.dry_run}")
    
    if args.backup and not args.dry_run:
        print("\n📁 JSON 파일 백업 중...")
        backup_json_files(state_dir)
    
    # 데이터베이스 초기화
    db_path = str(PROJECT_ROOT / args.db_path)
    db = get_db(db_path)
    
    # 마이그레이션 실행
    results = {
        'positions': migrate_positions(db, state_dir, args.dry_run),
        'balance': migrate_balance(db, state_dir, args.dry_run),
        'trades': migrate_trade_tape(db, state_dir, args.dry_run),
        'equity': migrate_equity_history(db, state_dir, args.dry_run),
        'evph': migrate_evph_history(db, state_dir, args.dry_run),
    }
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 마이그레이션 결과 요약")
    print("=" * 60)
    for key, value in results.items():
        status = "✅" if value else "⚠️"
        print(f"  {status} {key}: {value}")
    
    if not args.dry_run:
        # DB 통계 출력
        stats = db.get_stats()
        print("\n📈 데이터베이스 통계:")
        for table, count in stats.items():
            print(f"  - {table}: {count} 레코드")
    
    print("\n✅ 마이그레이션 완료!")
    
    if args.dry_run:
        print("\n⚠️  --dry-run 모드였습니다. 실제로 마이그레이션하려면 플래그를 제거하세요.")


if __name__ == "__main__":
    main()
