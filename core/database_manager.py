"""
Quant Bot을 위한 고성능 SQLite 데이터베이스 매니저.
기존의 JSON 파일 기반 저장을 대체하여 데이터 무결성과 I/O 성능을 보장합니다.

주요 기능:
- 거래 기록 (Trade Tape): 체결 이력 저장
- 자산 기록 (Equity History): 시계열 자산 추적
- 포지션 기록 (Position History): 포지션 스냅샷 및 변경 이력
- 봇 상태 (Bot State): KV 스토어 (포지션, 설정 등)
- 진단 데이터 (Diagnostics): 내부 메트릭, 슬리피지 분석 등

Paper vs Live 모드 차이점 처리:
- Paper: 추정 진입가, 추정 슬리피지, 시뮬레이션 fill
- Live: 실제 진입가, 실제 슬리피지(목표가-체결가), 거래소 fill

Author: codex_quant
Date: 2026-01-19
"""

import sqlite3
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from threading import Lock
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """트레이딩 모드 열거형"""
    PAPER = "paper"
    LIVE = "live"


class DatabaseManager:
    """
    Quant Bot을 위한 고성능 SQLite 데이터베이스 매니저.
    
    테이블 구조:
    1. trades: 체결 기록 (Paper/Live 공통, 모드 구분 컬럼 포함)
    2. equity_history: 자산 시계열
    3. positions: 현재 포지션 스냅샷
    4. position_history: 포지션 변경 이력
    5. bot_state: KV 스토어
    6. slippage_analysis: 슬리피지 분석 (Live 전용)
    7. diagnostics: 내부 진단 메트릭
    8. exit_policy_state: Exit policy 상태 (영속화)
    """
    
    def __init__(self, db_path: str = "state/bot_data.db"):
        self.db_path = db_path
        self.lock = Lock()
        
        # DB 디렉토리 생성
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._initialize_db()
        logger.info(f"DatabaseManager initialized: {db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """DB 연결을 생성하고 최적화 설정을 적용합니다."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # dict-like access
        # WAL 모드: 읽기/쓰기 동시성 향상 및 속도 최적화
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _initialize_db(self):
        """필요한 테이블들을 생성합니다."""
        with self.lock, self._get_connection() as conn:
            cursor = conn.cursor()
            
            # ─────────────────────────────────────────────────────────────────
            # 1. 거래 기록 (Trade Tape) - Paper/Live 통합
            # ─────────────────────────────────────────────────────────────────
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    -- 기본 정보
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,  -- 'LONG' or 'SHORT'
                    action TEXT NOT NULL,  -- 'OPEN' or 'CLOSE'
                    
                    -- 가격 정보
                    target_price REAL,      -- 목표가 (주문 시점)
                    fill_price REAL,        -- 체결가 (실제)
                    qty REAL NOT NULL,
                    notional REAL,          -- qty * fill_price
                    
                    -- 슬리피지 분석 (Live에서 중요)
                    slippage_bps REAL,      -- (fill_price - target_price) / target_price * 10000
                    slippage_est_bps REAL,  -- Paper: 추정 슬리피지 / Live: 사전 추정치
                    
                    -- 비용
                    fee REAL DEFAULT 0,
                    fee_rate REAL,
                    
                    -- 메타데이터
                    trading_mode TEXT NOT NULL,  -- 'paper' or 'live'
                    pos_source TEXT,            -- source of position data (e.g. 'simulator'/'exchange')
                    exec_type TEXT,          -- 'maker' or 'taker'
                    order_id TEXT,
                    timestamp_ms INTEGER NOT NULL,
                    
                    -- 진입 컨텍스트 (Paper/Live 공통으로 영속화)
                    entry_group TEXT,
                    entry_rank INTEGER,
                    entry_reason TEXT,
                    entry_ev REAL,
                    entry_kelly REAL,
                    entry_confidence REAL,
                    
                    -- PnL (CLOSE 시에만)
                    realized_pnl REAL,
                    roe REAL,
                    hold_duration_sec REAL,
                    
                    -- 원본 데이터 (디버깅용)
                    raw_data TEXT
                )
            """)
            
            # ─────────────────────────────────────────────────────────────────
            # 2. 자산 가치 기록 (Equity History)
            # ─────────────────────────────────────────────────────────────────
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS equity_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_ms INTEGER NOT NULL,
                    trading_mode TEXT NOT NULL,
                    pos_source TEXT,
                    
                    -- 자산 정보
                    total_equity REAL NOT NULL,
                    wallet_balance REAL,
                    available_balance REAL,
                    unrealized_pnl REAL,
                    
                    -- 포지션 요약
                    position_count INTEGER,
                    total_notional REAL,
                    total_margin REAL,
                    
                    -- 위험 지표
                    margin_ratio REAL,
                    total_leverage REAL,
                    
                    UNIQUE(timestamp_ms, trading_mode)
                )
            """)
            
            # ─────────────────────────────────────────────────────────────────
            # 3. 현재 포지션 (Positions) - 최신 스냅샷
            # ─────────────────────────────────────────────────────────────────
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    trading_mode TEXT NOT NULL,
                    pos_source TEXT,
                    
                    -- 기본 정보
                    side TEXT NOT NULL,
                    size REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL,
                    leverage REAL,
                    
                    -- 마진/노티널
                    margin REAL,
                    notional REAL,
                    cap_frac REAL,
                    
                    -- PnL
                    unrealized_pnl REAL,
                    roe REAL,
                    max_roe REAL,
                    
                    -- 타이밍
                    entry_time_ms INTEGER,
                    last_update_ms INTEGER,
                    age_sec REAL,
                    
                    -- 진입 컨텍스트 (Paper/Live 공통 영속화)
                    entry_group TEXT,
                    entry_rank INTEGER,
                    entry_order INTEGER,
                    entry_t_star REAL,
                    entry_reason TEXT,
                    
                    -- Exit Policy 상태 (재시작 시 복원용)
                    policy_horizon_sec INTEGER,
                    policy_flip_streak INTEGER,
                    policy_hold_bad INTEGER,
                    policy_last_eval_ms INTEGER,
                    policy_mu_annual REAL,
                    policy_sigma_annual REAL,
                    
                    -- 원본 (호환성)
                    raw_data TEXT
                )
            """)
            
            # ─────────────────────────────────────────────────────────────────
            # 4. 포지션 변경 이력 (Position History)
            # ─────────────────────────────────────────────────────────────────
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS position_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    trading_mode TEXT NOT NULL,
                    pos_source TEXT,
                    event_type TEXT NOT NULL,  -- 'OPEN', 'CLOSE', 'UPDATE', 'LIQUIDATION'
                    timestamp_ms INTEGER NOT NULL,
                    
                    -- 스냅샷 데이터
                    side TEXT,
                    size REAL,
                    entry_price REAL,
                    exit_price REAL,
                    leverage REAL,
                    
                    -- PnL
                    realized_pnl REAL,
                    unrealized_pnl REAL,
                    roe REAL,
                    
                    -- 이유
                    reason TEXT,
                    
                    -- 전체 스냅샷
                    snapshot_data TEXT
                )
            """)
            
            # ─────────────────────────────────────────────────────────────────
            # 5. 봇 상태 저장소 (KV Store)
            # ─────────────────────────────────────────────────────────────────
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bot_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at_ms INTEGER
                )
            """)
            
            # ─────────────────────────────────────────────────────────────────
            # 6. 슬리피지 분석 (Live 전용 - 목표가 vs 체결가 추적)
            # ─────────────────────────────────────────────────────────────────
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS slippage_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp_ms INTEGER NOT NULL,
                    
                    -- 가격 비교
                    target_price REAL NOT NULL,
                    fill_price REAL NOT NULL,
                    slippage_bps REAL NOT NULL,
                    
                    -- 예측 vs 실제
                    estimated_slippage_bps REAL,  -- 시스템 사전 추정
                    estimation_error_bps REAL,    -- 추정 오차
                    
                    -- 컨텍스트
                    side TEXT,
                    exec_type TEXT,
                    volatility REAL,
                    spread_bps REAL,
                    order_size REAL,
                    
                    -- 시장 상태
                    bid_price REAL,
                    ask_price REAL,
                    volume_24h REAL
                )
            """)
            
            # ─────────────────────────────────────────────────────────────────
            # 7. 진단 메트릭 (Diagnostics)
            # ─────────────────────────────────────────────────────────────────
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS diagnostics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_ms INTEGER NOT NULL,
                    metric_type TEXT NOT NULL,  -- 'engine', 'risk', 'execution', etc.
                    symbol TEXT,
                    
                    -- 메트릭 값들
                    data TEXT NOT NULL,  -- JSON
                    
                    UNIQUE(timestamp_ms, metric_type, symbol)
                )
            """)
            
            # ─────────────────────────────────────────────────────────────────
            # 8. EVPH/Score 히스토리 (기존 JSON 대체)
            # ─────────────────────────────────────────────────────────────────
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evph_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_ms INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    
                    ev_per_hour REAL,
                    ev_score REAL,
                    confidence REAL,
                    kelly REAL,
                    regime TEXT,
                    
                    -- 상세 데이터
                    details TEXT,  -- JSON
                    
                    UNIQUE(timestamp_ms, symbol)
                )
            """)
            
            # ─────────────────────────────────────────────────────────────────
            # 인덱스 생성 (조회 속도 향상)
            # ─────────────────────────────────────────────────────────────────
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(timestamp_ms);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_mode ON trades(trading_mode);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_history(timestamp_ms);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_mode ON equity_history(trading_mode);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pos_hist_symbol ON position_history(symbol);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pos_hist_ts ON position_history(timestamp_ms);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_slippage_symbol ON slippage_analysis(symbol);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_slippage_ts ON slippage_analysis(timestamp_ms);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_diag_ts ON diagnostics(timestamp_ms);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_diag_type ON diagnostics(metric_type);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_evph_ts ON evph_history(timestamp_ms);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_evph_symbol ON evph_history(symbol);")
            
            conn.commit()
            logger.info("Database tables initialized successfully")

    # ═══════════════════════════════════════════════════════════════════════
    # 거래 기록 메서드 (Trade Tape)
    # ═══════════════════════════════════════════════════════════════════════
    
    def log_trade(self, trade_data: Dict[str, Any], mode: TradingMode = TradingMode.PAPER):
        """
        체결된 거래를 DB에 기록합니다.
        
        Paper vs Live 차이:
        - Paper: target_price와 fill_price가 슬리피지 시뮬레이션으로 계산됨
        - Live: target_price는 주문가, fill_price는 거래소에서 받은 실제 체결가
        """
        with self.lock, self._get_connection() as conn:
            # 슬리피지 계산
            target = trade_data.get('target_price') or trade_data.get('price', 0)
            fill = trade_data.get('fill_price') or trade_data.get('price', 0)
            slippage_bps = 0
            if target and target > 0:
                slippage_bps = (fill - target) / target * 10000
            
            conn.execute("""
                INSERT INTO trades (
                    symbol, side, action, target_price, fill_price, qty, notional,
                    slippage_bps, slippage_est_bps, fee, fee_rate,
                    trading_mode, pos_source, exec_type, order_id, timestamp_ms,
                    entry_group, entry_rank, entry_reason, entry_ev, entry_kelly, entry_confidence,
                    realized_pnl, roe, hold_duration_sec, raw_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data.get('symbol'),
                trade_data.get('side'),
                trade_data.get('action', 'OPEN'),
                target,
                fill,
                float(trade_data.get('qty', 0)),
                float(trade_data.get('notional', 0)),
                slippage_bps,
                trade_data.get('slippage_est_bps'),
                float(trade_data.get('fee', 0)),
                trade_data.get('fee_rate'),
                mode.value,
                trade_data.get('exec_type'),
                trade_data.get('pos_source'),
                trade_data.get('order_id', ''),
                trade_data.get('timestamp_ms', int(time.time() * 1000)),
                trade_data.get('entry_group'),
                trade_data.get('entry_rank'),
                trade_data.get('entry_reason'),
                trade_data.get('entry_ev'),
                trade_data.get('entry_kelly'),
                trade_data.get('entry_confidence'),
                trade_data.get('realized_pnl'),
                trade_data.get('roe'),
                trade_data.get('hold_duration_sec'),
                json.dumps(trade_data)
            ))
            conn.commit()

    def get_recent_trades(self, limit: int = 100, mode: Optional[TradingMode] = None,
                          symbol: Optional[str] = None) -> List[Dict]:
        """최근 거래 내역을 불러옵니다."""
        with self.lock, self._get_connection() as conn:
            query = "SELECT * FROM trades WHERE 1=1"
            params = []
            
            if mode:
                query += " AND trading_mode = ?"
                params.append(mode.value)
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            query += " ORDER BY timestamp_ms DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows][::-1]

    # ═══════════════════════════════════════════════════════════════════════
    # 자산 기록 메서드 (Equity History)
    # ═══════════════════════════════════════════════════════════════════════
    
    def log_equity(self, equity_data: Dict[str, Any], mode: TradingMode = TradingMode.PAPER):
        """현재 자산 상태를 시계열로 기록합니다."""
        with self.lock, self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO equity_history (
                    timestamp_ms, trading_mode, pos_source,
                    total_equity, wallet_balance, available_balance, unrealized_pnl,
                    position_count, total_notional, total_margin,
                    margin_ratio, total_leverage
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                equity_data.get('timestamp_ms', int(time.time() * 1000)),
                mode.value,
                equity_data.get('pos_source'),
                equity_data.get('total_equity'),
                equity_data.get('wallet_balance'),
                equity_data.get('available_balance'),
                equity_data.get('unrealized_pnl'),
                equity_data.get('position_count'),
                equity_data.get('total_notional'),
                equity_data.get('total_margin'),
                equity_data.get('margin_ratio'),
                equity_data.get('total_leverage'),
            ))
            conn.commit()

    def get_equity_history(self, limit: int = 1000, mode: Optional[TradingMode] = None,
                           since_ms: Optional[int] = None) -> List[Dict]:
        """차트 그리기용 자산 이력을 불러옵니다."""
        with self.lock, self._get_connection() as conn:
            query = "SELECT * FROM equity_history WHERE 1=1"
            params = []
            
            if mode:
                query += " AND trading_mode = ?"
                params.append(mode.value)
            if since_ms:
                query += " AND timestamp_ms >= ?"
                params.append(since_ms)
            
            query += " ORDER BY timestamp_ms ASC LIMIT ?"
            params.append(limit)
            
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    # ═══════════════════════════════════════════════════════════════════════
    # 포지션 관리 메서드
    # ═══════════════════════════════════════════════════════════════════════
    
    def save_position(self, symbol: str, pos_data: Dict[str, Any], mode: TradingMode = TradingMode.PAPER):
        """포지션을 저장합니다 (upsert)."""
        with self.lock, self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO positions (
                    symbol, trading_mode, pos_source, side, size, entry_price, current_price, leverage,
                    margin, notional, cap_frac, unrealized_pnl, roe, max_roe,
                    entry_time_ms, last_update_ms, age_sec,
                    entry_group, entry_rank, entry_order, entry_t_star, entry_reason,
                    policy_horizon_sec, policy_flip_streak, policy_hold_bad,
                    policy_last_eval_ms, policy_mu_annual, policy_sigma_annual,
                    raw_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                mode.value,
                pos_data.get('pos_source'),
                pos_data.get('side'),
                pos_data.get('size') or pos_data.get('quantity'),
                pos_data.get('entry_price'),
                pos_data.get('current') or pos_data.get('price'),
                pos_data.get('leverage'),
                pos_data.get('margin'),
                pos_data.get('notional'),
                pos_data.get('cap_frac'),
                pos_data.get('unrealized_pnl') or pos_data.get('pnl'),
                pos_data.get('roe'),
                pos_data.get('max_roe'),
                pos_data.get('time') or pos_data.get('entry_time_ms'),
                int(time.time() * 1000),
                pos_data.get('age_sec'),
                pos_data.get('entry_group'),
                pos_data.get('entry_rank'),
                pos_data.get('entry_order'),
                pos_data.get('entry_t_star'),
                pos_data.get('reason') or pos_data.get('entry_reason'),
                pos_data.get('policy_horizon_sec'),
                pos_data.get('policy_flip_streak'),
                pos_data.get('policy_hold_bad'),
                pos_data.get('policy_last_eval_ms'),
                pos_data.get('policy_mu_annual'),
                pos_data.get('policy_sigma_annual'),
                json.dumps(pos_data)
            ))
            conn.commit()

    def delete_position(self, symbol: str):
        """포지션을 삭제합니다."""
        with self.lock, self._get_connection() as conn:
            conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            conn.commit()

    def get_all_positions(self, mode: Optional[TradingMode] = None) -> Dict[str, Dict]:
        """모든 포지션을 불러옵니다."""
        with self.lock, self._get_connection() as conn:
            query = "SELECT * FROM positions"
            params = []
            
            if mode:
                query += " WHERE trading_mode = ?"
                params.append(mode.value)
            
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            result = {}
            for row in cursor.fetchall():
                data = dict(row)
                # raw_data가 있으면 파싱하여 병합
                if data.get('raw_data'):
                    try:
                        raw = json.loads(data['raw_data'])
                        raw.update({k: v for k, v in data.items() if k != 'raw_data' and v is not None})
                        result[data['symbol']] = raw
                    except json.JSONDecodeError:
                        result[data['symbol']] = data
                else:
                    result[data['symbol']] = data
            return result

    def log_position_event(self, symbol: str, event_type: str, snapshot: Dict[str, Any],
                           mode: TradingMode = TradingMode.PAPER):
        """포지션 이벤트를 히스토리에 기록합니다."""
        with self.lock, self._get_connection() as conn:
            conn.execute("""
                INSERT INTO position_history (
                    symbol, trading_mode, pos_source, event_type, timestamp_ms,
                    side, size, entry_price, exit_price, leverage,
                    realized_pnl, unrealized_pnl, roe, reason, snapshot_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                mode.value,
                snapshot.get('pos_source'),
                event_type,
                int(time.time() * 1000),
                snapshot.get('side'),
                snapshot.get('size') or snapshot.get('quantity'),
                snapshot.get('entry_price'),
                snapshot.get('exit_price') or snapshot.get('current'),
                snapshot.get('leverage'),
                snapshot.get('realized_pnl'),
                snapshot.get('unrealized_pnl') or snapshot.get('pnl'),
                snapshot.get('roe'),
                snapshot.get('reason'),
                json.dumps(snapshot)
            ))
            conn.commit()

    # ═══════════════════════════════════════════════════════════════════════
    # 슬리피지 분석 (Live 모드 전용)
    # ═══════════════════════════════════════════════════════════════════════
    
    def log_slippage(self, slippage_data: Dict[str, Any]):
        """
        슬리피지 데이터를 기록합니다.
        Live 모드에서 목표가 vs 체결가 차이 분석용.
        """
        with self.lock, self._get_connection() as conn:
            target = slippage_data['target_price']
            fill = slippage_data['fill_price']
            actual_slip = (fill - target) / target * 10000 if target else 0
            est_slip = slippage_data.get('estimated_slippage_bps', 0)
            
            conn.execute("""
                INSERT INTO slippage_analysis (
                    symbol, timestamp_ms, target_price, fill_price, slippage_bps,
                    estimated_slippage_bps, estimation_error_bps,
                    side, exec_type, volatility, spread_bps, order_size,
                    bid_price, ask_price, volume_24h
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                slippage_data['symbol'],
                slippage_data.get('timestamp_ms', int(time.time() * 1000)),
                target,
                fill,
                actual_slip,
                est_slip,
                actual_slip - est_slip if est_slip else None,
                slippage_data.get('side'),
                slippage_data.get('exec_type'),
                slippage_data.get('volatility'),
                slippage_data.get('spread_bps'),
                slippage_data.get('order_size'),
                slippage_data.get('bid_price'),
                slippage_data.get('ask_price'),
                slippage_data.get('volume_24h'),
            ))
            conn.commit()

    def get_slippage_stats(self, symbol: Optional[str] = None, 
                           days: int = 7) -> Dict[str, Any]:
        """슬리피지 통계를 계산합니다."""
        with self.lock, self._get_connection() as conn:
            since_ms = int((time.time() - days * 86400) * 1000)
            
            query = """
                SELECT 
                    COUNT(*) as count,
                    AVG(slippage_bps) as avg_slippage,
                    AVG(ABS(slippage_bps)) as avg_abs_slippage,
                    MAX(slippage_bps) as max_slippage,
                    MIN(slippage_bps) as min_slippage,
                    AVG(estimation_error_bps) as avg_est_error
                FROM slippage_analysis
                WHERE timestamp_ms >= ?
            """
            params = [since_ms]
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row else {}

    # ═══════════════════════════════════════════════════════════════════════
    # 상태 저장 (KV Store)
    # ═══════════════════════════════════════════════════════════════════════
    
    def save_state(self, key: str, data: Any):
        """복잡한 객체를 JSON으로 저장합니다."""
        with self.lock, self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO bot_state (key, value, updated_at_ms)
                VALUES (?, ?, ?)
            """, (key, json.dumps(data), int(time.time() * 1000)))
            conn.commit()

    def load_state(self, key: str, default: Any = None) -> Any:
        """저장된 상태를 불러옵니다."""
        with self.lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM bot_state WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                try:
                    return json.loads(row[0])
                except json.JSONDecodeError:
                    return default
            return default

    def delete_state(self, key: str):
        """저장된 상태를 삭제합니다."""
        with self.lock, self._get_connection() as conn:
            conn.execute("DELETE FROM bot_state WHERE key = ?", (key,))
            conn.commit()

    # ═══════════════════════════════════════════════════════════════════════
    # 진단 메트릭 (Diagnostics)
    # ═══════════════════════════════════════════════════════════════════════
    
    def log_diagnostic(self, metric_type: str, data: Dict[str, Any], symbol: Optional[str] = None):
        """진단 메트릭을 기록합니다."""
        with self.lock, self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO diagnostics (timestamp_ms, metric_type, symbol, data)
                VALUES (?, ?, ?, ?)
            """, (
                int(time.time() * 1000),
                metric_type,
                symbol,
                json.dumps(data)
            ))
            conn.commit()

    def get_diagnostics(self, metric_type: Optional[str] = None, 
                        symbol: Optional[str] = None,
                        limit: int = 100) -> List[Dict]:
        """진단 메트릭을 조회합니다."""
        with self.lock, self._get_connection() as conn:
            query = "SELECT * FROM diagnostics WHERE 1=1"
            params = []
            
            if metric_type:
                query += " AND metric_type = ?"
                params.append(metric_type)
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            query += " ORDER BY timestamp_ms DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            result = []
            for row in cursor.fetchall():
                d = dict(row)
                if d.get('data'):
                    try:
                        d['data'] = json.loads(d['data'])
                    except json.JSONDecodeError:
                        pass
                result.append(d)
            return result

    # ═══════════════════════════════════════════════════════════════════════
    # EVPH/Score 히스토리
    # ═══════════════════════════════════════════════════════════════════════
    
    def log_evph(self, symbol: str, evph_data: Dict[str, Any]):
        """EVPH 데이터를 기록합니다."""
        with self.lock, self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO evph_history (
                    timestamp_ms, symbol, ev_per_hour, ev_score, confidence, kelly, regime, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                evph_data.get('timestamp_ms', int(time.time() * 1000)),
                symbol,
                evph_data.get('ev_per_hour'),
                evph_data.get('ev_score'),
                evph_data.get('confidence'),
                evph_data.get('kelly'),
                evph_data.get('regime'),
                json.dumps(evph_data.get('details', {}))
            ))
            conn.commit()

    def get_evph_history(self, symbol: Optional[str] = None, limit: int = 1000) -> List[Dict]:
        """EVPH 히스토리를 조회합니다."""
        with self.lock, self._get_connection() as conn:
            query = "SELECT * FROM evph_history WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            query += " ORDER BY timestamp_ms DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()][::-1]

    # ═══════════════════════════════════════════════════════════════════════
    # 유틸리티 메서드
    # ═══════════════════════════════════════════════════════════════════════
    
    def get_balance(self, mode: TradingMode = TradingMode.PAPER) -> float:
        """현재 잔고를 불러옵니다."""
        balance = self.load_state(f"balance_{mode.value}")
        return float(balance) if balance is not None else 10000.0  # 기본값

    def save_balance(self, balance: float, mode: TradingMode = TradingMode.PAPER):
        """현재 잔고를 저장합니다."""
        self.save_state(f"balance_{mode.value}", balance)

    def vacuum(self):
        """데이터베이스를 최적화합니다."""
        with self.lock, self._get_connection() as conn:
            conn.execute("VACUUM")
            logger.info("Database vacuumed")

    def get_stats(self) -> Dict[str, int]:
        """테이블별 레코드 수를 반환합니다."""
        with self.lock, self._get_connection() as conn:
            cursor = conn.cursor()
            tables = ['trades', 'equity_history', 'positions', 'position_history',
                      'bot_state', 'slippage_analysis', 'diagnostics', 'evph_history']
            stats = {}
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]
            return stats


# ═══════════════════════════════════════════════════════════════════════════
# 싱글톤 인스턴스
# ═══════════════════════════════════════════════════════════════════════════

_db_instance: Optional[DatabaseManager] = None


def get_db(db_path: str = "state/bot_data.db") -> DatabaseManager:
    """
    DatabaseManager 싱글톤 인스턴스를 반환합니다.
    
    Usage:
        from core.database_manager import get_db, TradingMode
        
        db = get_db()
        db.log_trade({...}, mode=TradingMode.PAPER)
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager(db_path)
    return _db_instance
