#!/usr/bin/env python3
"""
최근 2~3배치(~360건) 청산 거래 종합 진단 스크립트
- 배치별 성과 비교
- 종목별 분석
- 청산 로직(exit reason)별 분석
- 자본금 투입 전/후 비교
- 점수/지표 분포 비교
"""
import sqlite3, json, os, sys
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

DB = "state/bot_data_live.db"

def ts_to_dt(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000)

def safe_json(raw):
    if not raw:
        return {}
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except:
        return {}

def main():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    # ===== 1. 전체 개요 =====
    print("=" * 100)
    print("🔍 최근 거래 종합 진단 리포트")
    print("=" * 100)

    # 총 거래 수
    total_closes = conn.execute("SELECT COUNT(*) FROM trades WHERE action != 'OPEN'").fetchone()[0]
    total_opens = conn.execute("SELECT COUNT(*) FROM trades WHERE action = 'OPEN'").fetchone()[0]
    print(f"\n📊 전체 거래: OPEN={total_opens}, CLOSE={total_closes}")

    # 시간 범위
    time_range = conn.execute("SELECT MIN(timestamp_ms), MAX(timestamp_ms) FROM trades").fetchone()
    print(f"⏰ 기간: {ts_to_dt(time_range[0])} ~ {ts_to_dt(time_range[1])}")

    # ===== 2. 최근 360건 청산 거래 추출 =====
    recent_closes = conn.execute("""
        SELECT * FROM trades 
        WHERE action != 'OPEN' 
        ORDER BY timestamp_ms DESC 
        LIMIT 360
    """).fetchall()
    
    print(f"\n최근 360건 청산 거래 시간 범위:")
    if recent_closes:
        print(f"  최신: {ts_to_dt(recent_closes[0]['timestamp_ms'])}")
        print(f"  최오래: {ts_to_dt(recent_closes[-1]['timestamp_ms'])}")

    # ===== 3. 120건씩 3개 배치로 분할 =====
    batches = []
    for i in range(3):
        start = i * 120
        end = min((i + 1) * 120, len(recent_closes))
        if start < len(recent_closes):
            batch = recent_closes[start:end]
            batches.append(batch)
    
    print(f"\n📦 배치 분할: {len(batches)}개 배치 (각 ~120건)")
    
    for bi, batch in enumerate(batches):
        period_start = ts_to_dt(batch[-1]['timestamp_ms'])
        period_end = ts_to_dt(batch[0]['timestamp_ms'])
        print(f"\n{'='*100}")
        print(f"📦 배치 {bi+1} (최신이 배치1) | {period_start.strftime('%m/%d %H:%M')} ~ {period_end.strftime('%m/%d %H:%M')} | {len(batch)}건")
        print(f"{'='*100}")
        
        analyze_batch(batch, bi + 1, conn)

    # ===== 4. 자본금 투입 전/후 비교 =====
    print(f"\n{'='*100}")
    print("💰 자본금 $500 투입 전/후 비교")
    print(f"{'='*100}")
    
    # $500 투입 시점 추정 — equity history에서 점프 찾기
    analyze_capital_injection(conn, recent_closes)

    # ===== 5. auto_tune_overrides 분석 =====
    print(f"\n{'='*100}")
    print("⚙️ Auto-Tune Overrides 분석")
    print(f"{'='*100}")
    analyze_overrides()

    # ===== 6. 배치간 수치 비교 테이블 =====
    print(f"\n{'='*100}")
    print("📊 배치간 핵심 지표 비교 매트릭스")
    print(f"{'='*100}")
    compare_batches(batches)

    # ===== 7. Loss Driver 분석 =====
    print(f"\n{'='*100}")
    print("🔴 손실 원인 Top-10 분석")
    print(f"{'='*100}")
    analyze_loss_drivers(recent_closes, conn)

    # ===== 8. 기존 리포트 파일 요약 =====
    print(f"\n{'='*100}")
    print("📎 기존 리포트 파일 핵심 수치")
    print(f"{'='*100}")
    summarize_existing_reports()

    conn.close()


def analyze_batch(batch, batch_num, conn):
    """배치 하나에 대한 상세 분석"""
    
    # --- PnL 기본 통계 ---
    pnls = [r['realized_pnl'] for r in batch if r['realized_pnl'] is not None]
    roes = [r['roe'] for r in batch if r['roe'] is not None]
    holds = [r['hold_duration_sec'] for r in batch if r['hold_duration_sec'] is not None]
    
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    
    print(f"\n  💵 PnL 기본 통계:")
    print(f"    총 PnL: ${sum(pnls):.4f}")
    print(f"    평균 PnL: ${statistics.mean(pnls):.6f}" if pnls else "    PnL 데이터 없음")
    print(f"    중앙값 PnL: ${statistics.median(pnls):.6f}" if pnls else "")
    print(f"    승률: {len(wins)}/{len(pnls)} = {len(wins)/len(pnls)*100:.1f}%" if pnls else "")
    print(f"    평균 이익: ${statistics.mean(wins):.6f} ({len(wins)}건)" if wins else "    이익 거래 없음")
    print(f"    평균 손실: ${statistics.mean(losses):.6f} ({len(losses)}건)" if losses else "    손실 거래 없음")
    
    if wins and losses:
        profit_factor = abs(sum(wins) / sum(losses)) if sum(losses) != 0 else float('inf')
        print(f"    Profit Factor: {profit_factor:.3f}")
        avg_win_loss_ratio = abs(statistics.mean(wins) / statistics.mean(losses)) if statistics.mean(losses) != 0 else float('inf')
        print(f"    평균 W/L 비율: {avg_win_loss_ratio:.3f}")
    
    # --- ROE 통계 ---
    if roes:
        print(f"\n  📈 ROE 통계:")
        print(f"    평균 ROE: {statistics.mean(roes)*100:.4f}%")
        print(f"    중앙값 ROE: {statistics.median(roes)*100:.4f}%")
        print(f"    최대 이익: {max(roes)*100:.4f}%")
        print(f"    최대 손실: {min(roes)*100:.4f}%")
        print(f"    ROE StdDev: {statistics.stdev(roes)*100:.4f}%" if len(roes) > 1 else "")
    
    # --- Hold Duration ---
    if holds:
        print(f"\n  ⏱️ 보유 시간:")
        print(f"    평균: {statistics.mean(holds):.0f}초 ({statistics.mean(holds)/60:.1f}분)")
        print(f"    중앙값: {statistics.median(holds):.0f}초 ({statistics.median(holds)/60:.1f}분)")
        print(f"    최소: {min(holds):.0f}초 | 최대: {max(holds):.0f}초")
    
    # --- Notional 규모 ---
    notionals = [r['notional'] for r in batch if r['notional'] is not None]
    if notionals:
        print(f"\n  💎 포지션 규모:")
        print(f"    평균 Notional: ${statistics.mean(notionals):.2f}")
        print(f"    중앙값 Notional: ${statistics.median(notionals):.2f}")
        print(f"    최대 Notional: ${max(notionals):.2f}")
    
    # --- 종목별 성과 ---
    print(f"\n  🏷️ 종목별 성과:")
    symbol_stats = defaultdict(lambda: {"pnls": [], "cnt": 0, "wins": 0})
    for r in batch:
        s = r['symbol']
        symbol_stats[s]['cnt'] += 1
        if r['realized_pnl'] is not None:
            symbol_stats[s]['pnls'].append(r['realized_pnl'])
            if r['realized_pnl'] > 0:
                symbol_stats[s]['wins'] += 1
    
    sorted_symbols = sorted(symbol_stats.items(), key=lambda x: sum(x[1]['pnls']), reverse=True)
    print(f"    {'종목':<14} {'건수':>4} {'승률':>7} {'총PnL':>12} {'평균PnL':>12}")
    print(f"    {'-'*53}")
    for sym, stats in sorted_symbols:
        total_pnl = sum(stats['pnls'])
        avg_pnl = statistics.mean(stats['pnls']) if stats['pnls'] else 0
        wr = stats['wins'] / len(stats['pnls']) * 100 if stats['pnls'] else 0
        marker = "🟢" if total_pnl > 0 else "🔴"
        print(f"    {marker} {sym:<12} {stats['cnt']:>4} {wr:>6.1f}% ${total_pnl:>10.4f} ${avg_pnl:>10.6f}")
    
    # --- 청산 사유(entry_reason/action)별 분석 ---
    print(f"\n  🚪 청산 사유별 분석:")
    exit_stats = defaultdict(lambda: {"pnls": [], "cnt": 0, "wins": 0})
    for r in batch:
        raw = safe_json(r['raw_data'])
        exit_reason = raw.get('exit_reason', r['action'] or 'UNKNOWN')
        exit_stats[exit_reason]['cnt'] += 1
        if r['realized_pnl'] is not None:
            exit_stats[exit_reason]['pnls'].append(r['realized_pnl'])
            if r['realized_pnl'] > 0:
                exit_stats[exit_reason]['wins'] += 1
    
    sorted_exits = sorted(exit_stats.items(), key=lambda x: sum(x[1]['pnls']), reverse=True)
    print(f"    {'청산사유':<28} {'건수':>4} {'승률':>7} {'총PnL':>12} {'평균PnL':>12}")
    print(f"    {'-'*67}")
    for reason, stats in sorted_exits:
        total_pnl = sum(stats['pnls'])
        avg_pnl = statistics.mean(stats['pnls']) if stats['pnls'] else 0
        wr = stats['wins'] / len(stats['pnls']) * 100 if stats['pnls'] else 0
        marker = "🟢" if total_pnl > 0 else "🔴"
        print(f"    {marker} {reason:<26} {stats['cnt']:>4} {wr:>6.1f}% ${total_pnl:>10.4f} ${avg_pnl:>10.6f}")

    # --- Side 분석 ---
    side_stats = defaultdict(lambda: {"pnls": [], "cnt": 0, "wins": 0})
    for r in batch:
        side = r['side'] or 'UNKNOWN'
        side_stats[side]['cnt'] += 1
        if r['realized_pnl'] is not None:
            side_stats[side]['pnls'].append(r['realized_pnl'])
            if r['realized_pnl'] > 0:
                side_stats[side]['wins'] += 1
    
    print(f"\n  ↕️ Side별 분석:")
    for side, stats in sorted(side_stats.items()):
        total_pnl = sum(stats['pnls'])
        wr = stats['wins'] / len(stats['pnls']) * 100 if stats['pnls'] else 0
        print(f"    {side}: {stats['cnt']}건, 승률 {wr:.1f}%, 총PnL ${total_pnl:.4f}")

    # --- Entry EV / Kelly / Confidence 분석 ---
    evs = [r['entry_ev'] for r in batch if r['entry_ev'] is not None]
    kellys = [r['entry_kelly'] for r in batch if r['entry_kelly'] is not None]
    confs = [r['entry_confidence'] for r in batch if r['entry_confidence'] is not None]
    
    print(f"\n  📐 진입 점수 분석:")
    if evs:
        print(f"    Entry EV    - Mean: {statistics.mean(evs):.6f}, Median: {statistics.median(evs):.6f}, Std: {statistics.stdev(evs):.6f}" if len(evs) > 1 else f"    Entry EV    - Mean: {statistics.mean(evs):.6f}")
    if kellys:
        print(f"    Entry Kelly - Mean: {statistics.mean(kellys):.6f}, Median: {statistics.median(kellys):.6f}")
    if confs:
        print(f"    Confidence  - Mean: {statistics.mean(confs):.6f}, Median: {statistics.median(confs):.6f}")

    # --- Leverage 분석 ---
    leverages = []
    for r in batch:
        raw = safe_json(r['raw_data'])
        lev = raw.get('leverage') or raw.get('leverage_used')
        if lev is not None:
            leverages.append(float(lev))
    if leverages:
        print(f"\n  ⚡ 레버리지 분석:")
        print(f"    평균: {statistics.mean(leverages):.2f}x, 중앙값: {statistics.median(leverages):.2f}x, 최대: {max(leverages):.2f}x")

    # --- Slippage 분석 ---
    slippages = [r['slippage_bps'] for r in batch if r['slippage_bps'] is not None]
    if slippages:
        print(f"\n  📉 슬리피지:")
        print(f"    평균: {statistics.mean(slippages):.2f} bps, 중앙값: {statistics.median(slippages):.2f} bps")

    # --- Alpha/Regime 분석 ---
    regimes = defaultdict(int)
    vpins = []
    hursts = []
    for r in batch:
        if r['regime']:
            regimes[r['regime']] += 1
        if r['alpha_vpin'] is not None:
            vpins.append(r['alpha_vpin'])
        if r['alpha_hurst'] is not None:
            hursts.append(r['alpha_hurst'])
    
    if regimes:
        print(f"\n  🌊 Regime 분포: {dict(regimes)}")
    if vpins:
        print(f"    VPIN 평균: {statistics.mean(vpins):.4f}")
    if hursts:
        print(f"    Hurst 평균: {statistics.mean(hursts):.4f}")

    # --- Direction Hit Rate (raw_data에서) ---
    dir_hits = 0
    dir_total = 0
    for r in batch:
        raw = safe_json(r['raw_data'])
        dh = raw.get('direction_hit')
        if dh is not None:
            dir_total += 1
            if dh:
                dir_hits += 1
    if dir_total > 0:
        print(f"\n  🎯 Direction Hit Rate: {dir_hits}/{dir_total} = {dir_hits/dir_total*100:.1f}%")

    # --- Entry Quality / One-Way Move / Leverage Signal ---
    eq_scores = [r['entry_quality_score'] for r in batch if r['entry_quality_score'] is not None]
    owm_scores = [r['one_way_move_score'] for r in batch if r['one_way_move_score'] is not None]
    lev_scores = [r['leverage_signal_score'] for r in batch if r['leverage_signal_score'] is not None]
    
    if eq_scores or owm_scores or lev_scores:
        print(f"\n  🔬 품질 점수:")
    if eq_scores:
        print(f"    Entry Quality    - Mean: {statistics.mean(eq_scores):.4f}, Median: {statistics.median(eq_scores):.4f}")
    if owm_scores:
        print(f"    One-Way Move     - Mean: {statistics.mean(owm_scores):.4f}, Median: {statistics.median(owm_scores):.4f}")
    if lev_scores:
        print(f"    Leverage Signal  - Mean: {statistics.mean(lev_scores):.4f}, Median: {statistics.median(lev_scores):.4f}")


def analyze_capital_injection(conn, recent_closes):
    """자본금 투입 전/후 성과 비교"""
    # equity_history에서 큰 점프 감지
    eq_rows = conn.execute("""
        SELECT timestamp_ms, total_equity, unrealized_pnl
        FROM equity_history 
        ORDER BY timestamp_ms
    """).fetchall()
    
    if not eq_rows:
        print("  equity_history 데이터 없음")
        return
    
    # 점프 감지 (equity가 50% 이상 증가한 시점)
    injection_ts = None
    for i in range(1, len(eq_rows)):
        prev = eq_rows[i-1]['total_equity']
        curr = eq_rows[i]['total_equity']
        if prev and curr and prev > 0:
            jump = (curr - prev) / prev
            if jump > 0.3 and (curr - prev) > 100:  # 30% 이상 + $100 이상 증가
                injection_ts = eq_rows[i]['timestamp_ms']
                print(f"  💰 자본금 투입 감지: {ts_to_dt(injection_ts)}")
                print(f"     ${prev:.2f} → ${curr:.2f} (점프: +${curr-prev:.2f}, +{jump*100:.1f}%)")
                break
    
    if not injection_ts:
        # 점프를 못 찾으면 전체 equity 추이 표시
        print("  자본금 투입 시점을 equity에서 감지하지 못함.")
        print(f"  Equity 범위: ${eq_rows[0]['total_equity']:.2f} ~ ${eq_rows[-1]['total_equity']:.2f}")
        # 대안: 일자별 equity 추이 표시
        print(f"\n  📈 일자별 Equity 추이 (최근 10일):")
        daily_eq = {}
        for r in eq_rows:
            dt = ts_to_dt(r['timestamp_ms']).strftime('%m/%d')
            daily_eq[dt] = r['total_equity']
        for dt, eq in list(daily_eq.items())[-10:]:
            print(f"    {dt}: ${eq:.2f}")
        return
    
    # 투입 전 120건 / 투입 후 120건 비교
    before = conn.execute("""
        SELECT * FROM trades 
        WHERE action != 'OPEN' AND timestamp_ms < ?
        ORDER BY timestamp_ms DESC LIMIT 120
    """, (injection_ts,)).fetchall()
    
    after = conn.execute("""
        SELECT * FROM trades 
        WHERE action != 'OPEN' AND timestamp_ms >= ?
        ORDER BY timestamp_ms ASC LIMIT 120
    """, (injection_ts,)).fetchall()
    
    def batch_summary(data, label):
        pnls = [r['realized_pnl'] for r in data if r['realized_pnl'] is not None]
        roes = [r['roe'] for r in data if r['roe'] is not None]
        notionals = [r['notional'] for r in data if r['notional'] is not None]
        wins = [p for p in pnls if p > 0]
        
        if not pnls:
            print(f"  {label}: 데이터 없음")
            return
        
        period = f"{ts_to_dt(data[-1]['timestamp_ms']).strftime('%m/%d %H:%M')} ~ {ts_to_dt(data[0]['timestamp_ms']).strftime('%m/%d %H:%M')}"
        if label == "투입 후":
            period = f"{ts_to_dt(data[0]['timestamp_ms']).strftime('%m/%d %H:%M')} ~ {ts_to_dt(data[-1]['timestamp_ms']).strftime('%m/%d %H:%M')}"
        
        print(f"\n  {label} ({len(data)}건, {period}):")
        print(f"    총 PnL: ${sum(pnls):.4f}")
        print(f"    승률: {len(wins)/len(pnls)*100:.1f}%")
        print(f"    평균 ROE: {statistics.mean(roes)*100:.4f}%" if roes else "")
        print(f"    평균 Notional: ${statistics.mean(notionals):.2f}" if notionals else "")
        if wins and len([p for p in pnls if p <= 0]) > 0:
            losers = [p for p in pnls if p <= 0]
            pf = abs(sum(wins) / sum(losers)) if sum(losers) != 0 else float('inf')
            print(f"    Profit Factor: {pf:.3f}")
    
    batch_summary(before, "투입 전")
    batch_summary(after, "투입 후")
    
    # 투입 후 전체(모든 거래)
    all_after = conn.execute("""
        SELECT * FROM trades 
        WHERE action != 'OPEN' AND timestamp_ms >= ?
        ORDER BY timestamp_ms ASC
    """, (injection_ts,)).fetchall()
    
    if all_after:
        pnls = [r['realized_pnl'] for r in all_after if r['realized_pnl'] is not None]
        print(f"\n  투입 후 전체 ({len(all_after)}건):")
        print(f"    총 PnL: ${sum(pnls):.4f}")
        wins = [p for p in pnls if p > 0]
        print(f"    승률: {len(wins)/len(pnls)*100:.1f}%" if pnls else "")


def analyze_overrides():
    """auto_tune_overrides.json 분석"""
    try:
        with open("state/auto_tune_overrides.json") as f:
            overrides = json.load(f)
        
        if not overrides:
            print("  override 없음 (빈 파일)")
            return
        
        print(f"  총 {len(overrides)}개 override 적용 중:")
        # 주요 파라미터 표시
        important_keys = [
            'MAX_LEVERAGE', 'LEVERAGE_TARGET_MAX', 'MC_TP_BASE_ROE', 'MC_SL_BASE_ROE',
            'NOTIONAL_HARD_CAP_USD', 'TOP_N_SYMBOLS', 'POLICY_HORIZON_SEC',
            'MAX_POSITION_HOLD_SEC', 'HYBRID_EXIT_SCORE_FLOOR', 'UNIFIED_ENTRY_FLOOR',
            'ev_entry_threshold', 'CONFIRM_TICK_ev_drop', 'CONFIRM_TICK_hybrid_exit',
            'MAX_NOTIONAL_PER_SYMBOL', 'BASE_LEVERAGE', 'K_LEV',
        ]
        
        for k in important_keys:
            if k in overrides:
                print(f"    {k} = {overrides[k]}")
        
        # 분류
        categories = defaultdict(list)
        for k, v in overrides.items():
            if 'LEV' in k.upper() or 'LEVERAGE' in k.upper():
                categories['레버리지'].append(f"{k}={v}")
            elif 'TP' in k.upper() or 'SL' in k.upper() or 'EXIT' in k.upper():
                categories['출구전략'].append(f"{k}={v}")
            elif 'EV' in k.upper() or 'SCORE' in k.upper() or 'THRESHOLD' in k.upper():
                categories['진입필터'].append(f"{k}={v}")
            elif 'CONFIRM' in k.upper() or 'TICK' in k.upper():
                categories['확인틱'].append(f"{k}={v}")
            else:
                categories['기타'].append(f"{k}={v}")
        
        for cat, items in categories.items():
            print(f"\n    [{cat}] ({len(items)}개):")
            for item in items[:8]:
                print(f"      {item}")
            if len(items) > 8:
                print(f"      ... +{len(items)-8}개")
    except Exception as e:
        print(f"  오류: {e}")


def compare_batches(batches):
    """배치간 핵심 지표 비교"""
    headers = ["지표", "배치1(최신)", "배치2", "배치3(오래됨)", "추세"]
    
    def get_batch_metrics(batch):
        pnls = [r['realized_pnl'] for r in batch if r['realized_pnl'] is not None]
        roes = [r['roe'] for r in batch if r['roe'] is not None]
        holds = [r['hold_duration_sec'] for r in batch if r['hold_duration_sec'] is not None]
        notionals = [r['notional'] for r in batch if r['notional'] is not None]
        evs = [r['entry_ev'] for r in batch if r['entry_ev'] is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        # Direction hit from raw_data
        dir_hits = 0
        dir_total = 0
        for r in batch:
            raw = safe_json(r['raw_data'])
            dh = raw.get('direction_hit')
            if dh is not None:
                dir_total += 1
                if dh:
                    dir_hits += 1
        
        metrics = {
            "총PnL ($)": f"{sum(pnls):.4f}" if pnls else "N/A",
            "승률 (%)": f"{len(wins)/len(pnls)*100:.1f}" if pnls else "N/A",
            "평균ROE (%)": f"{statistics.mean(roes)*100:.4f}" if roes else "N/A",
            "PF": f"{abs(sum(wins)/sum(losses)):.3f}" if wins and losses and sum(losses) != 0 else "N/A",
            "평균PnL ($)": f"{statistics.mean(pnls):.6f}" if pnls else "N/A",
            "평균Hold (분)": f"{statistics.mean(holds)/60:.1f}" if holds else "N/A",
            "평균Notional ($)": f"{statistics.mean(notionals):.2f}" if notionals else "N/A",
            "평균EV": f"{statistics.mean(evs):.6f}" if evs else "N/A",
            "DirHitRate (%)": f"{dir_hits/dir_total*100:.1f}" if dir_total > 0 else "N/A",
            "거래수": str(len(batch)),
        }
        return metrics
    
    all_metrics = [get_batch_metrics(b) for b in batches]
    
    print(f"\n  {'지표':<20}", end="")
    for i in range(len(batches)):
        print(f" {'배치'+str(i+1):>14}", end="")
    print()
    print(f"  {'-'*20}", end="")
    for i in range(len(batches)):
        print(f" {'-'*14}", end="")
    print()
    
    for key in all_metrics[0].keys():
        print(f"  {key:<20}", end="")
        vals = []
        for i, m in enumerate(all_metrics):
            v = m[key]
            print(f" {v:>14}", end="")
            try:
                vals.append(float(v))
            except:
                vals.append(None)
        
        # 추세 화살표
        if len(vals) >= 2 and vals[0] is not None and vals[-1] is not None:
            if vals[0] > vals[-1]:
                print(" ↑ 개선" if key not in ["평균Hold (분)"] else " ↑")
            elif vals[0] < vals[-1]:
                print(" ↓ 악화" if key not in ["평균Hold (분)"] else " ↓")
            else:
                print(" →")
        else:
            print()
    

def analyze_loss_drivers(recent_closes, conn):
    """손실 원인 Top-10"""
    # PnL 기준 최악의 거래들
    worst_trades = sorted(recent_closes, key=lambda r: r['realized_pnl'] if r['realized_pnl'] is not None else 0)[:10]
    
    print(f"\n  최악 PnL Top-10:")
    print(f"  {'종목':<14} {'Side':<6} {'PnL':>10} {'ROE':>8} {'Hold':>8} {'Exit':>16} {'Notional':>10}")
    print(f"  {'-'*76}")
    for r in worst_trades:
        raw = safe_json(r['raw_data'])
        exit_r = raw.get('exit_reason', r['action'] or '?')
        hold = f"{r['hold_duration_sec']:.0f}s" if r['hold_duration_sec'] else "?"
        pnl = r['realized_pnl'] if r['realized_pnl'] is not None else 0
        roe = f"{r['roe']*100:.2f}%" if r['roe'] else "?"
        notional = f"${r['notional']:.2f}" if r['notional'] else "?"
        print(f"  {r['symbol']:<14} {r['side'] or '?':<6} ${pnl:>9.4f} {roe:>8} {hold:>8} {exit_r:>16} {notional:>10}")
    
    # 청산 로직별 손실 기여도
    exit_loss_contribution = defaultdict(float)
    for r in recent_closes:
        raw = safe_json(r['raw_data'])
        exit_reason = raw.get('exit_reason', r['action'] or 'UNKNOWN')
        if r['realized_pnl'] is not None and r['realized_pnl'] < 0:
            exit_loss_contribution[exit_reason] += r['realized_pnl']
    
    print(f"\n  청산 로직별 손실 기여도:")
    for reason, total_loss in sorted(exit_loss_contribution.items(), key=lambda x: x[1]):
        print(f"    {reason:<28}: ${total_loss:.4f}")


def summarize_existing_reports():
    """기존 리포트 파일들의 핵심 수치 요약"""
    report_files = [
        ("state/post_500_loss_driver_report.json", "$500 투입 후 손실 드라이버"),
        ("state/reval_loss_driver_history.json", "Reval Loss Driver History"),
        ("state/auto_reval_db_report.json", "Auto Reval DB Report"),
        ("state/counterfactual_replay_report_latest500.json", "Counterfactual Replay (최근500)"),
        ("state/entry_exit_diagnosis_report_live_now.json", "진입/청산 진단 (Live)"),
        ("state/mu_direction_tuning_report.json", "Mu Direction Tuning"),
        ("state/min_notional_tuning_report.json", "Min Notional Tuning"),
        ("state/trade_observability_report_now_run.json", "Trade Observability"),
    ]
    
    for path, label in report_files:
        try:
            with open(path) as f:
                data = json.load(f)
            
            print(f"\n  📄 {label} ({os.path.basename(path)}):")
            
            # 핵심 수치 추출
            if isinstance(data, dict):
                for key in ['direction_hit', 'direction_hit_rate', 'entry_issue_ratio', 
                           'avg_exit_regret', 'win_rate', 'profit_factor', 'total_pnl',
                           'avg_roe', 'avg_hold_sec', 'ready', 'new_closed_total',
                           'total_trades', 'total_exits', 'summary', 'kpi', 'overall']:
                    if key in data:
                        val = data[key]
                        if isinstance(val, dict):
                            for k2, v2 in list(val.items())[:6]:
                                print(f"    {key}.{k2}: {v2}")
                        else:
                            print(f"    {key}: {val}")
                
                # progress 섹션
                if 'progress' in data and isinstance(data['progress'], dict):
                    for k, v in list(data['progress'].items())[:5]:
                        print(f"    progress.{k}: {v}")
                
                # last_batch_kpi
                prog = data.get('progress', {})
                if isinstance(prog, dict) and 'last_batch_kpi' in prog:
                    for k, v in prog['last_batch_kpi'].items():
                        print(f"    last_batch_kpi.{k}: {v}")
        except Exception as e:
            print(f"\n  📄 {label}: 읽기 실패 ({e})")


if __name__ == "__main__":
    os.chdir("/Users/jeonghwakim/codex_quant_clean")
    main()
