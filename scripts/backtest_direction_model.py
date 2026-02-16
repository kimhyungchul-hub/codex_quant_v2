#!/usr/bin/env python3
"""Counterfactual backtest: DirectionModel vs historical trades."""
import sqlite3, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "state", "bot_data_live.db")

def main():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    total = c.execute("SELECT COUNT(*) FROM trades WHERE realized_pnl IS NOT NULL AND realized_pnl != 0").fetchone()[0]
    wins = c.execute("SELECT COUNT(*) FROM trades WHERE realized_pnl > 0").fetchone()[0]
    total_pnl_val = c.execute("SELECT COALESCE(SUM(realized_pnl),0) FROM trades WHERE realized_pnl IS NOT NULL AND realized_pnl != 0").fetchone()[0]
    print("=== Overall Stats ===")
    print(f"Total: {total}, WR: {wins}/{total} = {wins/max(1,total)*100:.1f}%, PnL: ${total_pnl_val:.2f}")
    for sl, sv in [("LONG","LONG"),("SHORT","SHORT")]:
        st = c.execute("SELECT COUNT(*) FROM trades WHERE side=? AND realized_pnl IS NOT NULL AND realized_pnl!=0",(sv,)).fetchone()[0]
        sw = c.execute("SELECT COUNT(*) FROM trades WHERE side=? AND realized_pnl>0",(sv,)).fetchone()[0]
        sp = c.execute("SELECT COALESCE(SUM(realized_pnl),0) FROM trades WHERE side=? AND realized_pnl IS NOT NULL AND realized_pnl!=0",(sv,)).fetchone()[0]
        print(f"  {sl}: WR={sw}/{st} ({sw/max(1,st)*100:.1f}%) PnL=${sp:.2f}")

    rows = c.execute("SELECT side,pred_mu_alpha,realized_pnl,regime,entry_confidence,symbol FROM trades WHERE realized_pnl IS NOT NULL AND realized_pnl!=0 AND pred_mu_alpha IS NOT NULL").fetchall()
    al_w,al_t,al_p = 0,0,0.0
    mis_w,mis_t,mis_p = 0,0,0.0
    ns_w,ns_t,ns_p = 0,0,0.0
    regime_st = {}
    for r in rows:
        side,mu,pnl,regime = r["side"],float(r["pred_mu_alpha"]),float(r["realized_pnl"]),str(r["regime"] or "unknown").lower()
        win = pnl > 0
        if regime not in regime_st:
            regime_st[regime] = dict(aw=0,at=0,ap=0.0,mw=0,mt=0,mp=0.0)
        if abs(mu) < 0.01:
            ns_t += 1; ns_p += pnl
            if win: ns_w += 1
        else:
            aligned = (side=="LONG" and mu>0) or (side=="SHORT" and mu<0)
            if aligned:
                al_t+=1; al_p+=pnl; regime_st[regime]["at"]+=1; regime_st[regime]["ap"]+=pnl
                if win: al_w+=1; regime_st[regime]["aw"]+=1
            else:
                mis_t+=1; mis_p+=pnl; regime_st[regime]["mt"]+=1; regime_st[regime]["mp"]+=pnl
                if win: mis_w+=1; regime_st[regime]["mw"]+=1

    print(f"\n=== mu_alpha Alignment (n={len(rows)}) ===")
    print(f"Aligned: {al_t} trades, WR={al_w/max(1,al_t)*100:.1f}%, PnL=${al_p:.2f}")
    print(f"Misaligned: {mis_t} trades, WR={mis_w/max(1,mis_t)*100:.1f}%, PnL=${mis_p:.2f}")
    print(f"Near-zero: {ns_t} trades, WR={ns_w/max(1,ns_t)*100:.1f}%, PnL=${ns_p:.2f}")
    tot_sig = al_t + mis_t
    if tot_sig > 0:
        print(f"Alignment ratio: {al_t/tot_sig*100:.1f}% aligned, {mis_t/tot_sig*100:.1f}% misaligned")

    # Counterfactual
    avoided = sum(abs(float(r["realized_pnl"])) for r in rows if abs(float(r["pred_mu_alpha"]))>=0.01 and not((r["side"]=="LONG" and float(r["pred_mu_alpha"])>0) or (r["side"]=="SHORT" and float(r["pred_mu_alpha"])<0)) and float(r["realized_pnl"])<0)
    missed = sum(float(r["realized_pnl"]) for r in rows if abs(float(r["pred_mu_alpha"]))>=0.01 and not((r["side"]=="LONG" and float(r["pred_mu_alpha"])>0) or (r["side"]=="SHORT" and float(r["pred_mu_alpha"])<0)) and float(r["realized_pnl"])>0)
    print(f"\n=== CF: Skip misaligned ===")
    print(f"Avoided loss: ${avoided:.2f}, Missed wins: ${missed:.2f}")
    print(f"Net saved: ${avoided-missed:.2f}, New PnL: ${total_pnl_val+avoided-missed:.2f}")

    flip_gain = sum(-2*float(r["realized_pnl"]) for r in rows if abs(float(r["pred_mu_alpha"]))>=0.01 and not((r["side"]=="LONG" and float(r["pred_mu_alpha"])>0) or (r["side"]=="SHORT" and float(r["pred_mu_alpha"])<0)))
    print(f"\n=== CF: Flip misaligned ===")
    print(f"PnL delta: ${flip_gain:.2f}, New PnL: ${total_pnl_val+flip_gain:.2f}")

    print(f"\n=== Regime Breakdown ===")
    for reg, s in sorted(regime_st.items(), key=lambda x: x[1]["at"]+x[1]["mt"], reverse=True):
        at,aw,ap,mt,mw,mp = s["at"],s["aw"],s["ap"],s["mt"],s["mw"],s["mp"]
        if at+mt == 0: continue
        print(f"  {reg:15s} aligned: WR={aw/max(1,at)*100:.0f}%({at}) ${ap:+.2f} | misaligned: WR={mw/max(1,mt)*100:.0f}%({mt}) ${mp:+.2f} | ratio={mt/max(1,at+mt)*100:.0f}%")

    # Confidence
    cr = c.execute("SELECT entry_confidence,realized_pnl FROM trades WHERE realized_pnl IS NOT NULL AND realized_pnl!=0 AND entry_confidence IS NOT NULL").fetchall()
    if cr:
        confs = np.array([float(r["entry_confidence"]) for r in cr])
        warr = np.array([1.0 if float(r["realized_pnl"])>0 else 0.0 for r in cr])
        if len(confs)>5 and np.std(confs)>1e-9:
            corr = float(np.corrcoef(confs,warr)[0,1])
            print(f"\n=== Confidence-WR Correlation: {corr:.4f} ===")
            for lo,hi in [(0,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,1.01)]:
                m=(confs>=lo)&(confs<hi); n=int(m.sum())
                if n>0: print(f"  [{lo:.1f},{hi:.1f}): n={n} WR={float(warr[m].mean())*100:.1f}%")

    # DirectionModel simulation
    print(f"\n=== DirectionModel Simulation (last 500) ===")
    try:
        from engines.mc.direction_model import DirectionModel
        model = DirectionModel()
        rec = c.execute("SELECT side,pred_mu_alpha,realized_pnl,regime FROM trades WHERE realized_pnl IS NOT NULL AND realized_pnl!=0 AND pred_mu_alpha IS NOT NULL ORDER BY id DESC LIMIT 500").fetchall()
        ow,nw,st2 = 0,0,0
        op,np2 = 0.0,0.0
        dc = 0
        for r in rec:
            side,mu,pnl,reg = r["side"],float(r["pred_mu_alpha"]),float(r["realized_pnl"]),str(r["regime"] or "chop")
            od = 1 if side=="LONG" else -1
            res = model.determine_direction(mu_alpha=mu,meta={},ctx={"regime":reg,"hurst":0.5,"vpin":0.5},ev_long=0.001,ev_short=0.001,score_long=0.001,score_short=0.001)
            nd = res.direction; st2+=1; ow += int(pnl>0); op += pnl
            if nd==od: nw+=int(pnl>0); np2+=pnl
            else: dc+=1; nw+=int((-pnl)>0); np2+=(-pnl)
        print(f"Dir changes: {dc}/{st2} ({dc/max(1,st2)*100:.1f}%)")
        print(f"Old: WR={ow/max(1,st2)*100:.1f}% PnL=${op:.2f}")
        print(f"New: WR={nw/max(1,st2)*100:.1f}% PnL=${np2:.2f}")
        print(f"Delta: WR +{(nw-ow)/max(1,st2)*100:.1f}pp, PnL ${np2-op:+.2f}")
    except Exception as e:
        import traceback; print(f"Failed: {e}"); traceback.print_exc()

    # Time windows
    print(f"\n=== Time Windows ===")
    now = int(time.time()*1000)
    for lb,wms in [("24h",86400000),("4h",14400000),("1h",3600000)]:
        tw = c.execute("SELECT side,pred_mu_alpha,realized_pnl FROM trades WHERE realized_pnl IS NOT NULL AND realized_pnl!=0 AND timestamp_ms>=?",(now-wms,)).fetchall()
        if not tw: print(f"  {lb}: no trades"); continue
        tt=len(tw); tw_=sum(1 for r in tw if float(r["realized_pnl"])>0); tp=sum(float(r["realized_pnl"]) for r in tw)
        al=sum(1 for r in tw if r["pred_mu_alpha"] and abs(float(r["pred_mu_alpha"]))>=0.01 and ((r["side"]=="LONG" and float(r["pred_mu_alpha"])>0) or (r["side"]=="SHORT" and float(r["pred_mu_alpha"])<0)))
        ml=sum(1 for r in tw if r["pred_mu_alpha"] and abs(float(r["pred_mu_alpha"]))>=0.01 and not((r["side"]=="LONG" and float(r["pred_mu_alpha"])>0) or (r["side"]=="SHORT" and float(r["pred_mu_alpha"])<0)))
        print(f"  {lb}: n={tt} WR={tw_/max(1,tt)*100:.1f}% PnL=${tp:.2f} aligned={al} misaligned={ml}")

    conn.close()
    print("\nDONE")

if __name__=="__main__": main()
