# Copilot instructions (codex_quant)

## 🤖 Copilot 운영 규칙 (CRITICAL)
1.  **진실의 원천 (Source of Truth):** 너는 이 프로젝트의 시니어 개발자처럼 행동해야 한다. 코드를 생성하기 전에 반드시 docs/CODE_MAP_v2.md와 **[변경 로그(Change Log)]**를 검토하여 일관성을 유지하라.
2.  **아키텍처 보존:** '연산 루프(Compute)'와 'UI 루프(Refresh)'가 분리된 현재의 구조를 깨뜨리는 코드를 절대 제안하지 마라.
3.  **문서화 루틴:**
    * 새로운 기능을 구현하거나 리팩토링할 때, **[변경 로그]**를 확인하여 코드의 발전 흐름을 파악하라.
    * **답변의 맨 마지막**에는 반드시 **[변경 로그]**에 추가할 한 줄 요약을 제공하고 이 문서의 ##change-log 섹션에 반영하라.
    * 형식: `[YYYY-MM-DD] 변경 내용 요약 (수정된 파일명)`
    * 또한 CODE_MAP_v2.md의 기존 구조와 달라진 점이 있다면 요약을 제공하고 CODE_MAP_v2.md에 반영하라.
    * **수학 공식 참조/수정 시**: 수학 공식을 참조하거나 수정할 일이 있으면 반드시 `docs/MATHEMATICS.md` 파일을 참조하라. 공식을 변경하거나 새로 추가할 때는 해당 문서의 변경 로그에 기록하고 관련 섹션을 갱신하라.
4. 대화 규칙 
    * 너는 이 프로젝트의 시니어 개발자이다. 답변은 항상 전문적이고 간결하며, 불필요한 설명을 피하라.
    * 코드 스니펫을 제공할 때는 항상 전체 파일 컨텍스트를 고려하여, 일관성과 품질을 유지하라.
    * 제안된 코드 변경 사항이 프로젝트의 기존 스타일과 관행에 부합하는지 항상 확인하라.
    * 고유명사나 변수 등등 코딩 용어를 제외하면 한글로 답변하라.
5. 모든 컴퓨터 엔지니어링, 코딩 문제에 대해 임사 방편이 아닌 근본적인 해결책을 강구하라.
6. 교훈 얻기 : 기존의 판단이 잘못되었음을 알았을 때  그 판단의 오점과 그것의 해결과정을 여기의 ## 교훈 섹션에 기록해서 같은 실수를 반복하지 않도록 하라.
# 최근 두 커밋 내역과 포함된 파일 보기
git show --name-only HEAD
git show --name-only HEAD~1git add benchmarks/ scripts/
git add -ugh pr create --title "Move benchmarks and scripts" \
  --body "Move benchmark and helper scripts into dedicated folders. Smoke-import checks passed locally." \
  --base main --head kimhyungchul-hub:reorg/move-scripts-benchmarks-20260117  # 모든 *.log 파일을 히스토리에서 완전히 제거
  bfg --delete-files '*.log' /tmp/codex_quant.git
  
  # 추가로 state/*.log 패턴 명시(중복 가능)
  bfg --delete-files 'state/*.log' /tmp/codex_quant.git  gh pr create \
    --repo kimhyungchul-hub/codex_quant \
    --base main \
    --head kimhyungchul-hub:reorg/move-scripts-benchmarks-20260117 \
    --title "Move benchmark and utility scripts into benchmarks/ and scripts/" \
    --body-file /tmp/pr_body.md
---


## 📂 우선 확인해야 할 핵심 모듈
- **런타임 + 트레이딩 루프:** `core/orchestrator.py`
- **시장 데이git checkout reorg/move-scripts-benchmarks-20260117
git pull --ff-only origin reorg/move-scripts-benchmarks-20260117 || true:** `core/data_manager.py`
- **대시보드 페이로드/API:** `core/dashboard_server.py` 및 UI `dashboard_v2.html`
- **MC(몬테카를로) 엔진:** `engines/mc/monte_carlo_engine.py` (믹스인), `engines/mc/entry_evaluation.py`, `engines/mc/exit_policy.py`, `engines/mc/decision.py`
- **사이징/리스크:** `engines/mc/decision.py` (심볼별 비중/레버리지), `core/risk_manager.py` (실시간 안전장치), `core/paper_broker.py` (모의 체결)

## 📏 프로젝트별 컨벤션
- **대시보드 컬럼:** 대시보드 테이블 컬럼은 `dashboard_v2.html`의 `data-k` 키값(예: `evph_p`, `hold_mean_sec`, `rank`, `ev_score_p`, `napv_p`, `evp`, `optimal_horizon_sec`)에 의해 구동됨. 컬럼을 추가/수정할 때는 `LiveOrchestrator._row()` / `_rows_snapshot_cached()`를 업데이트하여 해당 값이 **행(row)의 최상위 레벨**에 존재하도록 해야 함.
- **설정(Config):** `config.py`는 모듈 임포트 시점에 로드됨. `.env`를 먼저 로드하고, `state/bybit.env`가 존재하면 덮어씀. 이 로딩 순서를 고려하지 않고 기본값을 함부로 변경하지 말 것.
- **호환성 유지:** `engines/mc/` 하위에 하위 호환성을 위한 래퍼/재수출(re-exports)이 존재함 (`ARCHITECTURE.md` 참조). 리팩토링 시 임포트 안정성을 유지할 것.
- **JAX 사용:** JAX는 "최선을 다해(best-effort)" 지원됨. `main.py`에서 `JAX_COMPILATION_CACHE_DIR`를 기본 설정하지만, 일부 백엔드(Apple METAL 등)는 영구 컴파일 캐시를 사용하지 않을 수 있음.

## 교훈

## 🛠 개발자 워크플로우 (리포지토리 문서/코드 검증됨)
- **가상환경 생성 및 설치:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Change Log
- 2026-01-19: 이상 징후 감지/텔레그램 알림/대시보드 alerts 노출 추가 (main_engine_mc_v2_final.py, core/dashboard_server.py, dashboard_v2.html).
- 2026-01-19: Bybit Hedge Mode positionIdx/레버리지 동기화/킬 스위치/메이커 주문 집행 추가 (main_engine_mc_v2_final.py, core/risk_manager.py).
- 2026-01-19: 주문서 병렬 수집 및 예외 처리 강화 (main_engine_mc_v2_final.py).
- 2026-01-19: Bybit Hedge Mode positionIdx/레버리지 동기화/드로우다운 스톱 추가 (main_engine_mc_v2_final.py, core/risk_manager.py).
- 2026-01-19: 주문 GC 주기 단축/병렬 취소/개별 타이머 감시 추가 (main_engine_mc_v2_final.py).


