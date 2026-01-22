# codex_quant

Monte Carlo 기반 고빈도 거래 시스템

## Quick Start

```bash
# 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate  # JAX 메모리 선점 방지 자동 적용

# 의존성 설치
pip install -r requirements.txt

# 엔진 실행
python main_engine_mc_v2_final.py
```

## JAX 메모리 최적화

이 프로젝트는 JAX의 GPU 메모리 선점 문제를 방지하기 위해 다음을 자동으로 설정합니다:

- `.venv/bin/activate`: 가상환경 활성화 시 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 자동 설정
- `bootstrap.py`: 모든 엔트리포인트에서 임포트하여 환경변수 설정 보장

**메모리 사용량**: ~1.6GB (24GB 시스템 기준 6.7%)

## Documentation

- `docs/CODE_MAP_v2.md`: 전체 코드 구조 및 아키텍처
- `docs/MATHEMATICS.md`: 수학 공식 레퍼런스
- `.github/copilot-instructions.md`: 개발 규칙 및 변경 로그
