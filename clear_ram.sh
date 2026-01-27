#!/bin/bash

# 1. 가상환경 활성화 (경로가 다르면 수정하세요)
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✅ 가상환경 활성화 완료"
else
    echo "❌ .venv 폴더를 찾을 수 없습니다."
    exit 1
fi

# 2. 메모리 최적화 (사용하지 않는 캐시 강제 정리)
echo "🧹 메모리 최적화를 시작합니다 (비밀번호 입력이 필요할 수 있습니다)..."
sudo purge
echo "✅ 시스템 캐시 정리 완료"

# 3. 메모리 많이 먹는 앱 경고 (수동 종료 권장)
echo "⚠️  팁: 크롬(Chrome)이나 다른 무거운 앱을 끄면 30B 모델이 더 안정적으로 돌아갑니다."

# 4. LiteLLM 설정 파일 업데이트 여부 확인
echo "📝 litellm_config.yaml의 모델명을 'local-30b'로 매핑하여 실행합니다."

# 5. MLX 서버 실행
# --max-tokens 1024: 램 폭주 방지를 위한 제한
# --host 127.0.0.1: 로컬 접속 전용
python -m mlx_lm.server --model DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx --max-tokens 1024 --temp 0.7 --host 127.0.0.1:7860 &