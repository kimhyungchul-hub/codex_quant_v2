# 1. 베이스 이미지 (리눅스 + 파이썬 3.9)
FROM python:3.9-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 시스템 의존성 설치 (필요한 경우에만)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. 파이썬 라이브러리 설치
# (requirements.txt가 있다면 COPY 후 설치)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# (없다면 직접 명시: RUN pip install torch numpy ...)

# 5. 소스 코드 복사 (나중에 -v 옵션 쓰면 생략 가능하지만 기본적으로 넣음)
COPY . .

# 6. 실행 명령 (컨테이너 실행 시 기본으로 할 동작)
CMD ["python", "main_engine_mc_v2_final.py"]