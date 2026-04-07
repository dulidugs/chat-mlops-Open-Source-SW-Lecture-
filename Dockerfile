# === Stage 1: Builder ===
FROM python:3.11-slim AS builder

WORKDIR /build

# 가상 환경 생성 (의존성을 격리하여 최적화)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# === Stage 2: Runtime ===
FROM python:3.11-slim

# 보안을 강화하기 위한 non-root 사용자 생성
RUN useradd -m appuser

WORKDIR /app

# Builder 스테이지에서 생성된 의존성 복사
COPY --from=builder /opt/venv /opt/venv

# 환경변수에 가상환경 추가
ENV PATH="/opt/venv/bin:$PATH"

# 애플리케이션 코드 복사
COPY ./app ./app

# 디렉터리 소유권 변경
RUN chown -R appuser:appuser /app

# 사용자 변경
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
