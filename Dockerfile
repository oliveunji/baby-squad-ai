FROM python:3.11-slim

WORKDIR /app

# 필수 시스템 패키지 설치 (빌드 도구 등)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스코드 & DB 데이터 복사 (COPY . . 이 핵심!)
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]