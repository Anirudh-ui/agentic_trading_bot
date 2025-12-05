FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    poppler-utils \
    libpoppler-cpp-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

# 1. Copy ONLY requirements first to leverage Docker cache
COPY requirements.txt .

# 2. Install deps once (layer is cached until requirements.txt changes)
RUN uv pip install --system -r requirements.txt

# 3. Now copy the rest of your project
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
