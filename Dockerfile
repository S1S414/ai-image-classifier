# 目标检测系统 Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖（含编译工具）
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖（使用官方 PyPI，适配云端部署）
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 安装 GroundingDINO 本地源码（--no-deps 避免覆盖已装好的版本）
RUN cd GroundingDINO && pip install --no-cache-dir --no-deps .

# 创建必要目录
RUN mkdir -p uploads weights models

# 暴露端口
EXPOSE 5000

# 环境变量
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=5000
ENV FLASK_DEBUG=False

# 启动命令
CMD ["python", "app.py"]
