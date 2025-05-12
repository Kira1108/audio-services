FROM python:3.12.4-slim

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Install libgomp1 and ffmpeg for audio processing
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libgomp1 \
#     ffmpeg \
#     && rm -rf /var/lib/apt/lists/*

RUN echo 'deb http://mirrors.tuna.tsinghua.edu.cn/debian bookworm main non-free contrib\n\
deb-src http://mirrors.tuna.tsinghua.edu.cn/debian bookworm main non-free contrib\n\
deb http://mirrors.tuna.tsinghua.edu.cn/debian-security bookworm-security main\n\
deb http://mirrors.tuna.tsinghua.edu.cn/debian bookworm-updates main\n\
deb http://mirrors.tuna.tsinghua.edu.cn/debian bookworm-backports main' > /etc/apt/sources.list \
&& apt-get update && apt-get install -y --no-install-recommends libgomp1 ffmpeg \
&& rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple pip \
    && pip install setuptools wheel --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

COPY requirements.txt ./

RUN pip install -r requirements.txt --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-config=log_conf.yaml"]