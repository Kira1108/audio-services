FROM python:3.12.4-slim

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

RUN pip install --no-cache-dir -U pip && pip install setuptools wheel

COPY requirements.txt ./

RUN pip install -r requirements.txt --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple some-package

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000","--reload", "--log-config=log_conf.yaml"]
