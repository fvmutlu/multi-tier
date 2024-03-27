FROM python:3.10.14-bookworm

WORKDIR /multi-tier

COPY . /multi-tier

RUN pip install --no-cache-dir -r requirements.txt