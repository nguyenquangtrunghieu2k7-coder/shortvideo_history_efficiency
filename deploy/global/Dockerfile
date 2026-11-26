FROM python:3.11-slim-bookworm
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 SHORTVIDEO_DEVICE=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 SHORTVIDEO_PROCESSED_DIR=/runtime/data SHORTVIDEO_MODEL_DIR=/tmp/models SHORTVIDEO_RESULT_DIR=/tmp/phase8
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY deploy/m5/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r /tmp/requirements.txt
COPY src /app/src
COPY deploy/global /app/deploy/global
EXPOSE 8778
ENTRYPOINT ["python","-m","src.integration.global_serving"]
