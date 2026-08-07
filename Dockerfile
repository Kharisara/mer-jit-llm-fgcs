FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_cloud.txt .
RUN pip install --no-cache-dir -r requirements_cloud.txt

COPY run_fgcs_extended_benchmark.py .
COPY run_fgcs_cloud_job.py .
COPY replaybench ./replaybench
COPY configs ./configs

COPY paper_outputs/replay_input_v260.csv ./paper_outputs/replay_input_v260.csv
COPY paper_outputs/replay_input_v260_manifest.json ./paper_outputs/replay_input_v260_manifest.json
CMD ["python", "run_fgcs_cloud_job.py"]