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
COPY configs ./configs

COPY paper_outputs/replay_input_clean.csv ./paper_outputs/replay_input_clean.csv
COPY paper_outputs/policy_first_outputs_bc.csv ./paper_outputs/policy_first_outputs_bc.csv

COPY checkpoints ./checkpoints
COPY data/processed/MELD_state_embeddings_fixed ./data/processed/MELD_state_embeddings_fixed

CMD ["python", "run_fgcs_cloud_job.py"]