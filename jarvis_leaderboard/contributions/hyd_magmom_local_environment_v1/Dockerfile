FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /work
COPY requirements.txt /work/requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . /work
ENV OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLCONFIGDIR=/tmp/matplotlib
ENTRYPOINT ["python", "replay_benchmark.py", "--selection", "selection.json", "--ids", "test_ids.json", "--output", "AI-SinglePropertyClass-magmom_oszicar-dft_3d-test-acc.csv.zip"]
