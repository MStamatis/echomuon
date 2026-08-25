# RTX 5090 (Blackwell, sm_120) needs CUDA >= 12.8 -> PyTorch >= 2.7 cu128 image
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /lab
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "run_experiments.py"]
CMD ["smoke"]
