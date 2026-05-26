FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libasound2-dev \
    procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Add CUDA libraries to LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:\
/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:\
/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:\
/usr/local/lib/python3.11/site-packages/nvidia/cufft/lib:\
/usr/local/lib/python3.11/site-packages/nvidia/curand/lib:\
/usr/local/lib/python3.11/site-packages/nvidia/cusolver/lib:\
/usr/local/lib/python3.11/site-packages/nvidia/cusparse/lib:\
/usr/local/lib/python3.11/site-packages/tensorrt_libs:${LD_LIBRARY_PATH}

COPY . .

EXPOSE 50051
ENV PYTHONPATH=/app/app
CMD ["python", "app/node.py"]