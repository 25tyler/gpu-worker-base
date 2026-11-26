# Base image of the cloud worker: only open-source parts (no Labprint code, no model weights; boot.py fetches those
# from R2 when a worker starts). Same package versions as demo_live/mac/setup_cuda.sh (the CUDA port that matched the
# Mac) and, for the rest of the chain, the Mac's own venv. Built on GitHub Actions, never on the Mac.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git curl ca-certificates rsync libgl1 libglib2.0-0 \
      libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
ENV MAC_TRACK_HOME=/opt/mt
WORKDIR /opt/mt
RUN uv venv -q -p 3.12 /opt/venv && \
    uv pip install --no-cache -p /opt/venv/bin/python torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130 && \
    uv pip install --no-cache -p /opt/venv/bin/python transformers==5.20.2 rfdetr==1.11.2 supervision==0.31.5 timm==1.0.30 \
      accelerate==1.15.0 peft==0.21.2 hydra-core==1.3.8 ensemble-boxes opencv-python-headless==5.0.0.93 scipy==1.18.1 pycocotools \
      pillow==13.0.0 huggingface_hub ftfy==6.1.1 regex iopath einops "setuptools<81" ultralytics==8.4.190 numpy==2.5.4 \
      scikit-image==0.26.0 mediapipe==1.1.1 onnxruntime==1.31.1 rapidocr==3.10.0 boto3==1.43.119 Jinja2==3.1.6 openai \
      anthropic==2.7.0 python-dotenv==1.2.4 fastapi==0.162.0 pydantic==2.14.1 PyYAML==6.0.3 "PyJWT[crypto]==2.10.1" runpod \
      "torch==2.14.1" "torchvision==0.29.1" \
      --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
# Meta's SAM 3.1 code and EdgeTAM at the commits setup_cuda.sh uses (EdgeTAM with its perceiver fix); roboflow trackers
RUN mkdir sam3 && cd sam3 && git init -q && git remote add origin https://github.com/facebookresearch/sam3.git && \
    git fetch -q --depth 1 origin 31cb7e624e4fe95301b1be1d4e809111593d2e45 && git checkout -q FETCH_HEAD && rm -rf .git && cd .. && \
    /opt/venv/bin/python -c "import site; open(site.getsitepackages()[0] + '/sam3-repo.pth', 'w').write('/opt/mt/sam3\n')" && \
    mkdir EdgeTAM && cd EdgeTAM && git init -q && git remote add origin https://github.com/facebookresearch/EdgeTAM.git && \
    git fetch -q --depth 1 origin 7711e012a30a2402c4eaab637bdb00a521302c91 && git checkout -q FETCH_HEAD && rm -rf .git && \
    sed -i 's/expand(B, -1, -1).view(-1, 1, C)/expand(B, -1, -1).reshape(-1, 1, C)/' sam2/modeling/perceiver.py && \
    grep -q 'expand(B, -1, -1).reshape(-1, 1, C)' sam2/modeling/perceiver.py && cd .. && \
    uv pip install --no-cache -q --no-deps --target pk_trackers -p /opt/venv/bin/python trackers && \
    /opt/venv/bin/python -c "import torch, sam3, rfdetr, transformers, cv2, mediapipe, rapidocr, runpod; print('ok', torch.__version__)"
COPY boot.py /opt/boot.py
ENV PATH=/opt/venv/bin:$PATH LABPRINT_PY=/opt/venv/bin/python LABPRINT_REPO=/opt/labprint \
    SAM31_CKPT=/opt/mt/models/sam3.1/sam3.1_multiplex.pt RFDETR_PTH=/opt/mt/models/rfdetr-a602804c.pth \
    LABPRINT_DEVICE=cuda MAC_DETECTOR=torch
CMD ["python", "-u", "/opt/boot.py"]
