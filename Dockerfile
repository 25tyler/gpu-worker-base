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
# Same packages as before, frozen in requirements.lock (uv pip compile of requirements.in for linux x86_64), and split
# into several layers instead of one 3.7 GB layer: a new machine downloads and unpacks them side by side, and layers
# that did not change keep their digest between builds (registry build cache), so a machine that has any earlier
# version only pulls what changed. Largest and most stable first.
COPY requirements.lock /opt/requirements.lock
ENV UVI="--no-cache --no-deps -p /opt/venv/bin/python --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match"
RUN uv venv -q -p 3.12 /opt/venv && uv pip install $UVI $(grep -E '^nvidia-cudnn' /opt/requirements.lock)
RUN uv pip install $UVI $(grep -E '^nvidia-(cublas|nccl)' /opt/requirements.lock)
RUN uv pip install $UVI $(grep -E '^nvidia-' /opt/requirements.lock | grep -vE '^nvidia-(cudnn|cublas|nccl)')
RUN uv pip install $UVI $(grep -E '^(torch|torchvision|triton)==' /opt/requirements.lock)
RUN uv pip sync --no-cache -p /opt/venv/bin/python --index-url https://pypi.org/simple \
      --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match /opt/requirements.lock
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
# HOI-DETR (github.com/AhmadDarKhalil/HOI-DETR, lean.sh TRANSFERS=1) in its own Python 3.10 venv: it needs mmcv-full
# 1.x, which has no build for the main venv's PyTorch. OpenMMLab's prebuilt mmcv-full 1.7.2 wheel for PyTorch 2.0 / CUDA
# 11.8, so nothing is compiled here. Its weights come from R2 like the others (models/hoidetr/, boot.py links them).
RUN uv venv -q -p 3.10 /opt/hoi && uv pip install --no-cache -p /opt/hoi/bin/python --index-url https://pypi.org/simple \
      --extra-index-url https://download.pytorch.org/whl/cu118 --index-strategy unsafe-best-match \
      torch==2.0.1+cu118 torchvision==0.15.2+cu118 "numpy<2" "setuptools<70" wheel
RUN uv pip install --no-cache -p /opt/hoi/bin/python --index-url https://pypi.org/simple \
      https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/mmcv_full-1.7.2-cp310-cp310-manylinux1_x86_64.whl \
      timm==0.6.13 fairscale==0.4.6 scipy yapf==0.40.1 opencv-python-headless==4.11.0.86 pycocotools addict terminaltables \
      pillow packaging fvcore einops matplotlib "numpy<2" && \
    mkdir /opt/hoi/HOI-DETR && cd /opt/hoi/HOI-DETR && git init -q && \
    git remote add origin https://github.com/AhmadDarKhalil/HOI-DETR.git && \
    git fetch -q --depth 1 origin 1b367292f3833afd64a204bd4d9d84519541d035 && git checkout -q FETCH_HEAD && rm -rf .git && \
    uv pip install --no-cache -q -p /opt/hoi/bin/python --no-build-isolation --no-deps -e . && \
    /opt/hoi/bin/python -c "import torch, mmcv, mmcv.ops; print('hoi ok', torch.__version__, mmcv.__version__)"
COPY boot.py /opt/boot.py
ENV PATH=/opt/venv/bin:$PATH LABPRINT_PY=/opt/venv/bin/python LABPRINT_REPO=/opt/labprint HOI_PY=/opt/hoi/bin/python HOI_HOME=/opt/hoi \
    SAM31_CKPT=/opt/mt/models/sam3.1/sam3.1_multiplex.pt RFDETR_PTH=/opt/mt/models/rfdetr-a602804c.pth \
    LABPRINT_DEVICE=cuda MAC_DETECTOR=torch
CMD ["python", "-u", "/opt/boot.py"]
