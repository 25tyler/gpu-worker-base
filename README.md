# gpu-worker-base

CUDA 13 base image with PyTorch 2.14.1, Meta's SAM 3.1 code, EdgeTAM, RF-DETR and video/OCR packages, for a
serverless GPU worker. Open-source parts only: no application code and no model weights are in this image.

The packages are frozen in `requirements.lock` (made from `requirements.in` with
`uv pip compile requirements.in -o requirements.lock --python-version 3.12 --python-platform x86_64-manylinux_2_28 --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match`)
and installed in several layers, pushed with zstd compression and a registry build cache. Pushes to `main` publish
`:latest`; other branches publish only `:<branch>` and `:<commit>`.
