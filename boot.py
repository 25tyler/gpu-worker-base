"""Worker start-up, baked into the public base image (cloud/Dockerfile). The image holds only open-source parts;
the Labprint code and the model weights are fetched from R2 here, once per worker, then the handler starts.

R2 keys (under BUNDLE_PREFIX, default cloud-test/worker-bundle/): code.tar.gz (git archive of the branch),
models.tar (sam3.1/, EdgeTAM-hf/, edgetam.pt, rfdetr .pth, face model). A worker that still has them on disk
(FlashBoot) skips the download.
"""
import os
import subprocess
import sys
import tarfile
import time

import boto3
from boto3.s3.transfer import TransferConfig

PREFIX = os.environ.get("BUNDLE_PREFIX", "cloud-test/worker-bundle/")
CODE, MT = "/opt/labprint", os.environ.get("MAC_TRACK_HOME", "/opt/mt")


def fetch(s3, key, dst):
    s3.download_file(os.environ["R2_BUCKET"], PREFIX + key, dst,
                     Config=TransferConfig(max_concurrency=32, multipart_chunksize=64 << 20))


def main():
    t0 = time.time()
    s3 = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT"], aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                      aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"], region_name="auto")
    if not os.path.exists(f"{MT}/models/.ready"):
        fetch(s3, "models.tar", "/tmp/models.tar")
        with tarfile.open("/tmp/models.tar") as t:
            t.extractall(f"{MT}/models", filter="data")
        os.remove("/tmp/models.tar")
        os.makedirs(f"{MT}/EdgeTAM/checkpoints", exist_ok=True)
        os.replace(f"{MT}/models/edgetam.pt", f"{MT}/EdgeTAM/checkpoints/edgetam.pt")
        open(f"{MT}/models/.ready", "w").close()
    t1 = time.time()
    if not os.path.exists(f"{CODE}/.ready"):
        fetch(s3, "code.tar.gz", "/tmp/code.tar.gz")
        os.makedirs(CODE, exist_ok=True)
        with tarfile.open("/tmp/code.tar.gz") as t:
            t.extractall(CODE, filter="data")
        os.remove("/tmp/code.tar.gz")
        os.makedirs(f"{CODE}/models", exist_ok=True)
        for f in os.listdir(f"{MT}/models"):
            if f.endswith(".onnx"):
                os.symlink(f"{MT}/models/{f}", f"{CODE}/models/{f}")
        open(f"{CODE}/.ready", "w").close()
    print(f"boot: models {t1 - t0:.1f}s, code {time.time() - t1:.1f}s", flush=True)
    os.environ.setdefault("LABPRINT_COMMIT", open(f"{CODE}/COMMIT").read().strip() if os.path.exists(f"{CODE}/COMMIT") else "unknown")
    os.execv(sys.executable, [sys.executable, "-u", f"{CODE}/cloud/handler.py"] + sys.argv[1:])


if __name__ == "__main__":
    main()
