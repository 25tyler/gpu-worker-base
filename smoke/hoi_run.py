"""HOI-DETR on a folder of frames (an NVIDIA GPU when there is one, else the processor), released defaults (score 0.3, soft-NMS 0.5, the model's own
interaction decision). Same steps as the repo's demo/demo.py, without drawing; resumes from its own output.
Copied from the transfer-accuracy branch (work/hoidetr/hoi_run.py); HOI_HOME is the folder setup.sh built there
(HOI-DETR checkout, mmcv build, hoidetr_weights.pth), run with its Python (HOI_PY in lean.sh).
    hoi_run.py <frames dir> <out.json> [step=1]"""
import glob, json, os, sys, time
H = os.environ.get("HOI_HOME", os.path.expanduser("~/labprint-runs/transfer-acc/work/hoidetr"))
REPO = f"{H}/HOI-DETR"
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") not in (H, HERE)]   # H holds the mmcv checkout; lean/ has its own module names
sys.path[:0] = [REPO, f"{REPO}/demo"]
import mmcv as _m
_v = _m.__version__; _m.__version__ = "1.5.0"
import mmdet  # noqa: version gate (mmdet 2.25.3 asks for mmcv < 1.6; the HOI-DETR demo Space does the same)
_m.__version__ = _v
import torch
from mmdet.apis import init_detector
from mmdet.datasets.pipelines import Compose
from projects import *  # noqa
from configs import CLASS_NAMES
from helpers import find_interaction_branch, run_inference, call_interaction
from predictions_io import detections_record
import helpers
from mmcv.parallel import DataContainer


DEV = os.environ.get("HOI_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")   # the cloud worker's GPU; the Mac's processor


def _scatter(data, devices):
    """Stand-in for mmcv's scatter (it refuses 'cpu', and on some PyTorch versions a 'cuda:0' string): unwrap the
    DataContainers, as mmdet's inference_detector does on CPU, and move the tensors to DEV."""
    def un(x):
        if isinstance(x, DataContainer):
            return un(x.data[0])
        if isinstance(x, torch.Tensor):
            return x.to(DEV)
        if isinstance(x, dict):
            return {k: un(v) for k, v in x.items()}
        if isinstance(x, list):
            return [un(v) for v in x]
        return x
    return [un(data)]


helpers.scatter = _scatter

frames_dir, out = sys.argv[1:3]
step = int(sys.argv[3]) if len(sys.argv) > 3 else 1
torch.set_num_threads(int(os.environ.get("HOI_THREADS", "8")))
cfg = "projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco_with_relation_only_all_losses_custom.py"
model = init_detector(cfg, f"{H}/hoidetr_weights.pth", device=DEV).eval()
pipe = Compose(model.cfg.data.test.pipeline)
branch = find_interaction_branch(model.query_head)
done = json.load(open(out)) if os.path.exists(out) else {"images": []}
have = {r["file_name"] for r in done["images"]}
imgs = sorted(glob.glob(f"{frames_dir}/*.jpg"))[::step]
for i, p in enumerate(imgs):
    name = os.path.basename(p)
    if name in have:
        continue
    t0 = time.time()
    dets, emb = run_inference(model, pipe, p, device=DEV, class_names=CLASS_NAMES, score_thr=0.3, nms_iou=0.5)
    hands = [d for d in dets if d["class_id"] == 0]; firsts = [d for d in dets if d["class_id"] == 1]
    seconds = [d for d in dets if d["class_id"] == 2]
    hf = [(h, f, pr) for h in hands for f in firsts for ok, pr in [call_interaction(branch, emb[h["query_idx"]], emb[f["query_idx"]])] if ok]
    fs = [(f, s, pr) for f in firsts for s in seconds for ok, pr in [call_interaction(branch, emb[f["query_idx"]], emb[s["query_idx"]])] if ok]
    det_j, hf_j, fs_j = detections_record(dets, hf, fs)
    done["images"].append({"file_name": name, "detections": det_j, "hf": hf_j, "fs": fs_j})
    json.dump(done, open(out, "w"))
    print(f"{i+1}/{len(imgs)} {name} {len(dets)} dets {len(fs)} tool->object links {time.time()-t0:.1f}s", flush=True)
print("DONE", flush=True)
