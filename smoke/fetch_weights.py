"""Download only the model weights out of HOI-DETR's 5.9 GB training checkpoint (it also holds optimizer state).
Reads the zip over HTTP range requests, unpickles data.pkl to see which stored tensors belong to state_dict, fetches
those, and saves {'state_dict', 'meta'} locally.   fetch_weights.py <out.pth>"""
import io, os, pickle, shutil, sys, zipfile, urllib.request
import torch

URL = "https://huggingface.co/ahmaddarkhalil/hoi-detr/resolve/main/epoch_5.pth"
fetched = [0]


class HTTPFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD")
        r = urllib.request.urlopen(req)
        self.size = int(r.headers["Content-Length"]); self.real = r.geturl()
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0: return b""
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.real, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
                data = urllib.request.urlopen(req, timeout=120).read(); break
            except Exception as e:
                print("retry", e, flush=True)
        self.pos += len(data); fetched[0] += len(data)
        return data
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)


f = io.BufferedReader(HTTPFile(URL), buffer_size=1 << 20)
zf = zipfile.ZipFile(f)
names = zf.namelist()
pkl_name = next(n for n in names if n.endswith("data.pkl"))
prefix = pkl_name[: -len("data.pkl")]
raw = zf.read(pkl_name)
print("entries", len(names), "data.pkl", len(raw), flush=True)


class Marker:
    def __init__(self, key): self.key = key


class Pass1(pickle.Unpickler):
    def find_class(self, mod, name):
        if name in ("_rebuild_tensor_v2", "_rebuild_tensor", "_rebuild_parameter"):
            return lambda *a, **k: a[0] if isinstance(a[0], Marker) else Marker(None)
        return super().find_class(mod, name)
    def persistent_load(self, pid):
        return Marker(pid[2])


obj = Pass1(io.BytesIO(raw)).load()
print("top keys", list(obj) if isinstance(obj, dict) else type(obj), flush=True)
need = {m.key for m in obj["state_dict"].values() if isinstance(m, Marker)}
print("tensors in state_dict", len(obj["state_dict"]), "storages", len(need), flush=True)

cache = {}
CACHE = sys.argv[1] + ".parts"; os.makedirs(CACHE, exist_ok=True)   # resumable: one file per tensor


class Pass2(pickle.Unpickler):
    def find_class(self, mod, name):
        if name == "_rebuild_tensor_v2":
            real = torch._utils._rebuild_tensor_v2
            return lambda storage, *a, **k: None if storage is None else real(storage, *a, **k)
        if name == "_rebuild_parameter":
            real = torch._utils._rebuild_parameter
            return lambda data, *a, **k: None if data is None else real(data, *a, **k)
        return super().find_class(mod, name)
    def persistent_load(self, pid):
        _, stype, key, loc, numel = pid
        if key not in need: return None
        if key not in cache:
            cp = os.path.join(CACHE, key)
            if os.path.exists(cp):
                b = open(cp, "rb").read()
            else:
                b = zf.read(f"{prefix}data/{key}")
                open(cp + ".part", "wb").write(b); os.replace(cp + ".part", cp)
            us = torch.UntypedStorage.from_buffer(b, dtype=torch.uint8)
            cache[key] = torch.storage.TypedStorage(wrap_storage=us, dtype=stype.dtype, _internal=True)
            if len(cache) % 50 == 0: print("fetched", len(cache), "of", len(need), f"{fetched[0]/1e9:.2f} GB", flush=True)
        return cache[key]


full = Pass2(io.BytesIO(raw)).load()
torch.save({"state_dict": full["state_dict"], "meta": full.get("meta", {})}, sys.argv[1])
shutil.rmtree(CACHE)   # this script's own partial download, not a shared cache
print(f"saved {sys.argv[1]}; downloaded {fetched[0]/1e9:.2f} GB", flush=True)
