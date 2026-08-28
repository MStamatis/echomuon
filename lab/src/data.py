"""Datasets -> train.bin / val.bin (uint16).

- shakespeare: 1M chars, char-level vocab ~65. Good for smoke tests only (overfits fast).
- enwik8: 100M bytes, byte-level vocab 256, 95/5 split. Main benchmark; a full `final`
  run (4000 steps x 16k tokens = 65M tokens) stays under one epoch, so final val loss
  measures optimization quality rather than overfitting speed.
"""
import json
import os
import urllib.request
import zipfile

import numpy as np

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"


def _prepare_shakespeare(d: str) -> None:
    txt = os.path.join(d, "input.txt")
    if not os.path.exists(txt):
        print(f"downloading tiny shakespeare -> {txt}")
        urllib.request.urlretrieve(SHAKESPEARE_URL, txt)
    with open(txt, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.uint16)
    split = int(len(ids) * 0.9)
    ids[:split].tofile(os.path.join(d, "train.bin"))
    ids[split:].tofile(os.path.join(d, "val.bin"))
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"vocab_size": len(chars)}, f)


def _prepare_shakespeare_bytes(d: str) -> None:
    """Shakespeare as raw UTF-8 bytes with vocab 256, so a model pretrained on enwik8
    (byte-level) can be fine-tuned on it without any vocabulary change."""
    txt = os.path.join(d, "input.txt")
    if not os.path.exists(txt):
        print(f"downloading tiny shakespeare -> {txt}")
        urllib.request.urlretrieve(SHAKESPEARE_URL, txt)
    with open(txt, "rb") as f:
        raw = f.read()
    ids = np.frombuffer(raw, dtype=np.uint8).astype(np.uint16)
    split = int(len(ids) * 0.9)
    ids[:split].tofile(os.path.join(d, "train.bin"))
    ids[split:].tofile(os.path.join(d, "val.bin"))
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"vocab_size": 256}, f)


def _prepare_enwik8(d: str) -> None:
    raw = os.path.join(d, "enwik8")
    if not os.path.exists(raw):
        zpath = os.path.join(d, "enwik8.zip")
        print(f"downloading enwik8 (~35MB) -> {zpath}")
        urllib.request.urlretrieve(ENWIK8_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extract("enwik8", d)
        os.remove(zpath)
    ids = np.fromfile(raw, dtype=np.uint8).astype(np.uint16)
    split = int(len(ids) * 0.95)
    ids[:split].tofile(os.path.join(d, "train.bin"))
    ids[split:].tofile(os.path.join(d, "val.bin"))
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"vocab_size": 256}, f)


def _prepare_enwik8_noisy(d: str, p: float = 0.10) -> None:
    """enwik8 with p of TRAIN bytes replaced by uniform random bytes (seeded).
    The val split stays clean: we measure generalization from noisy data."""
    raw = os.path.join(d, "enwik8")
    if not os.path.exists(raw):
        zpath = os.path.join(d, "enwik8.zip")
        print(f"downloading enwik8 (~35MB) -> {zpath}")
        urllib.request.urlretrieve(ENWIK8_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extract("enwik8", d)
        os.remove(zpath)
    ids = np.fromfile(raw, dtype=np.uint8).astype(np.uint16)
    split = int(len(ids) * 0.95)
    train = ids[:split].copy()
    rng = np.random.default_rng(1234)
    mask = rng.random(len(train)) < p
    train[mask] = rng.integers(0, 256, int(mask.sum()), dtype=np.uint16)
    print(f"enwik8 noisy: corrupted {mask.sum()} / {len(train)} train bytes ({p:.0%})")
    train.tofile(os.path.join(d, "train.bin"))
    ids[split:].tofile(os.path.join(d, "val.bin"))
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"vocab_size": 256, "corruption": p}, f)


FINEWEB_BASE = "https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main/"
FINEWEB_TRAIN_SHARDS = 4  # 4 x 100M GPT-2 BPE tokens; final runs use ~49M -> well under 1 epoch


def _read_llmc_bin(path: str) -> np.ndarray:
    """llm.c token shard: 256 x int32 header (magic 20240520, version, n_tokens), then uint16."""
    header = np.fromfile(path, dtype=np.int32, count=256)
    if header[0] != 20240520:
        raise RuntimeError(f"{path}: bad magic {header[0]} (expected 20240520)")
    toks = np.fromfile(path, dtype=np.uint16, offset=1024)
    if len(toks) != header[2]:
        raise RuntimeError(f"{path}: token count mismatch {len(toks)} vs header {header[2]}")
    return toks


def _prepare_fineweb(d: str) -> None:
    """FineWeb-Edu pre-tokenized with the GPT-2 BPE (kjj0/fineweb10B-gpt2, the
    modded-nanogpt benchmark data). Concatenates 4 train shards (400M tokens) into
    train.bin and the val shard (100M tokens) into val.bin; vocab padded to 50304."""
    def fetch(name):
        path = os.path.join(d, name)
        if not os.path.exists(path):
            print(f"downloading {name} (~196MB) -> {path}")
            req = urllib.request.Request(FINEWEB_BASE + name,
                                         headers={"User-Agent": "optimizer-lab/1.0"})
            tmp = path + ".tmp"
            with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
                while True:
                    buf = r.read(1 << 22)
                    if not buf:
                        break
                    f.write(buf)
            os.replace(tmp, path)
        return path

    with open(os.path.join(d, "train.bin.tmp"), "wb") as out:
        total = 0
        for i in range(1, FINEWEB_TRAIN_SHARDS + 1):
            shard = fetch(f"fineweb_train_{i:06d}.bin")
            toks = _read_llmc_bin(shard)
            toks.tofile(out)
            total += len(toks)
            print(f"fineweb: appended shard {i} ({len(toks)} tokens, total {total})")
    os.replace(os.path.join(d, "train.bin.tmp"), os.path.join(d, "train.bin"))
    val = _read_llmc_bin(fetch("fineweb_val_000000.bin"))
    val.tofile(os.path.join(d, "val.bin.tmp"))
    os.replace(os.path.join(d, "val.bin.tmp"), os.path.join(d, "val.bin"))
    for i in range(1, FINEWEB_TRAIN_SHARDS + 1):  # shards are redundant once converted
        os.remove(os.path.join(d, f"fineweb_train_{i:06d}.bin"))
    os.remove(os.path.join(d, "fineweb_val_000000.bin"))
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"vocab_size": 50304, "tokenizer": "gpt2-bpe",
                   "train_tokens": total, "val_tokens": len(val)}, f)


CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"


def _prepare_cifar10(d: str, label_noise: float = 0.0) -> None:
    """CIFAR-10 from the binary distribution, parsed with numpy (no torchvision).
    label_noise > 0 replaces that fraction of TRAIN labels with uniform random
    classes (seeded); test labels stay clean."""
    import shutil
    import tarfile
    bindir = os.path.join(d, "cifar-10-batches-bin")
    sibling = os.path.join(os.path.dirname(d), "cifar10", "cifar-10-batches-bin")
    if not os.path.exists(bindir) and os.path.exists(sibling) and sibling != bindir:
        print(f"reusing extracted CIFAR from {sibling}")
        shutil.copytree(sibling, bindir)
    if not os.path.exists(bindir):
        tpath = os.path.join(d, "cifar.tar.gz")
        print(f"downloading CIFAR-10 (~170MB) -> {tpath}")
        urllib.request.urlretrieve(CIFAR_URL, tpath)
        with tarfile.open(tpath) as t:
            t.extractall(d)
        os.remove(tpath)
    xs, ys = [], []
    for i in range(1, 6):
        raw = np.fromfile(os.path.join(bindir, f"data_batch_{i}.bin"), dtype=np.uint8)
        raw = raw.reshape(10000, 3073)
        ys.append(raw[:, 0].copy())
        xs.append(raw[:, 1:].copy())
    train_x, train_y = np.concatenate(xs), np.concatenate(ys)
    raw = np.fromfile(os.path.join(bindir, "test_batch.bin"), dtype=np.uint8).reshape(10000, 3073)
    test_x, test_y = raw[:, 1:].copy(), raw[:, 0].copy()
    if label_noise > 0:
        rng = np.random.default_rng(1234)
        mask = rng.random(len(train_y)) < label_noise
        train_y[mask] = rng.integers(0, 10, int(mask.sum()), dtype=np.uint8)
        print(f"cifar10 noisy: corrupted {mask.sum()} / {len(train_y)} train labels")
    np.save(os.path.join(d, "train_x.npy"), train_x)
    np.save(os.path.join(d, "train_y.npy"), train_y)
    np.save(os.path.join(d, "test_x.npy"), test_x)
    np.save(os.path.join(d, "test_y.npy"), test_y)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"num_classes": 10, "label_noise": label_noise}, f)


CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"


def _prepare_cifar100(d: str, label_noise: float = 0.0) -> None:
    """CIFAR-100 binary distribution: each record = 1 coarse + 1 fine label + 3072 px.
    We use the fine labels (100 classes)."""
    import shutil
    import tarfile
    bindir = os.path.join(d, "cifar-100-binary")
    sibling = os.path.join(os.path.dirname(d), "cifar100", "cifar-100-binary")
    if not os.path.exists(bindir) and os.path.exists(sibling) and sibling != bindir:
        print(f"reusing extracted CIFAR-100 from {sibling}")
        shutil.copytree(sibling, bindir)
    if not os.path.exists(bindir):
        tpath = os.path.join(d, "cifar100.tar.gz")
        print(f"downloading CIFAR-100 (~170MB) -> {tpath}")
        urllib.request.urlretrieve(CIFAR100_URL, tpath)
        with tarfile.open(tpath) as t:
            t.extractall(d)
        os.remove(tpath)
    raw = np.fromfile(os.path.join(bindir, "train.bin"), dtype=np.uint8).reshape(50000, 3074)
    train_x, train_y = raw[:, 2:].copy(), raw[:, 1].copy()
    raw = np.fromfile(os.path.join(bindir, "test.bin"), dtype=np.uint8).reshape(10000, 3074)
    test_x, test_y = raw[:, 2:].copy(), raw[:, 1].copy()
    if label_noise > 0:
        rng = np.random.default_rng(1234)
        mask = rng.random(len(train_y)) < label_noise
        train_y[mask] = rng.integers(0, 100, int(mask.sum()), dtype=np.uint8)
        print(f"cifar100 noisy: corrupted {mask.sum()} / {len(train_y)} train labels")
    np.save(os.path.join(d, "train_x.npy"), train_x)
    np.save(os.path.join(d, "train_y.npy"), train_y)
    np.save(os.path.join(d, "test_x.npy"), test_x)
    np.save(os.path.join(d, "test_y.npy"), test_y)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"num_classes": 100, "label_noise": label_noise}, f)


TINYIMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def _prepare_tinyimagenet(d: str, label_noise: float = 0.0) -> None:
    """Tiny-ImageNet-200: 100k train / 10k val images, 64x64, 200 classes.
    Decoded from JPEG into the same CHW-flattened uint8 npy layout as CIFAR."""
    import shutil
    import zipfile

    from PIL import Image
    root = os.path.join(d, "tiny-imagenet-200")
    sibling = os.path.join(os.path.dirname(d), "tinyimagenet", "tiny-imagenet-200")
    if not os.path.exists(root) and os.path.exists(sibling) and sibling != root:
        print(f"reusing extracted Tiny-ImageNet from {sibling}")
        shutil.copytree(sibling, root)
    if not os.path.exists(root):
        zpath = os.path.join(d, "tiny-imagenet-200.zip")
        print(f"downloading Tiny-ImageNet (~248MB) -> {zpath}")
        urllib.request.urlretrieve(TINYIMAGENET_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(d)
        os.remove(zpath)
    with open(os.path.join(root, "wnids.txt")) as f:
        wnids = sorted(f.read().split())
    cls = {w: i for i, w in enumerate(wnids)}

    def load_img(path):
        arr = np.asarray(Image.open(path).convert("RGB"))
        return arr.transpose(2, 0, 1).reshape(-1)

    train_x = np.zeros((100000, 3 * 64 * 64), dtype=np.uint8)
    train_y = np.zeros(100000, dtype=np.uint8)
    i = 0
    for w in wnids:
        imgdir = os.path.join(root, "train", w, "images")
        for fn in sorted(os.listdir(imgdir)):
            train_x[i] = load_img(os.path.join(imgdir, fn))
            train_y[i] = cls[w]
            i += 1
    print(f"tinyimagenet: decoded {i} train images")
    val_map = {}
    with open(os.path.join(root, "val", "val_annotations.txt")) as f:
        for line in f:
            parts = line.split("\t")
            val_map[parts[0]] = cls[parts[1]]
    names = sorted(val_map)
    test_x = np.zeros((len(names), 3 * 64 * 64), dtype=np.uint8)
    test_y = np.zeros(len(names), dtype=np.uint8)
    for j, fn in enumerate(names):
        test_x[j] = load_img(os.path.join(root, "val", "images", fn))
        test_y[j] = val_map[fn]
    if label_noise > 0:
        rng = np.random.default_rng(1234)
        mask = rng.random(len(train_y)) < label_noise
        train_y[mask] = rng.integers(0, 200, int(mask.sum()), dtype=np.uint8)
        print(f"tinyimagenet noisy: corrupted {mask.sum()} / {len(train_y)} train labels")
    np.save(os.path.join(d, "train_x.npy"), train_x)
    np.save(os.path.join(d, "train_y.npy"), train_y)
    np.save(os.path.join(d, "test_x.npy"), test_x)
    np.save(os.path.join(d, "test_y.npy"), test_y)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"num_classes": 200, "img_size": 64, "label_noise": label_noise,
                   "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}, f)


def load_vision(data_dir: str, dataset: str):
    d = os.path.join(data_dir, dataset)
    os.makedirs(d, exist_ok=True)
    if not os.path.exists(os.path.join(d, "train_x.npy")):
        {"cifar10": lambda p: _prepare_cifar10(p, 0.0),
         "cifar10n10": lambda p: _prepare_cifar10(p, 0.10),
         "cifar10n20": lambda p: _prepare_cifar10(p, 0.20),
         "cifar10n40": lambda p: _prepare_cifar10(p, 0.40),
         "cifar100": lambda p: _prepare_cifar100(p, 0.0),
         "cifar100n10": lambda p: _prepare_cifar100(p, 0.10),
         "cifar100n20": lambda p: _prepare_cifar100(p, 0.20),
         "cifar100n40": lambda p: _prepare_cifar100(p, 0.40),
         "tinyimagenet": lambda p: _prepare_tinyimagenet(p, 0.0),
         "tinyimagenetn20": lambda p: _prepare_tinyimagenet(p, 0.20)}[dataset](d)
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    vmeta = {"num_classes": meta.get("num_classes", 10),
             "img_size": meta.get("img_size", 32),
             "mean": meta.get("mean", [0.4914, 0.4822, 0.4465]),
             "std": meta.get("std", [0.247, 0.243, 0.261])}
    return (np.load(os.path.join(d, "train_x.npy"), mmap_mode="r"),
            np.load(os.path.join(d, "train_y.npy"), mmap_mode="r"),
            np.load(os.path.join(d, "test_x.npy"), mmap_mode="r"),
            np.load(os.path.join(d, "test_y.npy"), mmap_mode="r"), vmeta)


def load(data_dir: str, dataset: str = "shakespeare"):
    d = os.path.join(data_dir, dataset)
    os.makedirs(d, exist_ok=True)
    if not os.path.exists(os.path.join(d, "train.bin")):
        {"shakespeare": _prepare_shakespeare, "shakespeare_bytes": _prepare_shakespeare_bytes,
         "enwik8": _prepare_enwik8, "enwik8p10": _prepare_enwik8_noisy,
         "fineweb": _prepare_fineweb}[dataset](d)
        n_train = len(np.memmap(os.path.join(d, "train.bin"), dtype=np.uint16, mode="r"))
        print(f"prepared {dataset}: {n_train} train tokens")
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    train = np.memmap(os.path.join(d, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(d, "val.bin"), dtype=np.uint16, mode="r")
    return train, val, meta["vocab_size"]
