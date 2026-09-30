"""Unlabelled fingerprint images for I-JEPA pretraining, read from
fin-jepa/dataset/pretrain_manifest.csv (built by build_pretrain_manifest.py).

Every image is resampled to a common ppi before cropping, matching the fine-tune
pipeline (fin-jepa/scratch/fp_preprocess.py), so ridge period is not a sensor cue.
"""

import csv
import sys
from logging import getLogger
from pathlib import Path

import torch
from PIL import Image

logger = getLogger()

TARGET_PPI = 500
# native ppi where the manifest has none (LivDet: fp_preprocess.SENSOR_PPI; FVC: FVC reports)
LIVDET_PPI = {"Biometrika": 569, "Identix": 686, "Italdata": 500, "Digital": 500, "Sagem": 500}
FVC_PPI = {("FVC2002", "Db2"): 569, ("FVC2004", "Db3"): 512}


def native_ppi(row, img):
    if row["ppi"]:
        return int(row["ppi"])
    if row["dataset"].startswith("LivDet"):
        return LIVDET_PPI.get(row["sensor"], TARGET_PPI)
    if row["dataset"].startswith("FVC"):
        return FVC_PPI.get((row["dataset"], row["sensor"]), TARGET_PPI)
    dpi = img.info.get("dpi")  # NIST302a: resolution is in the PNG header
    return int(round(dpi[0])) if dpi and dpi[0] > 1 else TARGET_PPI


def _foreground_crop(fin_jepa_root):
    sys.path.insert(0, str(Path(fin_jepa_root).resolve()))
    from fp_preprocess import foreground_crop
    return foreground_crop


def livdet_test_paths(livdet_manifest, images_root):
    """Image paths of the LivDet rows the fine-tune/eval pipeline treats as test."""
    out = set()
    with open(livdet_manifest, newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("split") or r["source_split"]) != "testing":
                continue
            parts = Path(r["source_path"]).parts
            rel = Path(*parts[parts.index(r["dataset"]) + 1:])
            out.add(str(Path(images_root) / r["dataset"] / r["sensor"] / r["source_split"] / r["label"] / rel))
    return out


class FingerprintManifest(torch.utils.data.Dataset):

    def __init__(self, manifest, split="train", transform=None, datasets=None, exclude_paths=(),
                 canvas=None, fin_jepa_root=None):
        with open(manifest, newline="") as f:
            rows = [r for r in csv.DictReader(f)
                    if (split is None or r["pretrain_split"] == split)
                    and (datasets is None or r["dataset"] in datasets)]
        exclude_paths = set(exclude_paths)
        n = len(rows)
        self.rows = [r for r in rows if r["path"] not in exclude_paths]
        self.transform = transform
        # canvas: size x size window around the ridge foreground (same as the fine-tune loader)
        self.canvas = canvas
        self.foreground_crop = _foreground_crop(fin_jepa_root) if canvas else None
        logger.info(f"fingerprint manifest {manifest}: {len(self.rows)} images "
                    f"(split={split}, {n - len(self.rows)} excluded)")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["path"])
        s = TARGET_PPI / native_ppi(r, img)
        img = img.convert("L")
        if abs(s - 1) > 1e-3:
            img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)
        if self.canvas:
            img = self.foreground_crop(img, self.canvas)
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


def make_fingerprint_manifest(
    transform,
    batch_size,
    manifest,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    split="train",
    datasets=None,
    exclude_livdet_test=True,
    livdet_manifest=None,
    livdet_images=None,
    canvas=None,
    fin_jepa_root=None,
    drop_last=True
):
    exclude = livdet_test_paths(livdet_manifest, livdet_images) if exclude_livdet_test else ()
    dataset = FingerprintManifest(manifest, split=split, transform=transform,
                                  datasets=datasets, exclude_paths=exclude,
                                  canvas=canvas, fin_jepa_root=fin_jepa_root)
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    logger.info('fingerprint unsupervised data loader created')
    return dataset, data_loader, dist_sampler
