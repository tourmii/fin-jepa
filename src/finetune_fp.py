"""Fine-tune a pretrained I-JEPA ViT for cross-sensor fingerprint matching.

Data, identity sampler, losses and the test protocol are imported from fin-jepa
(`data.fin_jepa_root` in the config) so results are directly comparable with the
fin-jepa checkpoints in its eval_report.py table.

  encoder : I-JEPA ViT (target encoder weights), first `n_frozen` blocks frozen
  pooling : mean over patch tokens -> EmbeddingHead -> L2-normalised 128-d embedding
  loss    : SupCon over P x K identity batches (cross-sensor positives) + w_inv * centroid invariance
  select  : cross-sensor EER on a PERSON-disjoint carve-out of the training split
  test    : fin-jepa eval_report protocol (EER / TAR@FAR / rank-1) on the test split
"""

import contextlib
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

import src.models.vision_transformer as vit

logger = logging.getLogger()

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def import_fin_jepa(root):
    root = str(Path(root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    import eval_report
    import fingerprint_data
    import jepa_losses
    import samplers
    return fingerprint_data, samplers, jepa_losses, eval_report


# --------------------------------------------------------------------------- #
#  Pretrained weights
# --------------------------------------------------------------------------- #

def _from_hf(sd):
    """HF `IJepaModel` safetensors -> src.models.vision_transformer key layout."""
    out = {"patch_embed.proj.weight": sd["embeddings.patch_embeddings.projection.weight"],
           "patch_embed.proj.bias": sd["embeddings.patch_embeddings.projection.bias"],
           "pos_embed": sd["embeddings.position_embeddings"],
           "norm.weight": sd["layernorm.weight"], "norm.bias": sd["layernorm.bias"]}
    n = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("encoder.layer."))
    for i in range(n):
        p, q = f"encoder.layer.{i}.", f"blocks.{i}."
        for t in ("weight", "bias"):
            out[q + f"attn.qkv.{t}"] = torch.cat([sd[p + f"attention.attention.{m}.{t}"]
                                                  for m in ("query", "key", "value")])
            out[q + f"attn.proj.{t}"] = sd[p + f"attention.output.dense.{t}"]
            out[q + f"norm1.{t}"] = sd[p + f"layernorm_before.{t}"]
            out[q + f"norm2.{t}"] = sd[p + f"layernorm_after.{t}"]
            out[q + f"mlp.fc1.{t}"] = sd[p + f"intermediate.dense.{t}"]
            out[q + f"mlp.fc2.{t}"] = sd[p + f"output.dense.{t}"]
    return out


def load_pretrained(encoder, path):
    """Accepts the HF safetensors (facebook/ijepa_vith14_1k) or an official
    I-JEPA .pth.tar (uses its `target_encoder`, as in the paper's evaluations)."""
    path = str(path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = _from_hf(load_file(path))
    else:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck["target_encoder"].items()}
    if sd["pos_embed"].dim() == 2:
        sd["pos_embed"] = sd["pos_embed"][None]
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"pretrained load mismatch: missing={missing} unexpected={unexpected}")
    logger.info(f"loaded pretrained encoder from {path}")


# --------------------------------------------------------------------------- #
#  Model
# --------------------------------------------------------------------------- #

class FingerprintEmbedder(nn.Module):
    """Frozen ViT prefix (no_grad) -> trainable suffix -> mean pool -> head.
    head=None gives the zero-shot baseline (L2-normalised pooled tokens)."""

    def __init__(self, encoder, head, n_frozen, grad_checkpointing=False):
        super().__init__()
        self.encoder, self.head = encoder, head
        self.n_frozen, self.grad_checkpointing = n_frozen, grad_checkpointing
        if n_frozen > 0:
            for m in [encoder.patch_embed, *encoder.blocks[:n_frozen]]:
                m.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        for blk in self.encoder.blocks[:self.n_frozen]:
            blk.eval()
        return self

    def forward(self, x):
        enc = self.encoder
        frozen = torch.no_grad() if self.n_frozen > 0 else contextlib.nullcontext()
        with frozen:
            x = enc.patch_embed(x)
            x = x + enc.interpolate_pos_encoding(x, enc.pos_embed)
            for blk in enc.blocks[:self.n_frozen]:
                x = blk(x)
        for blk in enc.blocks[self.n_frozen:]:
            if self.grad_checkpointing and self.training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        pooled = enc.norm(x).mean(1)
        return self.head(pooled.float()) if self.head is not None else F.normalize(pooled.float(), dim=-1)

    def trainable_state(self):
        """Only what fine-tuning changes: the unfrozen blocks, final norm and head."""
        keep = {f"encoder.blocks.{i}." for i in range(self.n_frozen, len(self.encoder.blocks))}
        keep |= {"encoder.norm.", "head."}
        if self.n_frozen == 0:
            keep |= {"encoder.patch_embed."}
        return {k: v for k, v in self.state_dict().items() if any(k.startswith(p) for p in keep)}


def build_model(cfg, fp_losses, device):
    m = cfg["model"]
    encoder = getattr(vit, m["arch"])(img_size=[m["img_size"]], patch_size=m["patch_size"],
                                      drop_path_rate=m.get("drop_path_rate", 0.0))
    load_pretrained(encoder, m["pretrained"])
    head = fp_losses.EmbeddingHead(in_dim=encoder.embed_dim, hidden=m["head_hidden"], out_dim=m["embed_dim"]) \
        if m.get("use_head", True) else None
    model = FingerprintEmbedder(encoder, head, m["n_frozen"], m.get("grad_checkpointing", False))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    logger.info(f"{m['arch']}/{m['patch_size']} @ {m['img_size']}: {n_all / 1e6:.0f}M params, "
                f"{n_train / 1e6:.0f}M trainable (blocks {m['n_frozen']}..{len(encoder.blocks) - 1} + head)")
    return model.to(device)


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #

class RandomRotateBorderFill:
    """Small in-plane rotation; the exposed corners get the image's background grey
    (median border level) instead of black, so rotation leaves no artificial edge."""

    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, img):
        a = np.asarray(img.convert("L"))
        fill = int(np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])))
        return img.rotate(random.uniform(-self.degrees, self.degrees), resample=2, fillcolor=(fill,) * 3)


def fp_transforms(img_size, train, rotate_deg):
    norm = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        return transforms.Compose([transforms.CenterCrop(img_size), *norm])
    aug = [RandomRotateBorderFill(rotate_deg)] if rotate_deg else []
    return transforms.Compose([*aug, transforms.RandomCrop(img_size), *norm])


class IndexedView(Dataset):
    """(img, identity label, sensor id) for absolute dataset indices."""

    def __init__(self, ds, sensor_ids):
        self.ds, self.sensor_ids = ds, sensor_ids

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, label, _ = self.ds[i]
        return img, label, int(self.sensor_ids[i])


def build_data(cfg, fp_data):
    d, img = cfg["data"], cfg["model"]["img_size"]
    common = dict(split="training", label_mode="subject", id_qualities=("finger",))
    # train canvas has a margin so RandomCrop / rotation see real ridges, not padding
    ds_train = fp_data.FingerprintDataset(transform=fp_transforms(img, True, d["rotate_deg"]),
                                          canvas=img + d["crop_margin"], **common)
    ds_eval = fp_data.FingerprintDataset(transform=fp_transforms(img, False, 0), canvas=img, **common)
    assert ds_train.subject_ids == ds_eval.subject_ids, "train/eval views must share row order"

    persons = sorted(set(ds_train.person_keys))
    rng = random.Random(cfg["meta"]["seed"])
    val_persons = set(rng.sample(persons, max(1, int(round(len(persons) * d["val_frac"])))))
    tr_idx = [i for i, p in enumerate(ds_train.person_keys) if p not in val_persons]
    va_idx = [i for i, p in enumerate(ds_train.person_keys) if p in val_persons]
    sensor_ids = torch.tensor([fp_data.SENSORS.index(s) for s in ds_train.sensor_names])
    logger.info(f"train {len(tr_idx)} images | val {len(va_idx)} images "
                f"({len(val_persons)}/{len(persons)} persons held out)")
    return ds_train, ds_eval, tr_idx, va_idx, sensor_ids


@torch.no_grad()
def embed(model, ds, indices, device, batch_size, num_workers):
    model.eval()
    view = ds if indices is None else Subset(ds, indices)
    out = []
    for batch in DataLoader(view, batch_size=batch_size, num_workers=num_workers):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out.append(model(batch[0].to(device, non_blocking=True)).float().cpu())
    return torch.cat(out)


# --------------------------------------------------------------------------- #
#  Validation (person-disjoint carve-out of the training split)
# --------------------------------------------------------------------------- #

def validate(model, ds_eval, va_idx, device, cfg, fp_eval):
    E = embed(model, ds_eval, va_idx, device, cfg["data"]["eval_batch_size"], cfg["data"]["num_workers"])
    keys = [ds_eval.subject_ids[i] for i in va_idx]
    persons = [ds_eval.person_keys[i] for i in va_idx]
    sensors = [ds_eval.sensor_names[i] for i in va_idx]
    uid = {k: n for n, k in enumerate(sorted(set(keys)))}
    upid = {k: n for n, k in enumerate(sorted(set(persons)))}
    ident = np.array([uid[k] for k in keys])
    person = np.array([upid[k] for k in persons])
    sensor = np.array([hash(s) for s in sensors])
    P = fp_eval.collect_pairs(E, ident, person, sensor, max_impostors=10**9, seed=0)
    out = {}
    for name, mask in (("cross", P["cross"]), ("within", ~P["cross"])):
        m = fp_eval.roc_metrics(P["score"][mask], P["genuine"][mask])
        if m is not None:
            out[f"val_eer_{name}"] = m["eer"]
            out[f"val_tar1e-2_{name}"] = m["tar@0.01"]
    r1 = fp_eval.rank1_accuracy(E, ident, sensor)
    out["val_rank1_cross"] = r1["cross"]["rank1"]
    out["val_rank1_within"] = r1["within"]["rank1"]
    return out


# --------------------------------------------------------------------------- #
#  Optimisation
# --------------------------------------------------------------------------- #

def param_groups(model, opt_cfg):
    """Layer-wise LR decay over the trainable blocks; head at head_lr; no weight
    decay on biases / norms / pos-embeddings."""
    enc = model.encoder
    depth = len(enc.blocks)
    groups = {}

    def add(p, lr, wd):
        key = (lr, wd)
        groups.setdefault(key, {"params": [], "lr": lr, "base_lr": lr, "weight_decay": wd})["params"].append(p)

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        wd = 0.0 if p.ndim <= 1 or name.endswith("pos_embed") else opt_cfg["weight_decay"]
        if name.startswith("head."):
            add(p, opt_cfg["head_lr"], wd)
            continue
        if name.startswith("encoder.blocks."):
            layer = int(name.split(".")[2]) + 1
        elif name.startswith("encoder.norm."):
            layer = depth + 1
        else:  # patch_embed (only trainable when n_frozen == 0)
            layer = 0
        add(p, opt_cfg["lr"] * opt_cfg["layer_decay"] ** (depth + 1 - layer), wd)
    return list(groups.values())


def lr_factor(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


# --------------------------------------------------------------------------- #
#  Entry points
# --------------------------------------------------------------------------- #

def _setup(cfg):
    fp_data, fp_samplers, fp_losses, fp_eval = import_fin_jepa(cfg["data"]["fin_jepa_root"])
    device = torch.device(cfg["meta"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg["meta"]["seed"])
    random.seed(cfg["meta"]["seed"])
    np.random.seed(cfg["meta"]["seed"])
    out_dir = Path(cfg["logging"]["folder"])
    out_dir.mkdir(parents=True, exist_ok=True)
    return fp_data, fp_samplers, fp_losses, fp_eval, device, out_dir


def train(cfg):
    fp_data, fp_samplers, fp_losses, fp_eval, device, out_dir = _setup(cfg)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    model = build_model(cfg, fp_losses, device)
    ds_train, ds_eval, tr_idx, va_idx, sensor_ids = build_data(cfg, fp_data)

    o, d = cfg["optimization"], cfg["data"]
    sampler = fp_samplers.PKCrossSensorSampler(ds_train.subject_ids, ds_train.sensor_names, tr_idx,
                                               P=d["P"], K=d["K"], cross_frac=d["cross_frac"],
                                               steps_per_epoch=d.get("steps_per_epoch"), seed=cfg["meta"]["seed"])
    loader = DataLoader(IndexedView(ds_train, sensor_ids), batch_sampler=sampler,
                        num_workers=d["num_workers"], pin_memory=device.type == "cuda",
                        persistent_workers=d["num_workers"] > 0)
    opt = torch.optim.AdamW(param_groups(model, o), betas=(0.9, 0.999))
    total, warmup = o["epochs"] * len(sampler), o["warmup_epochs"] * len(sampler)
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")

    history, best, step = [], float("inf"), 0
    if o.get("validate_zero_shot", True):
        rec = {"epoch": 0, **validate(model, ds_eval, va_idx, device, cfg, fp_eval)}
        logger.info(f"epoch 0 (pretrained + random head) {rec}")
        history.append(rec)

    for epoch in range(o["epochs"]):
        sampler.set_epoch(epoch)
        model.train()
        t0, sums, n = time.time(), {"loss": 0.0, "id": 0.0, "inv": 0.0}, 0
        for imgs, labels, sensors in loader:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_factor(step, total, warmup)
            imgs, labels, sensors = imgs.to(device, non_blocking=True), labels.to(device), sensors.to(device)
            with torch.autocast(**amp):
                emb = model(imgs)
            emb = emb.float()
            l_id = fp_losses.supcon_loss(emb, labels, sensors, tau=o["tau"],
                                         cross_sensor_positives=o["cross_sensor_positives"])
            l_inv = fp_losses.centroid_invariance_loss(emb, labels, sensors) if o["w_inv"] > 0 else emb.new_zeros(())
            loss = l_id + o["w_inv"] * l_inv
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], o["clip_grad"])
            opt.step()
            sums["loss"] += loss.item(); sums["id"] += l_id.item(); sums["inv"] += l_inv.item()
            n, step = n + 1, step + 1
            if n % cfg["logging"]["log_every"] == 0:
                logger.info(f"ep {epoch + 1} it {n}/{len(sampler)} loss {sums['loss'] / n:.4f} "
                            f"lr {opt.param_groups[0]['lr']:.2e}")

        rec = {"epoch": epoch + 1, **{k: v / max(n, 1) for k, v in sums.items()}, "sec": time.time() - t0}
        if (epoch + 1) % o["val_every"] == 0 or epoch + 1 == o["epochs"]:
            rec.update(validate(model, ds_eval, va_idx, device, cfg, fp_eval))
        logger.info(f"epoch {epoch + 1}: {rec}")
        history.append(rec)

        ck = {"epoch": epoch + 1, "cfg": cfg, "state": model.trainable_state(), "history": history}
        torch.save(ck, out_dir / "latest.pt")
        score = rec.get("val_eer_cross", rec.get("val_eer_within"))
        if score is not None and score < best:
            best = score
            torch.save(ck, out_dir / "best.pt")
            logger.info(f"  new best val EER {best:.2f}% -> {out_dir / 'best.pt'}")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    del model, opt, loader
    torch.cuda.empty_cache()
    final = out_dir / "best.pt"
    if not final.exists():
        logger.warning("no validation score was ever computed; testing latest.pt")
        final = out_dir / "latest.pt"
    return evaluate(cfg, checkpoint_path=final)


def evaluate(cfg, checkpoint_path=None, zero_shot=False, name=None):
    """Embed the TEST split with the fine-tuned model (or the pretrained encoder
    when zero_shot) and run the fin-jepa eval_report protocol on it."""
    fp_data, _, fp_losses, fp_eval, device, out_dir = _setup(cfg)
    if zero_shot:
        cfg = {**cfg, "model": {**cfg["model"], "use_head": False, "n_frozen": 0}}
    model = build_model(cfg, fp_losses, device)
    if not zero_shot:
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        missing = set(model.trainable_state()) - set(ck["state"])
        if missing:
            raise RuntimeError(f"checkpoint lacks {len(missing)} fine-tuned tensors, e.g. {next(iter(missing))}")
        model.load_state_dict(ck["state"], strict=False)
        logger.info(f"loaded fine-tuned weights from {checkpoint_path} (epoch {ck['epoch']})")

    img = cfg["model"]["img_size"]
    ds = fp_data.FingerprintDataset(split="testing", label_mode="subject", id_qualities=("finger",),
                                    transform=fp_transforms(img, False, 0), canvas=img)
    E = embed(model, ds, None, device, cfg["data"]["eval_batch_size"], cfg["data"]["num_workers"])
    name = name or ("ijepa_zs" if zero_shot else "ijepa_ft")
    torch.save({"E": E, "paths": [str(p) for p in ds.paths]}, out_dir / f"test_emb_{name}.pt")

    ev = cfg["evaluation"]
    rows = fp_eval.evaluate_embeddings(E, ds, ev["max_impostors"], ev["n_boot"], cfg["meta"]["seed"])
    table = fp_eval.format_table({name: rows}, [name])
    logger.info("\n" + table)
    (out_dir / f"test_report_{name}.md").write_text(table + "\n")
    (out_dir / f"test_report_{name}.json").write_text(json.dumps(rows, indent=2, default=str))
    return rows
