"""Fine-tune I-JEPA for cross-sensor fingerprint matching (single GPU).

    python main_fp.py --fname configs/fp_vith14_ft.yaml                         # train, then test best.pt
    python main_fp.py --fname configs/fp_vith14_ft.yaml --eval_only --zero_shot # pretrained baseline
    python main_fp.py --fname configs/fp_vith14_ft.yaml --eval_only             # test logs/.../best.pt

Test embeddings are saved as <logging.folder>/test_emb_<name>.pt; compare against
the fin-jepa checkpoints with fin-jepa/scratch/eval_report.py --emb NAME:PATH.
"""

import argparse
import logging
import pprint
from pathlib import Path

import yaml

from src.finetune_fp import evaluate, train

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--fname", default="configs/fp_vith14_ft.yaml")
parser.add_argument("--eval_only", action="store_true")
parser.add_argument("--zero_shot", action="store_true", help="with --eval_only: pretrained encoder, no head")
parser.add_argument("--checkpoint", default=None, help="with --eval_only: default <logging.folder>/best.pt")
parser.add_argument("--set", nargs="*", default=[], metavar="SECTION.KEY=VALUE",
                    help="config overrides, e.g. optimization.epochs=2 data.num_workers=0")


def apply_overrides(cfg, overrides):
    for item in overrides:
        key, value = item.split("=", 1)
        section, name = key.split(".", 1)
        cfg[section][name] = yaml.safe_load(value)
    return cfg


if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    with open(args.fname) as f:
        cfg = apply_overrides(yaml.safe_load(f), args.set)
    pprint.pprint(cfg)
    if args.eval_only:
        ckpt = args.checkpoint or Path(cfg["logging"]["folder"]) / "best.pt"
        evaluate(cfg, checkpoint_path=None if args.zero_shot else ckpt, zero_shot=args.zero_shot)
    else:
        train(cfg)
