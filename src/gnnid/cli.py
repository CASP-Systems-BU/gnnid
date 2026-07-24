"""gnnid CLI: ingest | train | score | eval."""
from __future__ import annotations

import argparse
import sys

from .config import load_config


def _add_common(p):
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="key=value", help="dotted config override")
    p.add_argument("--repo-root", default=".")


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser(prog="gnnid")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("ingest", "train", "score", "eval"):
        sp = sub.add_parser(name)
        _add_common(sp)

    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)

    if args.cmd == "ingest":
        from .ingest.run_dir import ingest_all
        ids = ingest_all(cfg, args.repo_root)
        print(f"[ingest] {len(ids)} runs -> {cfg.data.parquet_dir}")
    elif args.cmd == "train":
        from .train import run_training
        run_training(cfg, args.repo_root)
    elif args.cmd == "score":
        from .score import run_scoring
        run_scoring(cfg, args.repo_root)
    elif args.cmd == "eval":
        from .eval import run_eval
        run_eval(cfg, args.repo_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
