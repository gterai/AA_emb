# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_MAP = {
    "ank": "get_emb_ankh.py",
    "ank3": "get_emb_ankh.py",
    "esm2": "get_emb_esm2.py",
    "esm2L": "get_emb_esm2.py",
    "T5b": "get_emb_protT5.py",
    "T5u": "get_emb_protT5.py",
}


def main():
    parser = argparse.ArgumentParser(
        description="Unified wrapper for embedding extraction scripts."
    )
    parser.add_argument("xlsx", help="input excel file")
    parser.add_argument("out_pkl", help="output pickle file")
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(SCRIPT_MAP.keys()),
        help="embedding model type",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    script_name = SCRIPT_MAP[args.model]
    cmd = [
        sys.executable,
        str(script_dir / script_name),
        args.xlsx,
        args.out_pkl,
        "--mtype",
        args.model,
    ]

    print("Running:", " ".join(cmd), file=sys.stderr)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
