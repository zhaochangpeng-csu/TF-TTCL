#!/usr/bin/env python3
"""Download a revisioned model snapshot through the installed ModelScope SDK."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from modelscope import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--revision", default="master")
    parser.add_argument("--local-dir", required=True)
    args = parser.parse_args()
    destination = Path(args.local_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved = snapshot_download(
        args.model_id,
        revision=args.revision,
        local_dir=str(destination),
    )
    manifest = {
        "model_id": args.model_id,
        "requested_revision": args.revision,
        "local_dir": str(Path(resolved).resolve()),
        "downloaded_at": time.time(),
    }
    (destination / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
