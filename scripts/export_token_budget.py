#!/usr/bin/env python3
"""Export only IDs/costs; never open evaluation labels."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.token_budget import export_plan

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = export_plan(Path(args.predictions), Path(args.output))
    print(f"Exported {len(result['items'])} answer-free cost targets")
