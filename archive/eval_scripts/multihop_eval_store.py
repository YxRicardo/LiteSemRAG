"""Read/write merged HotpotQA multihop eval results."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("reports/multihop_eval_all.json")

_META = {
    "dataset": "HotpotQA distractor dev (first 500)",
    "cache": "cache/hotpotqa_latest_framework_index/litesemrag_hotpotqa_500_fastidx.pkl",
}


def load_all() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text(encoding="utf-8"))
    return {"meta": dict(_META)}


def save_all(data: dict) -> None:
    meta = dict(_META)
    meta.update(data.get("meta") or {})
    data["meta"] = meta
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
