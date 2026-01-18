from typing import Any, Dict

def _compress_reason_counts(counts: Any, *, top_k: int = 3) -> Dict[str, int]:
    if not isinstance(counts, dict) or not counts:
        return {}
    items = []
    other = 0
    for k, v in counts.items():
        try:
            key = str(k)
            val = int(v)
        except Exception:
            continue
        if val <= 0:
            continue
        items.append((key, val))
    if not items:
        return {}
    items.sort(key=lambda x: x[1], reverse=True)
    top = items[: max(0, int(top_k))]
    if len(items) > len(top):
        other = sum(v for _, v in items[len(top) :])
    out = {k: int(v) for k, v in top}
    if other > 0:
        out["_other"] = int(other)
    return out
