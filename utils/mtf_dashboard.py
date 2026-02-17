from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any


def _to_float(v: Any, default: float | None = None) -> float | None:
    try:
        if v is None:
            return default
        out = float(v)
        if out != out:  # NaN guard
            return default
        return out
    except Exception:
        return default


def _to_int(v: Any, default: int | None = None) -> int | None:
    try:
        if v is None:
            return default
        return int(float(v))
    except Exception:
        return default


def _read_json_file(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_csv_file(path: Path, *, max_rows: int = 20000) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= int(max_rows):
                    break
                rows.append({str(k): str(v) for k, v in dict(row).items()})
    except Exception:
        return []
    return rows


def _file_info(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(path),
        "exists": bool(path.exists()),
    }
    if not path.exists():
        return info
    try:
        st = path.stat()
        info["size_bytes"] = int(st.st_size)
        info["updated_at_ms"] = int(st.st_mtime * 1000)
    except Exception:
        pass
    return info


def _default_progress() -> list[dict[str, Any]]:
    return [
        {
            "status": "done",
            "title": "Task 1: MTF 잠재벡터 추출 인터페이스 확장",
            "detail": "MTFMultiTaskNet.forward에 return_latent 옵션을 추가했고, 기존 출력(win/long/short/hold/exit)은 유지한 채 latent(96d)를 선택적으로 반환하도록 반영했습니다.",
        },
        {
            "status": "done",
            "title": "Task 2: MTF-OFI/VPIN Cross-Attention 모듈 구현",
            "detail": "OFI/VPIN 시계열을 1D-CNN으로 96차원 임베딩 후 MultiheadAttention(Query=MTF latent, Key/Value=signal)으로 융합하는 모델을 추가했습니다. attention weights 반환을 지원합니다.",
        },
        {
            "status": "done",
            "title": "Task 3: Captum 기반 XAI 분석 스크립트 구축",
            "detail": "Integrated Gradients + SmoothGrad(NoiseTunnel)로 시점별 OFI/VPIN 기여도를 계산하고, attention overlay/3D 산점도/임계치 후보 CSV를 생성하는 분석 파이프라인을 추가했습니다.",
        },
        {
            "status": "done",
            "title": "누수 방지 정렬 및 결측치 내구성",
            "detail": "신호 시퀀스는 반드시 end_ts 이전 데이터만 사용하도록 필터링했으며, NaN/Inf는 fill(0) 및 mask로 처리해 학습/해석 안정성을 확보했습니다.",
        },
    ]


def _default_timeline() -> list[dict[str, Any]]:
    return [
        {
            "date": "2026-02-17",
            "title": "MTF latent 반환 스펙 확정",
            "detail": "기존 추론 경로를 깨지 않으면서 return_latent=True일 때만 latent를 노출하도록 인터페이스를 고정했습니다.",
        },
        {
            "date": "2026-02-17",
            "title": "Cross-Attention 네트워크 통합",
            "detail": "96차원 임베딩 정합성, all-masked edge case, attention weight shape(B,H,1,L) 규격을 확정했습니다.",
        },
        {
            "date": "2026-02-17",
            "title": "XAI/시각화 파이프라인 완성",
            "detail": "IG 및 SmoothGrad 결과를 sample_attribution.csv와 그래프(attention overlay, OFI-win-VPIN scatter)로 저장하도록 구성했습니다.",
        },
    ]


def _normalize_progress(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return _default_progress()
    out: list[dict[str, Any]] = []
    for row in items:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "status": str(row.get("status") or "pending").strip().lower(),
                "title": str(row.get("title") or "").strip(),
                "detail": str(row.get("detail") or "").strip(),
            }
        )
    return out or _default_progress()


def _normalize_timeline(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return _default_timeline()
    out: list[dict[str, Any]] = []
    for row in items:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "date": str(row.get("date") or "").strip(),
                "title": str(row.get("title") or "").strip(),
                "detail": str(row.get("detail") or "").strip(),
            }
        )
    return out or _default_timeline()


def _parse_threshold_candidates(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        q = _to_float(r.get("win_prob_quantile"))
        thr = _to_float(r.get("win_prob_threshold"))
        cnt = _to_int(r.get("count"))
        ofi_med = _to_float(r.get("ofi_median"))
        vpin_med = _to_float(r.get("vpin_median"))
        if q is None and thr is None and cnt is None:
            continue
        out.append(
            {
                "quantile": q,
                "win_prob_threshold": thr,
                "count": cnt,
                "ofi_median": ofi_med,
                "vpin_median": vpin_med,
            }
        )
    out.sort(key=lambda x: (x.get("quantile") if x.get("quantile") is not None else -1.0))
    return out


def _parse_top_attribution_steps(rows: list[dict[str, str]], *, top_k: int = 5) -> list[dict[str, Any]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    for r in rows:
        score = _to_float(r.get("attr_abs_total"), default=None)
        if score is None:
            continue
        scored.append(
            (
                float(score),
                {
                    "ts_ms": _to_int(r.get("ts_ms")),
                    "ofi": _to_float(r.get("ofi")),
                    "vpin": _to_float(r.get("vpin")),
                    "attr_abs_total": float(score),
                },
            )
        )
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[: max(1, int(top_k))]]


def _default_research_notes() -> list[str]:
    return [
        "Attention Weight는 모델이 어느 시점의 OFI/VPIN 토큰을 참조했는지 보여주며, 원인-결과를 단정하는 지표는 아닙니다.",
        "Integrated Gradients 기여도는 샘플별 민감도 기준이며, 운영 임계치 도출 시 배치 통계(분위수/중앙값)와 함께 해석해야 합니다.",
        "Lookback 동기화가 어긋나면 미래정보 누수로 해석 결과가 왜곡되므로 mtf_end_ts_ms 기준 정렬 검증을 항상 수행해야 합니다.",
    ]


def _default_next_actions() -> list[str]:
    return [
        "샘플을 고수익/대손실 구간으로 분리해 attribution 분포 차이를 비교",
        "threshold_candidates.csv 기반으로 OFI/VPIN 이중 임계치 grid search 수행",
        "SmoothGrad 샘플 수(nt_samples)와 stdev를 바꿔 attribution 안정성 민감도 점검",
    ]


def build_mtf_dashboard_payload(base_dir: Path, now_ts_ms: int | None = None) -> dict[str, Any]:
    now_ms_val = int(now_ts_ms if now_ts_ms is not None else int(time.time() * 1000))
    root = Path(base_dir)

    notes_path = root / "state" / "mtf_dashboard_notes.json"
    summary_path = root / "artifacts" / "mtf_signal_xai" / "summary.json"
    threshold_path = root / "artifacts" / "mtf_signal_xai" / "threshold_candidates.csv"
    attribution_path = root / "artifacts" / "mtf_signal_xai" / "sample_attribution.csv"

    notes = _read_json_file(notes_path)
    if not isinstance(notes, dict):
        notes = {}

    summary = _read_json_file(summary_path)
    if not isinstance(summary, dict):
        summary = {}

    threshold_rows = _read_csv_file(threshold_path, max_rows=200)
    attribution_rows = _read_csv_file(attribution_path, max_rows=5000)

    progress = _normalize_progress(notes.get("progress"))
    timeline = _normalize_timeline(notes.get("timeline"))
    research_notes = notes.get("research_notes")
    if not isinstance(research_notes, list):
        research_notes = _default_research_notes()
    research_notes = [str(x) for x in research_notes if str(x).strip()]
    if not research_notes:
        research_notes = _default_research_notes()

    next_actions = notes.get("next_actions")
    if not isinstance(next_actions, list):
        next_actions = _default_next_actions()
    next_actions = [str(x) for x in next_actions if str(x).strip()]
    if not next_actions:
        next_actions = _default_next_actions()

    sample = {
        "selected_sample_index": _to_int(summary.get("selected_sample_index")),
        "selected_sample_id": summary.get("selected_sample_id"),
        "selected_end_ts_ms": _to_int(summary.get("selected_end_ts_ms")),
        "selected_win_prob": _to_float(summary.get("selected_win_prob")),
    }
    top_from_summary = summary.get("top_contributing_steps")
    if isinstance(top_from_summary, list) and top_from_summary:
        top_steps = [
            {
                "ts_ms": _to_int(x.get("ts_ms")) if isinstance(x, dict) else None,
                "ofi": _to_float(x.get("ofi")) if isinstance(x, dict) else None,
                "vpin": _to_float(x.get("vpin")) if isinstance(x, dict) else None,
                "attr_abs_total": _to_float(x.get("attr_abs_total")) if isinstance(x, dict) else None,
            }
            for x in top_from_summary
            if isinstance(x, dict)
        ]
        top_steps = [x for x in top_steps if x]
    else:
        top_steps = _parse_top_attribution_steps(attribution_rows, top_k=5)

    threshold_candidates = _parse_threshold_candidates(threshold_rows)

    payload = {
        "ok": True,
        "title": str(notes.get("title") or "MTF-XAI 연구 진행 현황"),
        "overview": str(
            notes.get("overview")
            or "MTF 이미지 잠재벡터(96d)와 OFI/VPIN 시계열을 Cross-Attention으로 결합해, 방향성 급등/급락 구간의 설명 가능한 상호작용을 분석합니다."
        ),
        "updated_at_ms": now_ms_val,
        "progress": progress,
        "timeline": timeline,
        "research": {
            "sample": sample,
            "threshold_candidates": threshold_candidates,
            "top_attribution_steps": top_steps,
            "notes": research_notes,
            "next_actions": next_actions,
        },
        "files": {
            "notes": _file_info(notes_path),
            "summary_json": _file_info(summary_path),
            "threshold_candidates_csv": _file_info(threshold_path),
            "sample_attribution_csv": _file_info(attribution_path),
        },
    }
    return payload
