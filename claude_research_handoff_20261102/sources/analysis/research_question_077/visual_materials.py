#!/usr/bin/env python3
"""Create prediction-blind visual sheets from a frozen GT sample manifest.

This utility never assigns visibility, action, or annotation-validity labels.
Only OpenCV and numpy are required beyond the Python standard library.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import random
import sys
from typing import Any


ALIASES = {
    "blind_id": ("blind_id", "sample_id", "audit_id"),
    "video": ("video", "video_name", "full_video_id", "full_id", "video_id"),
    "gt_index": ("gt_index", "gt_idx", "relation_index", "gt_id"),
    "sid": ("sid", "subject_tid", "subject_id", "subject"),
    "oid": ("oid", "object_tid", "object_id", "object"),
    "predicate": ("predicate", "relation", "pred"),
    "begin": ("begin_fid", "begin", "start", "start_frame", "gt_begin", "gt_start"),
    "end": ("end_fid", "end", "end_frame", "gt_end"),
    "fps": ("fps", "video_fps"),
}
AUDIT_FIELDS = (
    "reviewer", "review_type", "reviewed", "subject_visible", "object_visible",
    "predicate_visually_identifiable", "onset_boundary_clear", "offset_boundary_clear",
    "annotation_plausible", "needs_full_video", "notes",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pick(row: dict[str, str], key: str, default: Any = None) -> Any:
    for alias in ALIASES[key]:
        if row.get(alias, "") != "":
            return row[alias]
    return default


def number(value: Any, name: str) -> int:
    if value is None:
        raise ValueError(f"missing {name}")
    result = int(value)
    return result


def safe_id(value: Any) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    if not result or result in {".", ".."}:
        raise ValueError("invalid blind identifier")
    return result


def locate(root: Path, video: str, suffix: str) -> Path:
    """Use exact full-id/group paths first; reject ambiguous fallback matches."""
    video = str(video).removesuffix(".mp4").removesuffix(".json")
    if "/" in video:
        group, basename = video.rsplit("/", 1)
        candidates = [root / group / (basename + suffix)]
    elif "_" in video:
        group, basename = video.split("_", 1)
        candidates = [root / group / (basename + suffix), root / (video + suffix)]
    else:
        candidates = [root / (video + suffix)]
    matches = [p for p in candidates if p.is_file()]
    if not matches:
        basename = video.rsplit("_", 1)[-1]
        matches = list(root.glob(f"*/{basename}{suffix}"))
    matches = list(dict.fromkeys(p.resolve() for p in matches))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {suffix} for {video}, got {len(matches)}")
    return matches[0]


def resolve_sample(row: dict[str, str], annotation: dict[str, Any], index: int) -> dict[str, Any]:
    gt_value = pick(row, "gt_index")
    relation = None
    if gt_value is not None:
        gt_idx = number(gt_value, "gt_index")
        relation = annotation["relation_instances"][gt_idx]
    else:
        gt_idx = None
    resolved = {"blind_id": safe_id(pick(row, "blind_id", f"audit_{index:03d}")),
                "video": str(pick(row, "video")), "gt_index": gt_idx}
    for key, ann_key in (("sid", "subject_tid"), ("oid", "object_tid"),
                         ("begin", "begin_fid"), ("end", "end_fid")):
        value = pick(row, key, relation.get(ann_key) if relation else None)
        resolved[key] = number(value, key)
        if relation is not None and resolved[key] != int(relation[ann_key]):
            raise ValueError(f"manifest/annotation mismatch for {key}: {resolved[key]} != {relation[ann_key]}")
    resolved["predicate"] = str(pick(row, "predicate", relation.get("predicate") if relation else ""))
    if relation is not None and resolved["predicate"] != str(relation["predicate"]):
        raise ValueError("manifest/annotation predicate mismatch")
    resolved["fps"] = float(annotation.get("fps") or pick(row, "fps"))
    if resolved["end"] <= resolved["begin"] or resolved["fps"] <= 0:
        raise ValueError("invalid event bounds or fps")
    if pick(row, "fps") is not None and abs(float(pick(row, "fps")) - resolved["fps"]) > 1e-3:
        raise ValueError("manifest/annotation fps mismatch")
    return resolved


def requested_frames(begin: int, end: int, fps: float, count: int, margin: float) -> list[tuple[str, int, int]]:
    last = end - 1
    requests = [
        ("before", begin - round(margin * fps)), ("first", begin),
        ("quarter", round(begin + 0.25 * (last - begin))),
        ("middle", round(begin + 0.5 * (last - begin))),
        ("three-quarter", round(begin + 0.75 * (last - begin))),
        ("last", last), ("after", last + round(margin * fps)),
    ]
    return [(label, raw, max(0, min(count - 1, raw))) for label, raw in requests]


def bbox_map(annotation: dict[str, Any], frame: int) -> dict[int, list[float]]:
    result = {}
    trajectories = annotation.get("trajectories", [])
    if frame < 0 or frame >= len(trajectories):
        return result
    for obj in trajectories[frame]:
        b = obj["bbox"]
        result[int(obj["tid"])] = ([b[k] for k in ("xmin", "ymin", "xmax", "ymax")]
                                    if isinstance(b, dict) else list(b))
    return result


def draw_frame(frame: Any, boxes: dict[int, list[float]], sample: dict[str, Any], cell_w: int,
               cell_h: int, title: str) -> Any:
    import cv2
    import numpy as np
    h, w = frame.shape[:2]
    scale = min(cell_w / w, (cell_h - 42) / h)
    out = np.full((cell_h, cell_w, 3), 24, dtype=np.uint8)
    image = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))))
    ox = (cell_w - image.shape[1]) // 2
    oy = 36 + (cell_h - 42 - image.shape[0]) // 2
    out[oy:oy + image.shape[0], ox:ox + image.shape[1]] = image
    cv2.putText(out, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .52, (240, 240, 240), 1, cv2.LINE_AA)
    for role, key, color in (("S", "sid", (30, 30, 235)), ("O", "oid", (240, 140, 30))):
        tid = sample[key]
        if tid not in boxes:
            continue
        x1, y1, x2, y2 = boxes[tid]
        p1 = (ox + round(x1 * scale), oy + round(y1 * scale))
        p2 = (ox + round(x2 * scale), oy + round(y2 * scale))
        cv2.rectangle(out, p1, p2, color, 2)
        cv2.putText(out, f"{role} tid={tid}", (p1[0], max(42, p1[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    return out


def make_sheet(video_path: Path, annotation: dict[str, Any], sample: dict[str, Any],
               output: Path, cell_w: int, cell_h: int, margin: float) -> dict[str, Any]:
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open video")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        cap.release()
        raise RuntimeError("video frame count unavailable")
    ann_count = int(annotation.get("frame_count", len(annotation["trajectories"])))
    if count != ann_count:
        cap.release()
        raise RuntimeError(f"video/annotation frame-count mismatch: {count} != {ann_count}")
    sheet = np.full((2 * cell_h + 96, 4 * cell_w, 3), 24, dtype=np.uint8)
    cv2.putText(sheet, f"{sample['blind_id']} | video {sample['video']} | predicate {sample['predicate']}",
                (16, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(sheet, f"GT S tid={sample['sid']} (red) -> O tid={sample['oid']} (blue); "
                f"GT frames [{sample['begin']}, {sample['end']}); {sample['fps']:.3f} fps",
                (16, 56), cv2.FONT_HERSHEY_SIMPLEX, .62, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(sheet, "Before/after are context frames, NOT verified relation-negative frames.",
                (16, 80), cv2.FONT_HERSHEY_SIMPLEX, .56, (210, 210, 210), 1, cv2.LINE_AA)
    records = []
    try:
        for pos, (label, unclipped, fid) in enumerate(requested_frames(
                sample["begin"], sample["end"], sample["fps"], count, margin)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"cannot decode frame {fid}")
            reported = round(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if reported != fid:
                raise RuntimeError(f"seek reports frame {reported} instead of {fid}")
            boxes = bbox_map(annotation, fid)
            title = f"{label} | f={fid} | t={fid / sample['fps']:.3f}s"
            cell = draw_frame(frame, boxes, sample, cell_w, cell_h, title)
            row, col = divmod(pos, 4)
            sheet[96 + row * cell_h:96 + (row + 1) * cell_h, col * cell_w:(col + 1) * cell_w] = cell
            records.append({"position": label, "requested_unclipped_frame": unclipped,
                            "frame": fid, "reported_frame": reported,
                            "context_clipped": unclipped != fid,
                            "subject_has_GT_box": sample["sid"] in boxes,
                            "object_has_GT_box": sample["oid"] in boxes,
                            "decoded_pixels_sha256": hashlib.sha256(frame.tobytes()).hexdigest()})
    finally:
        cap.release()
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError("OpenCV failed to write sheet")
    return {"frames": records, "sheet_sha256": sha256(output), "frame_count": count}


def make_clip(video_path: Path, sample: dict[str, Any], output: Path, margin: float,
              max_width: int = 960) -> dict[str, Any]:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = max(0, sample["begin"] - round(margin * sample["fps"]))
    last = min(count, sample["end"] + round(margin * sample["fps"]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    writer = None
    written = 0
    try:
        for fid in range(first, last):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"clip decode failed at {fid}")
            h, w = frame.shape[:2]
            scale = min(1., max_width / w)
            width, height = max(2, round(w * scale) // 2 * 2), max(2, round(h * scale) // 2 * 2)
            if (width, height) != (w, h):
                frame = cv2.resize(frame, (width, height))
            if writer is None:
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                                         sample["fps"], (width, height))
                if not writer.isOpened():
                    raise RuntimeError("mp4v clip writer unavailable")
            writer.write(frame)
            written += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return {"clip_begin_frame": first, "clip_end_frame": last,
            "clip_frames_written": written, "clip_sha256": sha256(output),
            "clip_has_audio": False, "clip_has_box_overlay": False}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--video-root", required=True, type=Path)
    ap.add_argument("--annotation-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--context-seconds", type=float, default=.5)
    ap.add_argument("--cell-width", type=int, default=320)
    ap.add_argument("--cell-height", type=int, default=224)
    ap.add_argument("--shuffle-seed", type=int, default=113, help="Blind material presentation order only; does not change the selected sample.")
    ap.add_argument("--clips", action="store_true", help="Also create unannotated, silent local context clips.")
    args = ap.parse_args()
    if args.context_seconds < 0:
        ap.error("context seconds must be nonnegative")
    if args.out.exists() and any(args.out.iterdir()):
        ap.error("output must be new or empty; refusing to overwrite existing audit material")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sheets").mkdir()
    if args.clips:
        (args.out / "clips").mkdir()
    with args.manifest.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    random.Random(args.shuffle_seed).shuffle(rows)
    records, audit_rows, hashes, seen = [], [], {}, set()
    for i, row in enumerate(rows, 1):
        blind_id = safe_id(pick(row, "blind_id", f"audit_{i:03d}"))
        if blind_id in seen:
            raise ValueError(f"duplicate blind id: {blind_id}")
        seen.add(blind_id)
        result = {"blind_id": blind_id, "status": "failed", "error": ""}
        try:
            video = pick(row, "video")
            if video is None:
                raise ValueError("video identifier missing")
            video_path = locate(args.video_root, str(video), ".mp4")
            annotation_path = locate(args.annotation_root, str(video), ".json")
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            sample = resolve_sample(row, annotation, i)
            result.update(sample)
            for label, path in (("video", video_path), ("annotation", annotation_path)):
                key = str(path)
                if key not in hashes:
                    hashes[key] = sha256(path)
                result[f"source_{label}_path"] = key
                result[f"source_{label}_sha256"] = hashes[key]
            sheet = args.out / "sheets" / f"{blind_id}.png"
            result.update(make_sheet(video_path, annotation, sample, sheet,
                                     args.cell_width, args.cell_height, args.context_seconds))
            result["sheet_path"] = str(sheet.resolve())
            result["status"] = "sheet_complete"
            if args.clips:
                clip = args.out / "clips" / f"{blind_id}.mp4"
                result.update(make_clip(video_path, sample, clip, args.context_seconds))
                result["clip_path"] = str(clip.resolve())
                result["status"] = "sheet_and_clip_complete"
            audit_rows.append({"blind_id": blind_id, "sheet_file": f"sheets/{blind_id}.png",
                               "clip_file": f"clips/{blind_id}.mp4" if args.clips else "",
                               **{key: "" for key in AUDIT_FIELDS}})
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        records.append(result)
        print(f"{i}/{len(rows)} {blind_id}: {result['status']} {result['error']}", flush=True)
    (args.out / "materials.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    index_fields = ["blind_id", "status", "error", "video", "gt_index", "sid", "oid", "predicate",
                    "begin", "end", "fps", "source_video_path", "source_video_sha256",
                    "source_annotation_path", "source_annotation_sha256", "sheet_path", "sheet_sha256",
                    "clip_path", "clip_sha256"]
    write_csv(args.out / "materials.csv", records, index_fields)
    write_csv(args.out / "blind_audit.csv", audit_rows, ["blind_id", "sheet_file", "clip_file", *AUDIT_FIELDS])
    complete = sum(r["status"] in {"sheet_complete", "sheet_and_clip_complete"} and not r["error"] for r in records)
    summary = {"manifest_path": str(args.manifest.resolve()), "manifest_sha256": sha256(args.manifest),
               "script_sha256": sha256(Path(__file__)), "requested": len(rows), "complete": complete,
               "failed": len(rows) - complete, "semantic_audits_completed": 0,
               "prediction_fields_read": False, "context_seconds": args.context_seconds,
               "frames_per_sheet": 7, "layout": "4x2", "clips_requested": args.clips,
               "shuffle_seed": args.shuffle_seed,
               "note": "GT box presence is annotation availability, not visible entity/action evidence. "
                       "All semantic labels are blank; context frames are not verified negatives."}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if complete == len(rows) else 2


if __name__ == "__main__":
    sys.exit(main())
