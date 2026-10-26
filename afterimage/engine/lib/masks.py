"""Keyframe-only object masks.
   candidates  : bgsub (flow-aware median background) ∪ upscaled flow-moving mask
   refinement  : RMBG-2.0 on ROI crops around each candidate (GPU 0, batch 1),
                 accepted only where it agrees with the candidate region;
                 fallback = candidate component itself.
   Never runs RMBG on every frame -- only on selected keyframes."""
from __future__ import annotations
import os, sys
from typing import List, Optional, Tuple
import cv2
import numpy as np

_RUNNER = None


def rmbg_runner():
    global _RUNNER
    if _RUNNER is None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import torch
        assert torch.cuda.is_available(), "CUDA not available -- refuse to run RMBG on CPU"
        from rmbg_v2 import RMBG2Runner
        _RUNNER = RMBG2Runner(device="cuda:0", model_id=str(Path(os.environ['AF_MODELS_ROOT']) / 'rmbg2'))
    return _RUNNER


def rmbg_soft(frame_bgr: np.ndarray) -> np.ndarray:
    """full-frame RMBG-2.0 soft mask (H,W) float32 in [0,1] -- one call, no background model."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return rmbg_runner().predict(rgb).astype(np.float32)


def flow_aware_background(frames: List[np.ndarray], occupied: List[np.ndarray]) -> np.ndarray:
    """per-pixel median over samples where the pixel was NOT occupied (dilated moving
    region around that time). Falls back to plain median where no free sample exists."""
    st = np.stack(frames).astype(np.float32)                    # S,H,W,3
    occ = np.stack(occupied)                                     # S,H,W bool
    free = ~occ
    cnt = free.sum(0)
    masked = np.where(free[..., None], st, np.nan)
    with np.errstate(all="ignore"):
        med = np.nanmedian(masked, axis=0)
    plain = np.median(st, axis=0)
    med = np.where((cnt >= 3)[..., None], med, plain)
    return np.nan_to_num(med).astype(np.uint8)


def bgsub(frame: np.ndarray, bg: np.ndarray, thr: int, min_area: int) -> np.ndarray:
    diff = cv2.absdiff(frame, bg).max(axis=2)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    m = (diff > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    nc, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(nc, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


def candidate_boxes(cand: np.ndarray, min_size: int, pad: float = 0.5) -> List[Tuple[int, int, int, int]]:
    H, W = cand.shape
    nc, lab, st, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    boxes = []
    for c in range(1, nc):
        x, y, w, h, a = st[c]
        px, py = int(w * pad), int(h * pad)
        x0, y0, x1, y1 = x - px, y - py, x + w + px, y + h + py
        # enforce minimum crop size (RMBG works badly on tiny crops)
        if x1 - x0 < min_size:
            cx = (x0 + x1) // 2; x0, x1 = cx - min_size // 2, cx + min_size // 2
        if y1 - y0 < min_size:
            cy = (y0 + y1) // 2; y0, y1 = cy - min_size // 2, cy + min_size // 2
        boxes.append([max(0, x0), max(0, y0), min(W, x1), min(H, y1)])
    # merge overlapping boxes
    merged = True
    while merged and len(boxes) > 1:
        merged = False
        out = []
        while boxes:
            b = boxes.pop()
            for i, o in enumerate(boxes):
                if not (b[2] < o[0] or o[2] < b[0] or b[3] < o[1] or o[3] < b[1]):
                    boxes[i] = [min(b[0], o[0]), min(b[1], o[1]), max(b[2], o[2]), max(b[3], o[3])]
                    merged = True
                    break
            else:
                out.append(b)
        boxes = out
    return [tuple(b) for b in boxes]


def rmbg_refine(frame_bgr: np.ndarray, cand: np.ndarray, min_crop: int, agree: float = 0.08,
                full_frame_first: bool = True) -> Tuple[np.ndarray, dict]:
    """Returns refined mask + receipt. Strategy:
    1. full-frame RMBG (cheap, 1 call). Components that overlap a candidate are accepted.
    2. candidates not covered by an accepted full-frame component -> ROI crop RMBG.
    3. candidates still not covered -> keep candidate pixels (fallback)."""
    H, W = cand.shape
    runner = rmbg_runner()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    out = np.zeros((H, W), bool)
    receipt = dict(full_frame_accepted=0, roi_calls=0, roi_accepted=0, fallback_components=0)
    nc, lab, st, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    cand_ids = list(range(1, nc))
    covered = set()

    def accept(soft: np.ndarray, region: np.ndarray, x0=0, y0=0):
        """accept RMBG components (soft>0.5) that agree with candidate region."""
        m = soft > 0.5
        n2, l2, s2, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        acc = np.zeros_like(m)
        k = 0
        for c in range(1, n2):
            comp = l2 == c
            ov = (comp & region).sum() / max(comp.sum(), 1)
            if ov >= agree:
                acc |= comp; k += 1
        return acc, k

    dil = cv2.dilate(cand.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(bool)
    if full_frame_first:
        soft = runner.predict(rgb)
        acc, k = accept(soft, dil)
        out |= acc
        receipt["full_frame_accepted"] = k
    for c in cand_ids:
        comp = lab == c
        if (comp & out).sum() >= 0.5 * comp.sum():
            covered.add(c)
    todo = [c for c in cand_ids if c not in covered]
    if todo:
        sub = np.isin(lab, todo)
        for (x0, y0, x1, y1) in candidate_boxes(sub, min_crop):
            crop = rgb[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            soft = runner.predict(np.ascontiguousarray(crop))
            receipt["roi_calls"] += 1
            region = dil[y0:y1, x0:x1]
            acc, k = accept(soft, region)
            receipt["roi_accepted"] += k
            out[y0:y1, x0:x1] |= acc
        for c in todo:
            comp = lab == c
            if (comp & out).sum() < 0.5 * comp.sum():
                out |= comp
                receipt["fallback_components"] += 1
    return out, receipt
