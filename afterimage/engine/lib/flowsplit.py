"""
Optical-flow based split + afterimage generation (dataset-agnostic).

Pipeline
  pass 1 (streaming)  : decode -> analysis-scale gray -> DIS optical flow (t-1 -> t)
                        per-frame signals: moving mask, motion area, motion flux,
                        mean velocity, *reversal score* (pixels re-traversed in the
                        opposite direction of the direction that last painted them)
  split decision      : active spans (motion vs static gap)
                        -> reversal split (turn-around),
                        -> budget split  (enough novel footprint painted = ~N poses),
                        -> hard time cap
  keyframe selection  : per segment, K frames at equal quantiles of cumulative
                        motion flux (arc-length sampling, not uniform time)
  pass 2 (streaming)  : grab keyframes + background samples (median background)
  compositing         : median background + bgsub masks of keyframes pasted
                        oldest->newest with an alpha ramp (latest opaque)

All thresholds that depend on object size are expressed in *object units*
(estimated from the moving-mask area), so the same config transfers across
resolutions / shooting distances.
"""
from __future__ import annotations
import json, math, os, time
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

cv2.setNumThreads(4)


# --------------------------------------------------------------------------- config
@dataclass
class Config:
    # geometry
    resize: Optional[Tuple[int, int]] = None   # (W,H) fixed experiment size, e.g. (512,512)
    resize_mode: str = "stretch"               # stretch | pad
    analysis_max_side: int = 384               # flow computed at this scale
    analysis_stride: int = 1                   # analyse every k-th frame (long videos: slow motion x k, cost / k)
    # flow / moving mask
    flow_backend: str = "dis"                  # dis (CPU) | waft (princeton-vl/WAFT, GPU0)
    flow_preset: str = "medium"                # DIS preset: fast|medium|ultrafast
    flow_noise_mult: float = 4.0               # moving if mag > max(min_flow, mult*median)
    min_flow_px: float = 0.3                   # px at analysis scale
    min_comp_frac: float = 0.00015             # min moving component area (frac of analysis px)
    chronic_frac: float = 0.25                 # pixel moving in > this fraction of frames = noise region
    chronic_min_sec: float = 20.0              # apply chronic exclusion only to videos at least this long
    long_video_sec: float = 20.0               # >= this: fixed-background gate + coherence filter (CCTV mode)
    bg_gate_thr: int = 20                      # analysis-scale |frame-bg| gate (long videos)
    coherence_min: float = 0.45                # |mean flow| / mean |flow| per component (long videos)
    sustain_sec: float = 0.3                   # motion must persist this long before a split / activity counts
    seg_min_poses_long: float = 1.2            # long videos: drop segments with less novel footprint than this (obj units)
    seg_min_comp_long: float = 0.4             # ... and whose largest moving component never reached this * obj_area
    comp_min_obj: float = 0.12                 # drop moving components smaller than this * obj_area
    # activity / spans
    active_area_frac: float = 0.0004           # motion area fraction to call a frame active
    static_gap_sec: float = 1.0                # pause longer than this closes the span
    min_span_sec: float = 0.3                  # spans shorter than this are dropped
    # reversal split
    reversal_cos: float = -0.5                 # opposite-direction criterion
    rev_hold: int = 3                          # frames an opposed pixel keeps its old direction before flipping
    dir_write_mult: float = 1.0                # |flow| >= thr*(1+mult) fully rewrites the direction memory; ~thr (jitter) does not
    reversal_thr: float = 0.25                 # fraction of moving flux that is reversing
    reversal_smooth: int = 3                   # frames (moving average)
    reversal_min_novel_obj: float = 0.25       # some new footprint (in object areas) must exist before a reversal split
    # budget split (object units)
    max_poses: float = 8.0                     # novel footprint / object area (pure translation: lenient)
    reentry_thr: float = 0.5                   # per-object: fraction of its pixels re-entering ground vacated >= reentry_gap_sec ago
    reentry_gap_sec: float = 0.4
    comp_obj_frac: float = 0.4                 # a component counts as "an object" if area >= this * obj_small
    split_sustain: int = 2                     # consecutive frames the per-object flag must hold
    contact_split: bool = False                # (off: fires when a person leaves the frame, wrong semantics)
    pause_split: bool = True                   # short clips: motion drops to < pause_frac of the segment peak -> phase boundary
    pause_frac: float = 0.3
    pause_sec: float = 0.3
    contact_sustain_sec: float = 0.25
    rev_blob_min: float = 0.20                 # turning blob must be >= this * reference object area to split
    rev_blob_obj: float = 0.5                  # a connected reversing blob >= this * obj_small also flags (merged objects)
    max_seg_sec: float = 12.0                  # hard cap
    min_seg_sec: float = 0.2                   # no split before this
    # keyframes
    k_min: int = 4
    k_max: int = 4
    # compositing
    bg_samples: int = 31                       # frames used for median background
    bg_thr: int = 28                           # |frame - bg| max-channel threshold
    alpha_min: float = 0.25
    alpha_max: float = 0.65
    feather: int = 2
    min_obj_frac: float = 0.0003               # bgsub component min area (frac of full px)
    draw_time_bar: bool = True
    mask_backend: str = "rmbg"                 # rmbg | bgsub
    rmbg_agree: float = 0.08                   # RMBG component accepted if >= this fraction of it touches the flow/bgsub candidate
    alpha_curve: str = "ref"                   # ref = [.22,.55,1] style ramp


DATASET_PRESETS = {
    "weizmann": dict(),
    "ut_interaction": dict(),
    "kisa": dict(resize=(512, 512), analysis_stride=3, pause_split=False, static_gap_sec=2.0, min_seg_sec=1.0, max_seg_sec=20.0,
                 max_poses=8.0, bg_samples=61, bg_thr=30),
}


def make_config(dataset: str, **over) -> Config:
    d = dict(DATASET_PRESETS.get(dataset, {}))
    d.update({k: v for k, v in over.items() if v is not None})
    return Config(**d)


# --------------------------------------------------------------------------- io
def _resize_frame(bgr: np.ndarray, cfg: Config) -> np.ndarray:
    if cfg.resize is None:
        return bgr
    W, H = cfg.resize
    if cfg.resize_mode == "stretch":
        return cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
    h, w = bgr.shape[:2]
    s = min(W / w, H / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    r = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((H, W, 3), np.uint8)
    y0, x0 = (H - nh) // 2, (W - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r
    return out


def iter_frames(path: str, cfg: Config, stride: int = 1):
    """yields (analysis_index, analysis_fps, frame). With stride k only every k-th
    source frame is decoded/yielded; analysis_index = source_index // k."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    i = 0
    while True:
        if stride > 1 and i % stride:
            if not cap.grab():
                break
            i += 1
            continue
        ok, bgr = cap.read()
        if not ok:
            break
        yield i // stride, fps / stride, _resize_frame(bgr, cfg)
        i += 1
    cap.release()


# --------------------------------------------------------------------------- pass 1
@dataclass
class Signals:
    fps: float
    n: int
    W: int
    H: int
    aw: int
    ah: int
    motion_area: np.ndarray      # frac of analysis px moving
    flux: np.ndarray             # sum |flow| over moving px (analysis px units)
    mean_vx: np.ndarray
    mean_vy: np.ndarray
    reversal: np.ndarray         # fraction of moving flux that reverses its own trail
    largest_comp: np.ndarray     # area (analysis px) of largest moving component
    moving_packed: np.ndarray    # packbits of moving masks (n, ah, ceil(aw/8))
    flow_thr: np.ndarray


def _dis(preset: str):
    p = {"ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
         "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
         "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM}[preset]
    d = cv2.DISOpticalFlow_create(p)
    d.setUseSpatialPropagation(True)
    return d


def analyze(path: str, cfg: Config, log=print) -> Signals:
    """pass 1: flow at analysis scale. Stores RAW moving masks and reversing masks;
    noise-aware refinement happens offline in refine()."""
    t0 = time.time()
    dis = _dis(cfg.flow_preset)
    waft = None
    if cfg.flow_backend == "waft":
        import flow_waft
        global _WAFT
        try:
            _WAFT
        except NameError:
            _WAFT = flow_waft.WAFTFlow(device="cuda:0")
        waft = _WAFT
    cap = cv2.VideoCapture(path); n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps0 = cap.get(cv2.CAP_PROP_FPS) or 25.0; cap.release()
    long_video = n_total >= cfg.long_video_sec * fps0
    bg_small = None
    if long_video:
        # CCTV mode: fixed clean background (plain median of sparse samples) gates the flow mask
        n_an = n_total // cfg.analysis_stride
        idx = set(np.linspace(0, n_an - 1, min(cfg.bg_samples, n_an)).round().astype(int).tolist())
        samples = [cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None, fx=min(1.0, cfg.analysis_max_side / max(b.shape[:2])),
                              fy=min(1.0, cfg.analysis_max_side / max(b.shape[:2])), interpolation=cv2.INTER_AREA)
                   for i, _, b in iter_frames(path, cfg, cfg.analysis_stride) if i in idx]
        bg_small = np.median(np.stack(samples), axis=0).astype(np.uint8)
        log(f"  long video ({n_total/fps0:.0f}s): fixed background gate + coherence filter on")
    prev = None
    rows, packed, packed_rev = [], [], []
    dir_x = dir_y = last_t = None
    fps = 25.0
    W = H = aw = ah = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for i, fps, bgr in iter_frames(path, cfg, cfg.analysis_stride):
        if i == 0:
            H, W = bgr.shape[:2]
            s = min(1.0, cfg.analysis_max_side / max(W, H))
            aw, ah = int(round(W * s)), int(round(H * s))
            dir_x = np.zeros((ah, aw), np.float32)
            dir_y = np.zeros((ah, aw), np.float32)
            last_t = np.full((ah, aw), -1, np.int32)
            opp_cnt = np.zeros((ah, aw), np.int16)
            chronic = np.zeros((ah, aw), np.float32)
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if (aw, ah) != (W, H):
            g = cv2.resize(g, (aw, ah), interpolation=cv2.INTER_AREA)
            small = cv2.resize(bgr, (aw, ah), interpolation=cv2.INTER_AREA) if waft is not None else None
        else:
            small = bgr if waft is not None else None
        if prev is None:
            z = np.zeros((ah, aw), bool)
            rows.append((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            packed.append(np.packbits(z, axis=1)); packed_rev.append(np.packbits(z, axis=1))
            prev = g; prev_small = small
            continue
        flow = waft.calc(prev_small, small) if waft is not None else dis.calc(prev, g, None)
        fx, fy = flow[..., 0], flow[..., 1]
        mag = np.sqrt(fx * fx + fy * fy)
        thr = max(cfg.min_flow_px, cfg.flow_noise_mult * float(np.median(mag)))
        moving = (mag > thr).astype(np.uint8)
        moving = cv2.morphologyEx(moving, cv2.MORPH_OPEN, kernel)
        if bg_small is not None:
            gate = (cv2.absdiff(g, bg_small) > cfg.bg_gate_thr).astype(np.uint8)
            gate = cv2.dilate(gate, kernel)
            moving &= gate
            # coherence: keep components whose flow points one way (objects), drop shimmer/rain
            nc, lab, st, _ = cv2.connectedComponentsWithStats(moving, connectivity=8)
            if nc > 1:
                sx = np.bincount(lab.ravel(), weights=fx.ravel(), minlength=nc)
                sy = np.bincount(lab.ravel(), weights=fy.ravel(), minlength=nc)
                sm = np.bincount(lab.ravel(), weights=mag.ravel(), minlength=nc)
                coh = np.hypot(sx, sy) / np.maximum(sm, 1e-6)
                keep = coh >= cfg.coherence_min; keep[0] = False
                moving = keep[lab].astype(np.uint8)
        m = moving.astype(bool)
        rev = np.zeros((ah, aw), bool)
        if m.any():
            fm, fxm, fym = mag[m], fx[m], fy[m]
            flux = float(fm.sum())
            vx, vy = float(fxm.mean()), float(fym.mean())
            ux, uy = fxm / (fm + 1e-6), fym / (fm + 1e-6)
            seen = last_t[m] >= 0
            dxm, dym = dir_x[m], dir_y[m]
            dn = np.sqrt(dxm * dxm + dym * dym) + 1e-6
            cosv = (ux * dxm + uy * dym) / dn
            r = (cosv < cfg.reversal_cos) & seen & (dn > 0.5)
            rev[m] = r
            rev_frac = float(fm[r].sum() / (flux + 1e-6))
            # direction memory: magnitude-gated EMA so that jitter (|flow| ~ thr) does not
            # overwrite the direction a real displacement wrote (e.g. bodies at rest mid-handshake)
            w = np.clip((fm - thr) / (cfg.dir_write_mult * thr), 0.0, 1.0)
            # opposed pixels keep their old direction for rev_hold frames (so a slow object walking
            # back over its own footprint produces a clear reversal pulse), then flip.
            oc = opp_cnt[m]
            oc = np.where(r, oc + 1, 0)
            allow = (~r) | (oc >= cfg.rev_hold)
            w = np.where(allow, w, 0.0)
            oc = np.where(oc >= cfg.rev_hold, 0, oc)
            opp_cnt[m] = oc
            dir_x[m] = (1 - w) * dxm + w * ux
            dir_y[m] = (1 - w) * dym + w * uy
            last_t[m] = np.where(w > 0.3, i, last_t[m])
            last_t[m] = np.where(last_t[m] < 0, i, last_t[m]) if False else last_t[m]
        else:
            flux = vx = vy = rev_frac = 0.0
        rows.append((float(m.mean()), flux, vx, vy, rev_frac, 0.0, thr))
        packed.append(np.packbits(m, axis=1)); packed_rev.append(np.packbits(rev, axis=1))
        prev = g; prev_small = small
        if i % 1000 == 0 and i:
            log(f"  flow {i} frames  {time.time()-t0:.1f}s")
    a = np.array(rows, np.float64)
    sig = Signals(fps=float(fps), n=len(rows), W=W, H=H, aw=aw, ah=ah,
                  motion_area=a[:, 0], flux=a[:, 1], mean_vx=a[:, 2], mean_vy=a[:, 3],
                  reversal=a[:, 4], largest_comp=a[:, 5], flow_thr=a[:, 6],
                  moving_packed=np.stack(packed))
    sig.rev_packed = np.stack(packed_rev)
    sig.stride = cfg.analysis_stride
    sig.src_fps = fps * cfg.analysis_stride
    sig.long_video = long_video
    refine(sig, cfg, log)
    return sig


def _unpack_any(packed, i, aw):
    return np.unpackbits(packed[i], axis=1)[:, :aw].astype(bool)


def refine(sig: Signals, cfg: Config, log=print):
    """Video-adaptive noise handling on the stored raw masks (no video re-read):
    1. chronic-motion map: pixels moving in > chronic_frac of frames (leaves, water,
       flicker) are excluded  -> per-video automatic noise mask
    2. temporal persistence: a moving pixel must also be moving (dilated) in the
       previous or next frame  -> kills 1-frame flicker / rain
    3. object-scale component filter: components < comp_min_obj * obj_area are
       dropped, obj_area estimated robustly from the persistent masks
    Recomputes motion_area, largest_comp, reversal (area-based) on refined masks."""
    n, aw, ah = sig.n, sig.aw, sig.ah
    raw = np.stack([_unpack_any(sig.moving_packed, i, aw) for i in range(n)])
    chronic = raw.mean(0)
    sig.chronic = chronic
    # chronic-noise exclusion only makes sense for long videos; in a 3 s clip the
    # object itself is "chronic".  Below chronic_min_sec no pixel is excluded.
    noise = (chronic > cfg.chronic_frac) if n >= cfg.chronic_min_sec * sig.fps else np.zeros_like(chronic, bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dil = np.stack([cv2.dilate(r.astype(np.uint8), k).astype(bool) for r in raw])
    persist = raw.copy()
    persist[1:] &= (dil[:-1] | np.concatenate([dil[2:], dil[-1:]]) if n > 2 else dil[:-1])
    persist &= ~noise
    # object scale from persistent masks (first estimate: largest component per frame)
    min_px = max(4, int(cfg.min_comp_frac * aw * ah))
    largest = np.zeros(n)
    comps = []
    for i in range(n):
        nc, lab, st, _ = cv2.connectedComponentsWithStats(persist[i].astype(np.uint8), connectivity=8)
        areas = st[1:, cv2.CC_STAT_AREA] if nc > 1 else np.zeros(0, int)
        comps.append((lab, areas))
        largest[i] = areas.max() if areas.size else 0
    act = largest >= min_px
    # single-object scale: median area of ALL persistent components (>= 4*min_px) over active frames.
    # Robust to two people merging into one blob (largest-only would double the size) and to
    # car + person scenes (largest-only would pick the car).
    all_areas = np.concatenate([areas[areas >= 4 * min_px] for (_, areas) in comps if areas.size]) if any(a.size for _, a in comps) else np.zeros(0)
    if act.sum() >= 3 and all_areas.size >= 3:
        obj_area = float(np.percentile(largest[act], 75))      # typical large object (budget scale)
        obj_small = float(np.median(all_areas))
    else:
        obj_area = obj_small = max(float(min_px * 4), 16.0)
    obj_small = max(obj_small, float(min_px * 2))
    comp_min = max(min_px, int(cfg.comp_min_obj * obj_small))
    refined = np.zeros_like(persist)
    for i in range(n):
        lab, areas = comps[i]
        if areas.size == 0:
            continue
        keep = np.zeros(len(areas) + 1, bool)
        keep[1:] = areas >= comp_min
        refined[i] = keep[lab]
        largest[i] = areas[areas >= comp_min].max() if keep[1:].any() else 0
    rev = np.stack([_unpack_any(sig.rev_packed, i, aw) for i in range(n)]) & refined
    sig.motion_area = refined.reshape(n, -1).mean(1)
    sig.largest_comp = largest
    cnt = refined.reshape(n, -1).sum(1)
    sig.reversal = rev.reshape(n, -1).sum(1) / np.maximum(cnt, 1)
    sig.moving_packed = np.stack([np.packbits(refined[i], axis=1) for i in range(n)])
    sig.obj_area = obj_area
    sig.obj_small = obj_small
    sig.noise_frac = float(noise.mean())
    # sustained activity: rolling median over sustain_sec kills impulsive 1-2 frame noise
    w = max(1, int(round(cfg.sustain_sec * sig.fps)))
    if w > 1:
        from scipy.ndimage import median_filter
        sig.motion_area_sus = median_filter(sig.motion_area, size=w, mode="nearest")
    else:
        sig.motion_area_sus = sig.motion_area.copy()
    log(f"  refine: obj_area={obj_area:.0f}px obj_small={obj_small:.0f}px comp_min={comp_min} chronic-noise={noise.mean()*100:.1f}% "
        f"active-frames={(sig.motion_area > cfg.active_area_frac).mean()*100:.0f}%")


def unpack(sig: Signals, i: int) -> np.ndarray:
    return np.unpackbits(sig.moving_packed[i], axis=1)[:, :sig.aw].astype(bool)


# --------------------------------------------------------------------------- split decision
@dataclass
class Segment:
    idx: int
    start: int
    end: int                      # inclusive
    reason_end: str
    poses_est: float
    keyframes: List[int] = field(default_factory=list)
    obj_area: float = 0.0
    evidence: str = "strong"
    t_start: float = 0.0
    t_end: float = 0.0


def _smooth(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def active_spans(sig: Signals, cfg: Config) -> List[Tuple[int, int]]:
    fps = sig.fps
    act = getattr(sig, "motion_area_sus", sig.motion_area) > cfg.active_area_frac
    gap = max(1, int(round(cfg.static_gap_sec * fps)))
    spans, i, n = [], 0, sig.n
    while i < n:
        if not act[i]:
            i += 1
            continue
        s = i
        last = i
        j = i + 1
        while j < n and (j - last) <= gap:
            if act[j]:
                last = j
            j += 1
        spans.append((s, last))
        i = j
    min_len = max(2, int(cfg.min_span_sec * fps))
    kept = [sp for sp in spans if sp[1] - sp[0] + 1 >= min_len]
    if not kept and spans:
        kept = [max(spans, key=lambda s: s[1] - s[0])]
    if not kept:
        kept = [(0, n - 1)]
    return kept


def object_scale(sig: Signals, a: int, b: int, cfg: Config) -> float:
    """object area estimate (analysis px): median largest moving component over active frames."""
    return float(getattr(sig, "obj_area", 0) or max(16.0, cfg.min_comp_frac * sig.aw * sig.ah * 4))


def split_segments(sig: Signals, cfg: Config) -> List[Segment]:
    """Per-object split decision (offline, on the refined masks):
      reversal : some object-sized component has >= reversal_thr of its pixels moving against
                 the direction that last painted them (turn-around, arm up->down, approach->retreat)
      reentry  : some object-sized component has >= reentry_thr of its pixels on ground it vacated
                 >= reentry_gap_sec ago within this segment (coming back over its own trail, circling)
      budget   : novel footprint / object area >= max_poses (translation: lenient)
      timecap  : safety cap.  Flags must hold split_sustain consecutive frames."""
    fps = sig.fps
    min_seg = int(round(cfg.min_seg_sec * fps))
    max_seg = int(round(cfg.max_seg_sec * fps))
    gap = max(1, int(round(cfg.reentry_gap_sec * fps)))
    obj_small = float(getattr(sig, "obj_small", 16.0))
    comp_obj = max(4.0, cfg.comp_obj_frac * obj_small)
    # reference object scale for the moderation rule: short single-action clips -> typical largest mover,
    # long CCTV videos (cars + people) -> typical single object
    ref_obj = obj_small if getattr(sig, "long_video", False) else float(getattr(sig, "obj_area", obj_small))
    sus = getattr(sig, "motion_area_sus", sig.motion_area)
    rev_trace = np.zeros(sig.n); reentry_trace = np.zeros(sig.n); blob_trace = np.zeros(sig.n)
    segs: List[Segment] = []
    for (a, b) in active_spans(sig, cfg):
        start = a
        painted = np.zeros((sig.ah, sig.aw), bool)
        last_paint = np.full((sig.ah, sig.aw), -1, np.int32)
        novel = 0.0
        hold = 0; hold_reason = None
        lc_hist = []; prev_blob_re = 0; nobj_hist = []; contact_done = False; seg_peak = 0.0
        t = a
        while t <= b:
            m = unpack(sig, t)
            r = _unpack_any(sig.rev_packed, t, sig.aw) & m
            # per-object flags
            flag = None
            n_obj = 0
            if m.any():
                nc, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
                vac = (last_paint >= 0) & ((t - last_paint) >= gap)
                best_rev = best_re = 0.0
                n_obj = int(sum(1 for c in range(1, nc) if st[c, cv2.CC_STAT_AREA] >= max(comp_obj, 0.3 * ref_obj)))
                for c in range(1, nc):
                    area = st[c, cv2.CC_STAT_AREA]
                    if area < comp_obj:
                        continue
                    comp = lab == c
                    rv = float((r & comp).sum() / area); re = float((vac & comp).sum() / area)
                    best_rev = max(best_rev, rv); best_re = max(best_re, re)
                # merged objects (contact): a body-sized connected blob of reversing / re-entering
                # pixels is an object turning around even if the merged component's fraction is diluted
                def _blob(mask):
                    if not mask.any():
                        return 0
                    mm = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                    n2, _, st2, _ = cv2.connectedComponentsWithStats(mm, connectivity=8)
                    return int(st2[1:, cv2.CC_STAT_AREA].max()) if n2 > 1 else 0
                blob_rev = _blob(r); blob_re = _blob(vac & m); blob_trace[t] = blob_rev
                best_rev = max(best_rev, min(1.0, blob_rev / (cfg.rev_blob_obj * obj_small)) * cfg.reversal_thr)
                best_re = max(best_re, min(1.0, blob_re / (cfg.rev_blob_obj * obj_small)) * cfg.reentry_thr)
                rev_trace[t], reentry_trace[t] = best_rev, best_re
                # moderation: the turning part must be a substantial piece of an object
                # (hands shaking / hugging arms: ~0.1 of the object; jack arms >= 0.4; a body retreating >= 0.2)
                # (max over this and the previous frame: the turning blob peaks for ~1 frame)
                big_enough = max(blob_rev, blob_trace[t - 1] if t > 0 else 0) >= cfg.rev_blob_min * ref_obj
                if best_rev >= cfg.reversal_thr and big_enough:
                    flag = "reversal"
                elif best_re >= cfg.reentry_thr and max(blob_re, prev_blob_re) >= cfg.rev_blob_min * ref_obj:
                    flag = "reentry"
                prev_blob_re = blob_re
            # contact: >=2 object-sized movers for a while, then 1 for a while  -> "approach" ends here
            nobj_hist.append(n_obj)
            cs = max(2, int(round(cfg.contact_sustain_sec * fps)))
            if (cfg.contact_split and flag is None and len(nobj_hist) >= 2 * cs
                    and min(nobj_hist[-2 * cs:-cs]) >= 2 and max(nobj_hist[-cs:]) == 1 and not contact_done):
                flag = "contact"; contact_done = True
            # pause: approach -> (hug / handshake / stand) ; the low-motion phase starts a new segment
            ps = max(2, int(round(cfg.pause_sec * fps)))
            seg_peak = max(seg_peak, float(sus[t]))
            if (cfg.pause_split and flag is None and t - start >= ps and seg_peak > 2 * cfg.active_area_frac
                    and (sus[t - ps + 1:t + 1] < cfg.pause_frac * seg_peak).all()
                    and (sus[max(start, t - 2 * ps + 1):t - ps + 1] >= cfg.pause_frac * seg_peak).any()):
                flag = "pause"
            novel += float((m & ~painted).sum())
            painted |= m
            last_paint[m] = t
            if sig.largest_comp[t] > 0:
                lc_hist.append(sig.largest_comp[t])
            if getattr(sig, "long_video", False):   # CCTV: cars and people share the scene -> per-segment object size
                obj_seg = float(np.median(lc_hist)) if lc_hist else max(obj_small, 16.0)
            else:                                    # single-action clip -> the video's reference object
                obj_seg = ref_obj
            poses = novel / obj_seg
            length = t - start + 1
            if flag is not None and sus[t] > cfg.active_area_frac:
                hold = hold + 1 if hold_reason == flag else 1
                hold_reason = flag
            else:
                hold = 0; hold_reason = None
            reason = None
            if length >= min_seg:
                if hold_reason in ("contact", "pause") and poses >= cfg.reversal_min_novel_obj:
                    reason = hold_reason
                elif hold >= cfg.split_sustain and poses >= cfg.reversal_min_novel_obj:
                    reason = hold_reason
                elif poses >= cfg.max_poses:
                    reason = "budget"
                elif length >= max_seg:
                    reason = "timecap"
            if reason is not None and t < b:
                back = cfg.split_sustain - 1 if reason in ("reversal", "reentry") else (cs - 1 if reason == "contact" else (ps - 1 if reason == "pause" else -1))
                cut = max(start, t - back)             # new segment starts at `cut`
                segs.append(Segment(len(segs), start, cut - 1 if cut > start else t, reason, poses, obj_area=obj_seg))
                start = cut if cut > start else t + 1
                painted[:] = False; last_paint[:] = -1; novel = 0.0; lc_hist = []; hold = 0; hold_reason = None; nobj_hist = []; seg_peak = 0.0
                for tt in range(start, t + 1):        # re-add frames that moved into the new segment
                    mm = unpack(sig, tt); novel += float((mm & ~painted).sum()); painted |= mm; last_paint[mm] = tt
                    if sig.largest_comp[tt] > 0: lc_hist.append(sig.largest_comp[tt])
            t += 1
        if start <= b:
            obj_seg = (float(np.median(lc_hist)) if lc_hist else max(obj_small, 16.0)) if getattr(sig, "long_video", False) else ref_obj
            segs.append(Segment(len(segs), start, b, "span_end", novel / obj_seg, obj_area=obj_seg))
    sig.rev_trace, sig.reentry_trace, sig.blob_trace = rev_trace, reentry_trace, blob_trace
    # merge too-short tails into previous segment
    merged: List[Segment] = []
    for s_ in segs:
        if merged and (s_.end - s_.start + 1) < min_seg and merged[-1].end + 1 == s_.start:
            p = merged[-1]
            p.end, p.reason_end, p.poses_est = s_.end, s_.reason_end, p.poses_est + s_.poses_est
        elif merged and (merged[-1].end - merged[-1].start + 1) < min_seg and merged[-1].end + 1 == s_.start:
            p = merged.pop()                       # short head segment -> merge forward
            s_.start, s_.poses_est = p.start, p.poses_est + s_.poses_est
            merged.append(s_)
        else:
            merged.append(s_)
    if getattr(sig, "long_video", False) and len(merged) > 1:
        # CCTV mode: a segment must contain object-like evidence (leaves/rain never do).
        # strong: big enough vs the typical large object.  weak: only vs the small-object scale ->
        # kept only if RMBG confirms an object on one of its keyframes (checked in process_video).
        obj_big = float(getattr(sig, "obj_area", obj_small))
        for x in merged:
            mc = sig.largest_comp[x.start:x.end + 1].max()
            x.evidence = ("strong" if (x.poses_est >= cfg.seg_min_poses_long and mc >= cfg.seg_min_comp_long * obj_big)
                          else "weak" if mc >= cfg.seg_min_comp_long * obj_small else "none")
        kept = [x for x in merged if x.evidence != "none"]
        merged = kept or merged
    for k, x in enumerate(merged):
        x.idx = k
        x.t_start, x.t_end = x.start / fps, (x.end + 1) / fps
    return merged


def select_keyframes(sig: Signals, seg: Segment, cfg: Config) -> List[int]:
    a, b = seg.start, seg.end
    flux = sig.flux[a:b + 1].copy()
    K = int(np.clip(math.ceil(seg.poses_est) + 1, cfg.k_min, cfg.k_max))
    K = min(K, b - a + 1)
    if K <= 1:
        return [b]
    c = np.cumsum(flux)
    if c[-1] <= 0:
        idx = np.linspace(a, b, K).round().astype(int)
        return sorted(set(idx.tolist()))
    c = c / c[-1]
    targets = np.linspace(0, 1, K)
    picks = []
    for q in targets:
        j = int(np.searchsorted(c, q, side="left"))
        j = min(max(j, 0), b - a)
        picks.append(a + j)
    # first pick: first frame with motion in the segment (not a frozen frame before motion)
    act = np.nonzero(sig.motion_area[a:b + 1] > cfg.active_area_frac)[0]
    if act.size:
        picks[0] = a + int(act[0])
    picks[-1] = b
    picks = sorted(set(picks))
    return picks


# --------------------------------------------------------------------------- pass 2 + compositing
def grab_frames(path: str, cfg: Config, wanted: set, bg_idx: List[int]):
    need = set(wanted) | set(bg_idx)
    out, bg = {}, []
    for i, fps, bgr in iter_frames(path, cfg, cfg.analysis_stride):
        if i in need:
            if i in wanted:
                out[i] = bgr
            if i in bg_idx:
                bg.append(bgr)
    return out, bg


def median_background(bg_frames: List[np.ndarray]) -> np.ndarray:
    st = np.stack(bg_frames).astype(np.uint8)
    return np.median(st, axis=0).astype(np.uint8)


def bgsub_mask(frame: np.ndarray, bg: np.ndarray, cfg: Config,
               keep_region: Optional[np.ndarray] = None) -> np.ndarray:
    diff = cv2.absdiff(frame, bg).max(axis=2)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    m = (diff > cfg.bg_thr).astype(np.uint8)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k7)
    ncomp, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    H, W = m.shape
    min_area = max(8, int(cfg.min_obj_frac * H * W))
    keep = np.zeros(ncomp, bool)
    for c in range(1, ncomp):
        if stats[c, cv2.CC_STAT_AREA] < min_area:
            continue
        if keep_region is not None:
            comp = lab == c
            if not (comp & keep_region).any():
                continue
        keep[c] = True
    return keep[lab]


def _up(mask_small: np.ndarray, W: int, H: int, dilate: int = 9) -> np.ndarray:
    m = cv2.resize(mask_small.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    if dilate:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate)))
    return m.astype(bool)






# --------------------------------------------------------------------------- driver




def draw_time_bar(img: np.ndarray, times: List[float]) -> np.ndarray:
    H, W = img.shape[:2]
    bar_h = max(10, int(0.045 * H))
    canvas = np.zeros((H + bar_h, W, 3), np.uint8)
    canvas[:H] = img
    canvas[H:] = (32, 32, 32)
    K = len(times)
    t0, t1 = times[0], times[-1]
    span = max(t1 - t0, 1e-6)
    pad = int(0.03 * W)
    for i, t in enumerate(times):
        x = pad + int((W - 2 * pad) * (t - t0) / span)
        a = 0.3 + 0.7 * (i / max(K - 1, 1))
        col = (int(255 * a), int(255 * a), int(255 * a))
        cv2.circle(canvas, (x, H + bar_h // 2), max(2, bar_h // 4), col, -1)
    cv2.line(canvas, (pad, H + bar_h // 2), (W - pad, H + bar_h // 2), (120, 120, 120), 1)
    scale = max(0.3, bar_h / 32)
    cv2.putText(canvas, f"{t0:.1f}s", (2, H + bar_h - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (200, 200, 200), 1, cv2.LINE_AA)
    txt = f"{t1:.1f}s"
    (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.putText(canvas, txt, (W - tw - 2, H + bar_h - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


# --------------------------------------------------------------------------- driver
def process_video(path: str, out_dir: str, cfg: Config, log=print) -> dict:
    import masks as MK
    os.makedirs(out_dir, exist_ok=True)
    for old in os.listdir(out_dir):                      # stale outputs from a previous run
        if old.startswith("seg") or old in ("segments.json", "signals.png", "signals.npz", "index.html", "background.png"):
            os.remove(os.path.join(out_dir, old))
    t0 = time.time()
    log(f"[{os.path.basename(out_dir)}] pass1 flow ...")
    sig = analyze(path, cfg, log)
    segs = split_segments(sig, cfg)
    for s in segs:
        s.keyframes = select_keyframes(sig, s, cfg)
    t_flow = time.time() - t0
    log(f"  frames={sig.n} fps={sig.fps:.2f} size={sig.W}x{sig.H} analysis={sig.aw}x{sig.ah} "
        f"segments={len(segs)}  ({t_flow:.1f}s)")
    wanted = {k for s in segs for k in s.keyframes}
    bg_idx = sorted(set(np.linspace(0, sig.n - 1, min(cfg.bg_samples, sig.n)).round().astype(int).tolist()))
    frames, bg_frames = grab_frames(path, cfg, wanted, bg_idx)
    # flow-aware background: a sample pixel is "occupied" if it moved within +-win frames
    win = max(2, int(0.5 * sig.fps))
    occ = []
    kd = max(5, int(0.02 * max(sig.W, sig.H)))
    for i in bg_idx:
        u = np.zeros((sig.ah, sig.aw), bool)
        for t in range(max(0, i - win), min(sig.n, i + win + 1)):
            u |= unpack(sig, t)
        occ.append(_up(u, sig.W, sig.H, dilate=kd))
    bg = MK.flow_aware_background(bg_frames, occ)
    cv2.imwrite(os.path.join(out_dir, "background.png"), bg)
    scale2 = (sig.W * sig.H) / (sig.aw * sig.ah)
    obj_area_full = getattr(sig, "obj_small", 100.0) * scale2
    min_area_full = max(8, int(cfg.min_obj_frac * sig.W * sig.H))
    min_crop = int(np.clip(4 * math.sqrt(obj_area_full), 96, max(sig.W, sig.H)))
    results = []
    t_mask = 0.0
    for s in segs:
        union = np.zeros((sig.ah, sig.aw), bool)
        for t in range(s.start, s.end + 1):
            union |= unpack(sig, t)
        region = _up(union, sig.W, sig.H, dilate=kd)
        fr, ms, receipts = [], [], []
        for j, k in enumerate(s.keyframes):
            f = frames[k]
            last = j == len(s.keyframes) - 1
            tm = time.time()
            if not getattr(sig, "long_video", False) and cfg.mask_backend == "rmbg":
                # Weizmann / UT: no background estimation. RMBG soft mask of the keyframe itself
                # (components14 style); older ghosts keep only components that moved in this segment.
                soft = MK.rmbg_soft(f)
                m = soft.copy()
                if not last:
                    core = (soft >= 0.5).astype(np.uint8)
                    nc, lab, st, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
                    keep = np.zeros(nc, bool)
                    for c in range(1, nc):
                        comp = lab == c
                        keep[c] = (comp & region).sum() >= 0.05 * st[c, cv2.CC_STAT_AREA]
                    keepmask = cv2.dilate(keep[lab].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                    m = np.where(keepmask, soft, 0.0).astype(np.float32)
                rc = dict(backend="rmbg_soft", core_px=int((m >= 0.5).sum()))
            else:
                cand = MK.bgsub(f, bg, cfg.bg_thr, min_area_full)
                mv = _up(unpack(sig, k) | unpack(sig, max(0, k - 1)), sig.W, sig.H, dilate=3)
                cand = cand | mv
                if cfg.mask_backend == "rmbg" and cand.any():
                    m, rc = MK.rmbg_refine(f, cand, min_crop, agree=cfg.rmbg_agree)
                else:
                    m, rc = cand, dict(backend="bgsub")
                if not last:
                    nc, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
                    keep = np.zeros(nc, bool)
                    for c in range(1, nc):
                        comp = lab == c
                        keep[c] = (comp & region).sum() >= 0.05 * st[c, cv2.CC_STAT_AREA]
                    m = keep[lab]
                m = m.astype(np.float32)
            t_mask += time.time() - tm
            fr.append(f); ms.append(m); receipts.append(rc)
        if s.evidence == "weak" and not any((rc.get("full_frame_accepted", 0) + rc.get("roi_accepted", 0)) > 0 for rc in receipts):
            log(f"  drop seg {s.idx} {s.t_start:.1f}-{s.t_end:.1f}s: weak flow evidence and no RMBG object")
            continue
        times = [k / sig.fps for k in s.keyframes]
        img = compose_ref(fr, ms, cfg, times)
        name = f"seg{s.idx:02d}_{s.t_start:06.2f}-{s.t_end:06.2f}s.png"
        cv2.imwrite(os.path.join(out_dir, name), img)
        sheet = keyframe_sheet(fr, ms, times)
        cv2.imwrite(os.path.join(out_dir, f"seg{s.idx:02d}_keyframes.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
        results.append(dict(idx=s.idx, start=s.start, end=s.end, evidence=s.evidence, t_start=round(s.t_start, 3), t_end=round(s.t_end, 3),
                            reason_end=s.reason_end, poses_est=round(s.poses_est, 2), obj_area_px=round(s.obj_area, 1),
                            keyframes=s.keyframes, keyframes_src=[k * sig.stride for k in s.keyframes],
                            start_src=s.start * sig.stride, end_src=s.end * sig.stride,
                            keyframe_times=[round(t, 3) for t in times],
                            mask_px=[int((m >= 0.5).sum()) for m in ms], mask_receipts=receipts,
                            image=name, sheet=f"seg{s.idx:02d}_keyframes.jpg"))
    np.savez_compressed(os.path.join(out_dir, "signals.npz"), motion_area=sig.motion_area, flux=sig.flux,
                        mean_vx=sig.mean_vx, mean_vy=sig.mean_vy, reversal=sig.reversal,
                        largest_comp=sig.largest_comp, flow_thr=sig.flow_thr, chronic=sig.chronic)
    meta = dict(video=path, fps=sig.fps, src_fps=sig.src_fps, analysis_stride=sig.stride, n_frames=sig.n,
                n_src_frames=sig.n * sig.stride, width=sig.W, height=sig.H,
                analysis=[sig.aw, sig.ah], obj_area_analysis_px=round(float(sig.obj_area), 1),
                chronic_noise_frac=round(float(sig.noise_frac), 4),
                config=asdict(cfg), segments=results, n_keyframes=len(wanted),
                elapsed_sec=round(time.time() - t0, 1), flow_sec=round(t_flow, 1), mask_sec=round(t_mask, 1))
    with open(os.path.join(out_dir, "segments.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    plot_signals(sig, segs, cfg, os.path.join(out_dir, "signals.png"))
    log(f"  done {len(segs)} segments, {len(wanted)} keyframes, flow {t_flow:.1f}s mask {t_mask:.1f}s total {time.time()-t0:.1f}s")
    return meta


def compose_ref(frames: List[np.ndarray], masks: List[np.ndarray], cfg: Config,
                times: Optional[List[float]] = None) -> np.ndarray:
    """reference-style: base = last original frame (full context), older poses pasted
    oldest->newest with alpha ramp, the latest object re-pasted opaque on top."""
    K = len(frames)
    base = frames[-1].astype(np.float32).copy()
    if K > 1:
        alphas = np.linspace(0.35, 0.7, K - 1) if K > 2 else np.array([0.5])
        if cfg.alpha_curve != "ref":
            alphas = np.linspace(cfg.alpha_min, 0.8, K - 1)
        for f, m, a in zip(frames[:-1], masks[:-1], alphas):
            mf = m.astype(np.float32)
            if cfg.feather > 0:
                mf = cv2.GaussianBlur(mf, (0, 0), cfg.feather)
            al = (mf * a)[..., None]
            base = base * (1 - al) + f.astype(np.float32) * al
        # latest object opaque
        mf = masks[-1].astype(np.float32)
        if cfg.feather > 0:
            mf = cv2.GaussianBlur(mf, (0, 0), cfg.feather)
        al = mf[..., None]
        base = base * (1 - al) + frames[-1].astype(np.float32) * al
    img = np.clip(base, 0, 255).astype(np.uint8)
    if cfg.draw_time_bar and times:
        img = draw_time_bar(img, times)
    return img


def keyframe_sheet(frames, masks, times, cell_w=240):
    cells = []
    for f, m, t in zip(frames, masks, times):
        h, w = f.shape[:2]
        s = cell_w / w
        fr = cv2.resize(f, (cell_w, int(h * s)))
        mk = cv2.resize((m * 255).astype(np.uint8), (cell_w, int(h * s)), interpolation=cv2.INTER_NEAREST)
        ov = fr.copy(); ov[mk > 0] = (0.5 * ov[mk > 0] + 0.5 * np.array([0, 200, 255])).astype(np.uint8)
        cell = np.concatenate([fr, ov], axis=0)
        cv2.putText(cell, f"{t:.2f}s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cells.append(cell)
    return np.concatenate(cells, axis=1)


def plot_signals(sig: Signals, segs: List[Segment], cfg: Config, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(sig.n) / sig.fps
    fig, ax = plt.subplots(3, 1, figsize=(max(8, min(30, sig.n / sig.fps * 0.6)), 6), sharex=True)
    ax[0].plot(t, sig.motion_area, lw=0.8, color="#2a6f97"); ax[0].set_ylabel("motion area")
    ax[0].axhline(cfg.active_area_frac, color="gray", ls=":", lw=0.7)
    ax[1].plot(t, getattr(sig, "rev_trace", sig.reversal), lw=0.8, color="#c0392b", label="reversal (per object)")
    ax[1].plot(t, getattr(sig, "reentry_trace", np.zeros(sig.n)), lw=0.8, color="#2980b9", label="re-entry (per object)")
    ax[1].axhline(cfg.reversal_thr, color="gray", ls=":", lw=0.7); ax[1].set_ylabel("split flags"); ax[1].legend(loc="upper right", fontsize=7)
    ax[2].plot(t, sig.flux, lw=0.8, color="#3a7d44"); ax[2].set_ylabel("flux"); ax[2].set_xlabel("time (s)")
    colors = {"reversal": "#c0392b", "reentry": "#2980b9", "contact": "#16a085", "pause": "#27ae60", "budget": "#f39c12", "timecap": "#8e44ad", "span_end": "#7f8c8d"}
    for s in segs:
        for a in ax:
            a.axvspan(s.t_start, s.t_end, color="#000", alpha=0.04)
            a.axvline(s.t_end, color=colors.get(s.reason_end, "k"), lw=1.0)
        for k in s.keyframes:
            ax[2].plot(k / sig.fps, sig.flux[k], "o", ms=3, color="#1a1a1a")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
