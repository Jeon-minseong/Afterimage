def process_video(path: str, out_dir: str, cfg: Config, log=print) -> dict:
    import masks as MK
    os.makedirs(out_dir, exist_ok=True)
    for old in os.listdir(out_dir):                      # stale outputs from a previous run
        if old.startswith("seg") or old in ("segments.json", "signals.png", "signals.npz", "index.html", "background.png"):
            os.remove(os.path.join(out_dir, old))
    t0 = time.time()
    log(f"[{os.path.basename(out_dir)}] reuse cached flow and frozen splits ...")
    sig = analyze(path, cfg, log)
    segs = split_segments(sig, cfg)
    for s in segs:
        s.keyframes = select_keyframes(sig, s, cfg)
    t_flow = time.time() - t0
    log(f"  frames={sig.n} fps={sig.fps:.2f} size={sig.W}x{sig.H} analysis={sig.aw}x{sig.ah} "
        f"segments={len(segs)}  ({t_flow:.1f}s)")
    wanted = {k for s in segs for k in s.keyframes}
    frames, _ = grab_frames(path, cfg, wanted, [])
    bg = MK.flow_aware_background([], [])
    kd = max(5, int(0.02 * max(sig.W, sig.H)))
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
            log(f"  empty-mask warning seg {s.idx} {s.t_start:.1f}-{s.t_end:.1f}s: weak flow evidence and no RMBG object")
            log("  frozen interval retained; empty-mask warning recorded")
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
