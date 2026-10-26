"""Match local videos by source hash and export OpenCV frame-count metadata.

Requires the generation environment (opencv-python). No generation or API call.
This reads container metadata; it does not decode and count every frame.
"""
import argparse
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from afterimage.common import check, read, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--videos', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    check(args.videos.is_dir(), 'Video directory missing')
    check(not args.out.exists(), 'Output exists; choose a new file')
    import cv2
    videos = read(args.inputs/'manifest.json')['videos']
    wanted = {v['source_sha256']: v for v in videos}
    check(len(wanted) == len(videos), 'Duplicate source hash in manifest')
    found = {}
    for path in sorted(args.videos.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in ('.avi', '.mp4', '.mov', '.mkv', '.mpg', '.mpeg', '.wmv'):
            continue
        digest = sha(path)
        if digest not in wanted:
            continue
        check(digest not in found, 'Duplicate matching source video')
        cap = cv2.VideoCapture(str(path))
        n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        check(n > 0 and fps > 0, 'Invalid container frame metadata')
        v = wanted[digest]
        found[digest] = dict(video_id=v['video_id'], source_sha256=digest, N_original_frames=n,
                             fps=fps, basis='OpenCV container metadata; no full decode audit')
    check(set(found) == set(wanted), 'Not all manifest source videos were found')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', newline='', encoding='utf-8-sig') as stream:
        rows = list(found.values())
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f'Exported metadata for {len(found)} source-hash-matched videos; no API calls.')


if __name__ == '__main__':
    main()
