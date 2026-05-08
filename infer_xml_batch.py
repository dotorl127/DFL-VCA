import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch
from tqdm import tqdm

from model_utils import list_videos, load_dinov2_small, build_transform, extract_video_embeddings, get_video_duration
from fcp_xml import write_fcp7_xml
from lgbm_utils import make_lgbm_features


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size == 0:
        return x.astype(np.float32)
    pad = window // 2
    xp = np.pad(x, (pad, pad), mode='edge')
    kernel = np.ones(window, dtype=np.float32) / window
    y = np.convolve(xp, kernel, mode='valid')
    return y[:x.shape[0]].astype(np.float32)


def seconds_to_timecode_like(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f'{h:02d}:{m:02d}:{s:06.3f}'




def normalize_name(s: str) -> str:
    import re
    s = Path(s).stem.lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[_\-\.\[\]\(\)]+", "", s)
    return s


def get_base_profile(profile):
    """LightGBM profiles wrap the paired profile used to create labels/prototypes."""
    if profile.get('profile_type') == 'lightgbm_keep_model':
        return profile.get('base_profile', profile)
    return profile


def find_source_profile(profile, video_path):
    """Find per-source paired profile for this raw video, if available."""
    profile = get_base_profile(profile)
    if profile.get('profile_type') != 'paired_raw_edited':
        return None
    vk = normalize_name(Path(video_path).stem)
    best = None
    best_len = -1
    for sp in profile.get('source_profiles', []):
        rk = sp.get('raw_key') or normalize_name(sp.get('raw_stem', ''))
        if rk and (rk == vk or rk in vk or vk in rk):
            if len(rk) > best_len:
                best = sp
                best_len = len(rk)
    return best

def make_segments(times, scores, mask, label, half_interval, min_len, merge_gap, duration):
    segs = []
    in_seg = False
    start = 0.0
    vals = []
    for t, s, m in zip(times, scores, mask):
        t = float(t)
        if m and not in_seg:
            in_seg = True
            start = max(0.0, t - half_interval)
            vals = [float(s)]
        elif m and in_seg:
            vals.append(float(s))
        elif (not m) and in_seg:
            end = min(duration, t - half_interval)
            if end - start >= min_len:
                segs.append({'start': start, 'end': end, 'label': label, 'score': float(np.mean(vals))})
            in_seg = False
            vals = []
    if in_seg and len(times) > 0:
        end = min(duration, float(times[-1]) + half_interval)
        if end - start >= min_len:
            segs.append({'start': start, 'end': end, 'label': label, 'score': float(np.mean(vals))})

    merged = []
    for seg in segs:
        if merged and seg['label'] == merged[-1]['label'] and seg['start'] - merged[-1]['end'] <= merge_gap:
            prev = merged[-1]
            len1 = prev['end'] - prev['start']
            len2 = seg['end'] - seg['start']
            if len1 + len2 > 0:
                prev['score'] = float((prev['score'] * len1 + seg['score'] * len2) / (len1 + len2))
            prev['end'] = seg['end']
        else:
            merged.append(seg)
    return merged


def infer_one(video_path, model, tf, profile, args, out_xml_path: Path):
    base_profile = get_base_profile(profile)
    profile_type = profile.get('profile_type', '')
    sample_fps = args.sample_fps if args.sample_fps > 0 else float(profile.get('sample_fps', base_profile.get('sample_fps', 1.0)))
    source_profile = find_source_profile(base_profile, video_path)

    if args.use_cached_paired_segments and source_profile and source_profile.get('segments'):
        markers = [dict(m) for m in source_profile['segments']]
        for i, m in enumerate(markers, 1):
            m['id'] = i
            m['label'] = m.get('label', 'KEEP')
            m['start'] = round(float(m['start']), 3)
            m['end'] = round(float(m['end']), 3)
            m['score'] = round(float(m.get('score', 1.0)), 5)
            m['start_tc'] = seconds_to_timecode_like(m['start'])
            m['end_tc'] = seconds_to_timecode_like(m['end'])
        info = write_fcp7_xml(
            str(out_xml_path),
            str(video_path),
            markers,
            sequence_name=Path(video_path).stem + '_PAIRED_CACHED_SPLIT',
        )
        print(f'[DONE] {Path(video_path).name}: cached paired markers={len(markers)} xml={out_xml_path}')
        return info

    times, embs = extract_video_embeddings(
        str(video_path), model, tf, args.device,
        sample_fps=sample_fps,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
        desc=f'infer:{Path(video_path).name[:10]}...',
    )
    if embs.shape[0] == 0:
        print(f'[WARN] no embeddings: {video_path}')
        return None

    if profile_type == 'lightgbm_keep_model':
        clf = profile.get('lgbm_model')
        if clf is None:
            raise RuntimeError('LightGBM profile does not contain lgbm_model')
        feats = make_lgbm_features(embs, times, sample_fps, base_profile, source_profile)
        scores = clf.predict_proba(feats)[:, 1].astype(np.float32)
        profile_note = 'LightGBM KEEP model'
    else:
        if source_profile is not None and 'prototypes' in source_profile and np.asarray(source_profile['prototypes']).size > 0:
            protos = np.asarray(source_profile['prototypes'], dtype=np.float32)
            profile_note = 'source-specific paired prototypes'
        else:
            protos = base_profile['prototypes'].astype(np.float32)
            profile_note = 'global prototypes'
        proto_score = (embs @ protos.T).max(axis=1)

        mean_weight = float(args.mean_weight)
        if mean_weight > 0 and 'global_mean' in base_profile:
            gm = base_profile['global_mean'].astype(np.float32).reshape(1, -1)
            mean_score = (embs @ gm.T).reshape(-1)
            scores = (1.0 - mean_weight) * proto_score + mean_weight * mean_score
        else:
            scores = proto_score

    smooth_n = max(1, int(round(args.smooth_sec * sample_fps)))
    scores_smooth = moving_average(scores.astype(np.float32), smooth_n)
    duration = get_video_duration(str(video_path))
    half_interval = 0.5 / max(sample_fps, 1e-6)

    keep_mask = scores_smooth >= args.keep_thr
    review_mask = (scores_smooth >= args.review_thr) & (scores_smooth < args.keep_thr)
    keep_segments = make_segments(times, scores_smooth, keep_mask, 'KEEP', half_interval, args.min_keep_sec, args.merge_gap_sec, duration)
    review_segments = make_segments(times, scores_smooth, review_mask, 'REVIEW', half_interval, args.min_keep_sec, args.merge_gap_sec, duration)

    markers = sorted(keep_segments + review_segments, key=lambda x: (x['start'], x['label']))
    for i, m in enumerate(markers, 1):
        m['id'] = i
        m['start'] = round(float(m['start']), 3)
        m['end'] = round(float(m['end']), 3)
        m['score'] = round(float(m['score']), 5)
        m['start_tc'] = seconds_to_timecode_like(m['start'])
        m['end_tc'] = seconds_to_timecode_like(m['end'])

    info = write_fcp7_xml(
        str(out_xml_path),
        str(video_path),
        markers,
        sequence_name=Path(video_path).stem + '_AI_MARKED_SPLIT',
    )

    print(f'[DONE] {Path(video_path).name}: markers={len(markers)} KEEP={len(keep_segments)} REVIEW={len(review_segments)} profile={profile_note} xml={out_xml_path}')
    return info


def main():
    ap = argparse.ArgumentParser(description='Analyze a directory of raw videos and export one FCP7 XML per video for Premiere Pro import.')
    ap.add_argument('--input_dir', required=True, help='Folder containing raw/original videos.')
    ap.add_argument('--profile', default='lgbm_profile.pkl')
    ap.add_argument('--out_dir', default='', help='Default: sibling folder named <input_dir>_xml')
    ap.add_argument('--sample_fps', type=float, default=0.0, help='0 = use sample_fps saved in profile.')
    ap.add_argument('--input_size', type=int, default=0, help='0 = use input_size saved in profile.')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--keep_thr', type=float, default=0.74)
    ap.add_argument('--review_thr', type=float, default=0.68)
    ap.add_argument('--smooth_sec', type=float, default=7.0)
    ap.add_argument('--min_keep_sec', type=float, default=3.0)
    ap.add_argument('--merge_gap_sec', type=float, default=2.0)
    ap.add_argument('--mean_weight', type=float, default=0.10)
    ap.add_argument('--use_cached_paired_segments', action='store_true', help='If profile was built by build_paired_profile.py and this raw video exists in it, write the already matched paired segments directly.')
    ap.add_argument('--max_frames', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    videos = list_videos(str(input_dir))
    if not videos:
        raise SystemExit(f'No videos found: {input_dir}')

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = input_dir.parent / f'{input_dir.name}_xml'
    out_dir.mkdir(parents=True, exist_ok=True)

    profile = joblib.load(args.profile)
    input_size = args.input_size if args.input_size > 0 else int(profile.get('input_size', 224))
    sample_fps = args.sample_fps if args.sample_fps > 0 else float(profile.get('sample_fps', 1.0))

    print(f'[INFO] device={args.device}')
    print(f'[INFO] input_dir={input_dir}')
    print(f'[INFO] out_dir={out_dir}')
    print(f'[INFO] videos={len(videos)}')
    print(f'[INFO] sample_fps={sample_fps} input_size={input_size}')

    model = load_dinov2_small(args.device)
    tf = build_transform(input_size)

    results = []
    for vp in videos:
        rel = vp.relative_to(input_dir) if input_dir.is_dir() else Path(vp.name)
        out_xml = (out_dir / rel).with_suffix('.xml')
        out_xml.parent.mkdir(parents=True, exist_ok=True)
        info = infer_one(vp, model, tf, profile, args, out_xml)
        if info:
            results.append(info)

    summary = {
        'input_dir': str(input_dir),
        'out_dir': str(out_dir),
        'profile': str(Path(args.profile).resolve()),
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'count': len(results),
        'results': results,
    }
    print(f'[DONE] wrote {len(results)} XML files to: {out_dir}')


if __name__ == '__main__':
    main()
