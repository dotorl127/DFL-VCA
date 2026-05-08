import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from model_utils import (
    list_videos,
    load_dinov2_small,
    build_transform,
    extract_video_embeddings,
    get_video_duration,
    l2_normalize,
)


def canonical_stem(x) -> str:
    """Strict filename key for pairing.

    Important:
    - If x is a Path, use x.stem once.
    - If x is already a stem string, do not call Path(x).stem again.
      Otherwise names like 'hhd800.com@FAYS-012' become only 'hhd800'.
    """
    if isinstance(x, Path):
        s = x.stem
    else:
        s = str(x)
    s = s.lower()
    s = re.sub(r"\s+", "", s)
    return s


def series_base_key(x) -> str:
    """Base key for split raw videos such as *_1, *_2, *_3.

    Examples:
      hhd800.com@FC2-PPV-4894253_1 -> hhd800.com@fc2-ppv-4894253
      hhd800.com@FC2-PPV-4894253_2 -> hhd800.com@fc2-ppv-4894253

    This is used only when --pair_series_parts is enabled.
    """
    s = canonical_stem(x)
    return re.sub(r'[_-]\d+$', '', s)


def is_same_or_variant(raw_path, edit_path) -> bool:
    """Match only when edit filename contains the full raw filename stem.

    Examples:
      raw:  hhd800.com@FAYS-012
      edit: hhd800.com@FAYS-012       -> True
      edit: hhd800.com@FAYS-012_1     -> True
      edit: hhd800.com@FAYS-012-cut   -> True

    But:
      raw:  hhd800.com@FAYS-012
      edit: hhd800.com@FAYS-0123      -> False
      edit: hhd800.com@FAYS-013       -> False
    """
    raw_stem = canonical_stem(raw_path)
    edit_stem = canonical_stem(edit_path)

    pos = edit_stem.find(raw_stem)
    if pos < 0:
        return False

    before_ok = pos == 0 or edit_stem[pos - 1] in "_- .+[]()"
    end = pos + len(raw_stem)
    after_ok = end == len(edit_stem) or edit_stem[end] in "_- .+[]()"
    return before_ok and after_ok


def group_pairs(raw_videos, edited_videos, pair_series_parts=False):
    """Return [(raw_path, [edited_path...]), ...].

    Primary rule:
      An edited video belongs to a raw video when the edited filename contains
      the full raw filename stem.

    Optional series rule:
      If --pair_series_parts is enabled, raw videos ending with part suffixes
      such as _1, _2, _3 share the same series base. An edit matching one part
      is also assigned to other raw videos with the same base.

      Example:
        raw:  XXX_1.mp4, XXX_2.mp4
        edit: XXX_1.mp4
        -> edit is assigned to both XXX_1 and XXX_2.
    """
    edited_sorted = sorted(edited_videos, key=canonical_stem)
    raw_sorted = sorted(raw_videos, key=canonical_stem)
    pairs = {rv: [] for rv in raw_sorted}
    unmatched = []
    multi_source = []

    for ev in edited_sorted:
        candidates = []

        # 1) Strict full-stem match.
        for rv in raw_sorted:
            if is_same_or_variant(rv, ev):
                candidates.append(rv)

        # 2) Optional multi-part source support.
        # If raw videos are split as *_1, *_2 and one edit is named after *_1,
        # attach that edit to all raw parts sharing the same base key.
        if pair_series_parts:
            edit_base = series_base_key(ev)
            if edit_base:
                for rv in raw_sorted:
                    if rv not in candidates and series_base_key(rv) == edit_base:
                        candidates.append(rv)

        if not candidates:
            unmatched.append(ev)
            continue

        if len(candidates) > 1:
            multi_source.append({'edited': str(ev), 'raws': [str(rv) for rv in candidates]})

        for rv in candidates:
            pairs[rv].append(ev)

    grouped = [
        (rv, sorted(evs, key=canonical_stem))
        for rv, evs in sorted(pairs.items(), key=lambda x: canonical_stem(x[0]))
        if evs
    ]
    return grouped, unmatched, multi_source


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size == 0:
        return x.astype(np.float32)
    pad = window // 2
    xp = np.pad(x, (pad, pad), mode='edge')
    kernel = np.ones(window, dtype=np.float32) / window
    y = np.convolve(xp, kernel, mode='valid')
    return y[:x.shape[0]].astype(np.float32)


def max_cosine_score(query_embs: np.ndarray, ref_embs: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """For each query embedding, return max cosine similarity to ref embeddings."""
    out = np.empty((query_embs.shape[0],), dtype=np.float32)
    ref_t = ref_embs.T.astype(np.float32)
    for i in range(0, query_embs.shape[0], chunk):
        q = query_embs[i:i + chunk].astype(np.float32)
        out[i:i + chunk] = (q @ ref_t).max(axis=1)
    return out


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
        if merged and seg['start'] - merged[-1]['end'] <= merge_gap:
            prev = merged[-1]
            len1 = prev['end'] - prev['start']
            len2 = seg['end'] - seg['start']
            if len1 + len2 > 0:
                prev['score'] = float((prev['score'] * len1 + seg['score'] * len2) / (len1 + len2))
            prev['end'] = seg['end']
        else:
            merged.append(seg)
    return merged


def fit_prototypes(embs: np.ndarray, clusters: int, seed: int):
    if embs.shape[0] == 0:
        return np.zeros((0, embs.shape[1] if embs.ndim == 2 else 384), dtype=np.float32)
    n_clusters = min(int(clusters), int(embs.shape[0]))
    if n_clusters <= 1:
        return l2_normalize(embs.mean(axis=0).astype(np.float32))[None, :].astype(np.float32)
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=min(4096, max(256, embs.shape[0])),
        n_init='auto',
        max_iter=200,
    )
    km.fit(embs.astype(np.float32))
    return l2_normalize(km.cluster_centers_.astype(np.float32)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description='Build paired profile from raw videos and their edited versions.')
    ap.add_argument('--raw_dir', required=True, help='Folder containing original/raw videos.')
    ap.add_argument('--edited_dir', required=True, help='Folder containing edited videos. Edited filename must contain raw filename stem.')
    ap.add_argument('--pair_series_parts', action='store_true', help='Also pair split raw parts sharing the same base suffix, e.g. XXX_1 and XXX_2.')
    ap.add_argument('--out', default='paired_profile.pkl')
    ap.add_argument('--sample_fps', type=float, default=2.0)
    ap.add_argument('--input_size', type=int, default=224)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--match_thr', type=float, default=0.78, help='Raw frame is positive if max cosine to paired edited frames >= this threshold.')
    ap.add_argument('--smooth_sec', type=float, default=2.0)
    ap.add_argument('--min_keep_sec', type=float, default=1.0)
    ap.add_argument('--merge_gap_sec', type=float, default=1.0)
    ap.add_argument('--clusters', type=int, default=64, help='Global positive prototypes.')
    ap.add_argument('--source_clusters', type=int, default=32, help='Per-raw source-specific prototypes.')
    ap.add_argument('--max_frames_per_video', type=int, default=0)
    ap.add_argument('--max_edited_embeddings_per_raw', type=int, default=60000)
    ap.add_argument('--max_positive_embeddings', type=int, default=100000)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    raw_videos = list_videos(args.raw_dir)
    edited_videos = list_videos(args.edited_dir)
    if not raw_videos:
        raise SystemExit(f'No raw videos found: {args.raw_dir}')
    if not edited_videos:
        raise SystemExit(f'No edited videos found: {args.edited_dir}')

    grouped, unmatched, multi_source = group_pairs(raw_videos, edited_videos, args.pair_series_parts)
    if not grouped:
        raise SystemExit('No raw↔edited pairs found. Edited filenames must contain raw filename stem.')

    print(f'[INFO] device={args.device}')
    print(f'[INFO] raw videos={len(raw_videos)} edited videos={len(edited_videos)} paired raw={len(grouped)} unmatched edited={len(unmatched)} multi_source_edits={len(multi_source)}')
    for rv, evs in grouped:
        print(f'[PAIR] {rv.name} <- {len(evs)} edited')
        for ep in evs:
            print(f'  - {ep.name}')
    if unmatched:
        print('[WARN] unmatched edited videos:')
        for ev in unmatched[:20]:
            print(f'  - {ev.name}')
        if len(unmatched) > 20:
            print(f'  ... {len(unmatched) - 20} more')
    if multi_source:
        print('[INFO] multi-source edited videos detected:')
        for item in multi_source[:20]:
            raws = ', '.join(Path(r).name for r in item['raws'])
            print(f'  - {Path(item["edited"]).name} <- {raws}')
        if len(multi_source) > 20:
            print(f'  ... {len(multi_source) - 20} more')

    rng = np.random.default_rng(args.seed)
    model = load_dinov2_small(args.device)
    tf = build_transform(args.input_size)

    source_profiles = []
    all_positive = []

    for raw_path, edited_paths in grouped:
        print(f'\n[RAW] {raw_path.name}')
        raw_times, raw_embs = extract_video_embeddings(
            str(raw_path), model, tf, args.device,
            sample_fps=args.sample_fps,
            batch_size=args.batch_size,
            max_frames=args.max_frames_per_video,
            desc=f'raw:{raw_path.name[:16]}',
        )
        if raw_embs.shape[0] == 0:
            print(f'[WARN] no raw embeddings: {raw_path}')
            continue

        edit_embs_list = []
        edit_stats = []
        for ep in edited_paths:
            _, eemb = extract_video_embeddings(
                str(ep), model, tf, args.device,
                sample_fps=args.sample_fps,
                batch_size=args.batch_size,
                max_frames=args.max_frames_per_video,
                desc=f'edit:{ep.name[:16]}',
            )
            if eemb.shape[0] == 0:
                print(f'[WARN] no edited embeddings: {ep}')
                continue
            edit_embs_list.append(eemb)
            edit_stats.append({'path': str(ep), 'frames': int(eemb.shape[0])})

        if not edit_embs_list:
            continue
        edit_embs = np.concatenate(edit_embs_list, axis=0).astype(np.float32)
        if args.max_edited_embeddings_per_raw and edit_embs.shape[0] > args.max_edited_embeddings_per_raw:
            idx = rng.choice(edit_embs.shape[0], size=args.max_edited_embeddings_per_raw, replace=False)
            edit_embs = edit_embs[idx]

        raw_score = max_cosine_score(raw_embs, edit_embs)
        smooth_n = max(1, int(round(args.smooth_sec * args.sample_fps)))
        raw_score_smooth = moving_average(raw_score, smooth_n)
        pos_mask = raw_score_smooth >= args.match_thr

        duration = get_video_duration(str(raw_path))
        half_interval = 0.5 / max(args.sample_fps, 1e-6)
        segments = make_segments(
            raw_times, raw_score_smooth, pos_mask, 'KEEP', half_interval,
            args.min_keep_sec, args.merge_gap_sec, duration,
        )
        for i, seg in enumerate(segments, 1):
            seg['id'] = i
            seg['start'] = round(float(seg['start']), 3)
            seg['end'] = round(float(seg['end']), 3)
            seg['score'] = round(float(seg['score']), 5)

        pos_embs = raw_embs[pos_mask]
        if pos_embs.shape[0] == 0:
            print(f'[WARN] no positives above match_thr={args.match_thr}: {raw_path.name}. Try lower threshold.')
            src_protos = np.zeros((0, raw_embs.shape[1]), dtype=np.float32)
        else:
            src_protos = fit_prototypes(pos_embs, args.source_clusters, args.seed)
            all_positive.append(pos_embs)

        print(f'[INFO] raw_frames={raw_embs.shape[0]} edited_frames={edit_embs.shape[0]} positives={int(pos_mask.sum())} segments={len(segments)} score_mean={raw_score_smooth.mean():.3f} score_max={raw_score_smooth.max():.3f}')

        source_profiles.append({
            'raw_path': str(raw_path),
            'raw_name': raw_path.name,
            'raw_stem': raw_path.stem,
            'raw_key': normalize_name(raw_path.stem),
            'edited': edit_stats,
            'segments': segments,
            'prototypes': src_protos.astype(np.float32),
            'score_stats': {
                'mean': float(raw_score_smooth.mean()),
                'p50': float(np.percentile(raw_score_smooth, 50)),
                'p90': float(np.percentile(raw_score_smooth, 90)),
                'p95': float(np.percentile(raw_score_smooth, 95)),
                'p99': float(np.percentile(raw_score_smooth, 99)),
                'max': float(raw_score_smooth.max()),
            },
            'positive_frames': int(pos_mask.sum()),
            'raw_frames': int(raw_embs.shape[0]),
        })

    if not all_positive:
        raise SystemExit('No positive frames found. Lower --match_thr or increase --sample_fps.')

    positive_embs = np.concatenate(all_positive, axis=0).astype(np.float32)
    if args.max_positive_embeddings and positive_embs.shape[0] > args.max_positive_embeddings:
        idx = rng.choice(positive_embs.shape[0], size=args.max_positive_embeddings, replace=False)
        positive_for_km = positive_embs[idx]
    else:
        positive_for_km = positive_embs

    global_protos = fit_prototypes(positive_for_km, args.clusters, args.seed)
    global_mean = l2_normalize(positive_embs.mean(axis=0).astype(np.float32))[0]

    profile = {
        'profile_type': 'paired_raw_edited',
        'model': 'dinov2_vits14',
        'embedding_dim': int(positive_embs.shape[1]),
        'sample_fps': float(args.sample_fps),
        'input_size': int(args.input_size),
        'match_thr': float(args.match_thr),
        'smooth_sec': float(args.smooth_sec),
        'min_keep_sec': float(args.min_keep_sec),
        'merge_gap_sec': float(args.merge_gap_sec),
        'clusters': int(global_protos.shape[0]),
        'prototypes': global_protos.astype(np.float32),
        'global_mean': global_mean.astype(np.float32),
        'source_profiles': source_profiles,
        'total_positive_embeddings': int(positive_embs.shape[0]),
        'multi_source_edits': multi_source,
        'generated_at': datetime.now().isoformat(timespec='seconds'),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(profile, out)

    def strip_arrays(obj):
        if isinstance(obj, np.ndarray):
            return {'array_shape': list(obj.shape), 'dtype': str(obj.dtype)}
        if isinstance(obj, dict):
            return {k: strip_arrays(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [strip_arrays(v) for v in obj]
        return obj

    out.with_suffix('.json').write_text(json.dumps(strip_arrays(profile), ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n[DONE] saved paired profile: {out}')
    print(f'[DONE] saved metadata: {out.with_suffix(".json")}')


if __name__ == '__main__':
    main()
