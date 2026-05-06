import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from model_utils import list_videos, load_dinov2_small, build_transform, extract_video_embeddings, l2_normalize


def main():
    ap = argparse.ArgumentParser(description='Build style profile from edited videos using DINOv2-small embeddings.')
    ap.add_argument('--edited_dir', required=True, help='Folder containing edited videos, or one edited video file.')
    ap.add_argument('--out', default='style_profile.pkl', help='Output profile path.')
    ap.add_argument('--sample_fps', type=float, default=1.0)
    ap.add_argument('--input_size', type=int, default=224)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--clusters', type=int, default=64, help='Number of style prototypes.')
    ap.add_argument('--max_frames_per_video', type=int, default=0, help='0 = no limit.')
    ap.add_argument('--max_total_embeddings', type=int, default=80000, help='Subsample embeddings before KMeans if larger. 0 = no limit.')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    videos = list_videos(args.edited_dir)
    if not videos:
        raise SystemExit(f'No videos found: {args.edited_dir}')

    print(f'[INFO] device={args.device}')
    print(f'[INFO] edited videos={len(videos)}')

    model = load_dinov2_small(args.device)
    tf = build_transform(args.input_size)

    all_embs = []
    video_stats = []
    for vp in videos:
        _, emb = extract_video_embeddings(
            str(vp), model, tf, args.device,
            sample_fps=args.sample_fps,
            batch_size=args.batch_size,
            max_frames=args.max_frames_per_video,
            desc=f'profile:{vp.name}',
        )
        if emb.shape[0] == 0:
            print(f'[WARN] no frames: {vp}')
            continue
        all_embs.append(emb)
        video_stats.append({'path': str(vp), 'frames': int(emb.shape[0])})
        print(f'[INFO] {vp.name}: embeddings={emb.shape}')

    if not all_embs:
        raise SystemExit('No embeddings extracted.')

    embs = np.concatenate(all_embs, axis=0).astype(np.float32)
    rng = np.random.default_rng(args.seed)
    if args.max_total_embeddings and embs.shape[0] > args.max_total_embeddings:
        idx = rng.choice(embs.shape[0], size=args.max_total_embeddings, replace=False)
        embs_for_kmeans = embs[idx]
    else:
        embs_for_kmeans = embs

    n_clusters = min(args.clusters, embs_for_kmeans.shape[0])
    print(f'[INFO] KMeans embeddings={embs_for_kmeans.shape[0]} clusters={n_clusters}')
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=args.seed,
        batch_size=4096,
        n_init='auto',
        max_iter=200,
    )
    kmeans.fit(embs_for_kmeans)

    prototypes = l2_normalize(kmeans.cluster_centers_.astype(np.float32)).astype(np.float32)
    global_mean = l2_normalize(embs.mean(axis=0).astype(np.float32))[0]

    profile = {
        'model': 'dinov2_vits14',
        'embedding_dim': int(embs.shape[1]),
        'sample_fps': float(args.sample_fps),
        'input_size': int(args.input_size),
        'clusters': int(n_clusters),
        'prototypes': prototypes,
        'global_mean': global_mean.astype(np.float32),
        'video_stats': video_stats,
        'total_embeddings': int(embs.shape[0]),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(profile, out)

    meta = {k: v for k, v in profile.items() if k not in ('prototypes', 'global_mean')}
    out.with_suffix('.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[DONE] saved profile: {out}')
    print(f'[DONE] saved metadata: {out.with_suffix(".json")}')


if __name__ == '__main__':
    main()
