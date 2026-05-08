import numpy as np


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size == 0:
        return x.astype(np.float32)
    pad = window // 2
    xp = np.pad(x, (pad, pad), mode='edge')
    kernel = np.ones(window, dtype=np.float32) / window
    y = np.convolve(xp, kernel, mode='valid')
    return y[:x.shape[0]].astype(np.float32)


def mask_from_segments(times: np.ndarray, segments, label: str = 'KEEP') -> np.ndarray:
    mask = np.zeros((len(times),), dtype=bool)
    for seg in segments or []:
        if seg.get('label', label) != label:
            continue
        s = float(seg.get('start', 0.0))
        e = float(seg.get('end', 0.0))
        if e > s:
            mask |= (times >= s) & (times <= e)
    return mask


def safe_max_proto_score(embs: np.ndarray, protos) -> np.ndarray:
    if protos is None:
        return np.zeros((embs.shape[0],), dtype=np.float32)
    protos = np.asarray(protos, dtype=np.float32)
    if protos.ndim != 2 or protos.shape[0] == 0:
        return np.zeros((embs.shape[0],), dtype=np.float32)
    return (embs.astype(np.float32) @ protos.T).max(axis=1).astype(np.float32)


def make_temporal_stats(times: np.ndarray, embs: np.ndarray, sample_fps: float, windows_sec=(1.0, 3.0, 7.0)) -> np.ndarray:
    """Small temporal descriptors that work on top of fixed frame embeddings."""
    n = embs.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    prev_sim = np.ones((n,), dtype=np.float32)
    next_sim = np.ones((n,), dtype=np.float32)
    if n > 1:
        prev_sim[1:] = np.sum(embs[1:] * embs[:-1], axis=1)
        next_sim[:-1] = np.sum(embs[:-1] * embs[1:], axis=1)
    motion_like = 1.0 - ((prev_sim + next_sim) * 0.5)

    cols = [prev_sim, next_sim, motion_like]
    for sec in windows_sec:
        w = max(1, int(round(float(sec) * max(float(sample_fps), 1e-6))))
        cols.append(moving_average(motion_like, w))

    # Normalized timeline position helps the model learn intro/outro bias without depending on absolute duration.
    if len(times) > 1 and float(times[-1]) > float(times[0]):
        pos = (times.astype(np.float32) - float(times[0])) / max(float(times[-1] - times[0]), 1e-6)
    else:
        pos = np.zeros((n,), dtype=np.float32)
    cols.append(pos.astype(np.float32))
    return np.stack(cols, axis=1).astype(np.float32)


def make_lgbm_features(embs: np.ndarray, times: np.ndarray, sample_fps: float, profile: dict, source_profile: dict = None) -> np.ndarray:
    """Feature layout is intentionally stable and stored as feature_version=1."""
    embs = embs.astype(np.float32)
    global_protos = profile.get('prototypes')
    src_protos = None if source_profile is None else source_profile.get('prototypes')
    global_score = safe_max_proto_score(embs, global_protos).reshape(-1, 1)
    source_score = safe_max_proto_score(embs, src_protos).reshape(-1, 1)

    if 'global_mean' in profile:
        gm = np.asarray(profile['global_mean'], dtype=np.float32).reshape(1, -1)
        mean_score = (embs @ gm.T).astype(np.float32)
    else:
        mean_score = np.zeros((embs.shape[0], 1), dtype=np.float32)

    temporal = make_temporal_stats(times, embs, sample_fps)
    return np.concatenate([embs, global_score, source_score, mean_score, temporal], axis=1).astype(np.float32)
