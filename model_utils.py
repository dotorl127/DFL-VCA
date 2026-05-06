from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

VIDEO_EXTS = {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v'}


def list_videos(path: str) -> List[Path]:
    p = Path(path)
    if p.is_file():
        return [p] if p.suffix.lower() in VIDEO_EXTS else []
    videos = []
    for f in p.rglob('*'):
        if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
            videos.append(f)
    return sorted(videos)


def get_video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    cap.release()
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    return {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration': duration,
    }


def get_video_duration(video_path: str) -> float:
    return float(get_video_info(video_path)['duration'])


def iter_video_frames(video_path: str, sample_fps: float = 1.0, max_frames: int = 0):
    """Yield (time_seconds, PIL.Image) without writing frames to disk."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    interval = max(int(round(src_fps / sample_fps)), 1)
    frame_idx = 0
    yielded = 0

    while True:
        ok = cap.grab()
        if not ok:
            break
        if frame_idx % interval == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame_idx / src_fps, Image.fromarray(frame_rgb)
            yielded += 1
            if max_frames and yielded >= max_frames:
                break
        frame_idx += 1

    cap.release()


def load_dinov2_small(device: str):
    # First run downloads the model through torch.hub.
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval().to(device)
    return model


def build_transform(input_size: int = 224):
    return transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(1, -1)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


@torch.inference_mode()
def extract_video_embeddings(
    video_path: str,
    model,
    transform,
    device: str,
    sample_fps: float = 1.0,
    batch_size: int = 64,
    max_frames: int = 0,
    desc: str = None,
) -> Tuple[np.ndarray, np.ndarray]:
    times = []
    feats = []
    batch_imgs = []
    batch_times = []

    info = get_video_info(video_path)
    total_est = int(info['duration'] * sample_fps) if info['duration'] > 0 and sample_fps > 0 else None
    if max_frames and total_est:
        total_est = min(total_est, max_frames)

    iterator = iter_video_frames(video_path, sample_fps=sample_fps, max_frames=max_frames)
    pbar = tqdm(iterator, total=total_est, desc=desc or Path(video_path), unit='frm')

    def flush():
        nonlocal batch_imgs, batch_times
        if not batch_imgs:
            return
        x = torch.stack([transform(img) for img in batch_imgs]).to(device, non_blocking=True)
        y = model(x)
        if isinstance(y, dict):
            y = y.get('x_norm_clstoken', next(iter(y.values())))
        y = y.detach().float().cpu().numpy()
        feats.append(y)
        times.extend(batch_times)
        batch_imgs = []
        batch_times = []

    for t, img in pbar:
        batch_imgs.append(img)
        batch_times.append(float(t))
        if len(batch_imgs) >= batch_size:
            flush()
    flush()
    pbar.close()

    if not feats:
        return np.zeros((0,), dtype=np.float32), np.zeros((0, 384), dtype=np.float32)

    emb = np.concatenate(feats, axis=0).astype(np.float32)
    emb = l2_normalize(emb).astype(np.float32)
    return np.array(times, dtype=np.float32), emb
