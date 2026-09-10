import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
# torch must be imported before faiss: on macOS, loading faiss's bundled
# OpenMP runtime first corrupts torch's own thread pool init and segfaults
# on the first model forward pass (see write_faiss_index.py). That import
# order isn't enough on its own here though: index.search() (unlike the
# add()-only path write_faiss_index.py uses) spins up faiss's own OpenMP
# thread pool, which segfaults if a torch forward pass already ran in this
# process — so pin faiss to single-threaded search too.
import torch
import faiss
faiss.omp_set_num_threads(1)

from models.configs import get_model_config
from models.frame_extractor import frames_dir_for_video, get_video_fps
from models.write_faiss_index import faiss_index_path

FRAME_IDX_RE = re.compile(r"(\d+)")


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def id_map_path(video_path: str, video_name: str) -> str:
    video_dir = os.path.dirname(os.path.abspath(video_path))
    return os.path.join(video_dir, f"{video_name}_id_map.json")


def load_id_map(video_path: str, video_name: str) -> Dict[int, Dict[str, str]]:
    with open(id_map_path(video_path, video_name), "r") as f:
        entries = json.load(f)
    return {entry["faiss_id"]: entry for entry in entries}


def frame_idx_from_label(label: str) -> Optional[int]:
    match = FRAME_IDX_RE.search(label)
    return int(match.group(1)) if match else None


def embed_query(
    query: str,
    model_family: str = "clip",
    model_id: Optional[str] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Encode `query` with the same VLM family used to build the FAISS index,
    L2-normalized so its inner product with the (also normalized) index
    vectors is cosine similarity.
    """
    if device is None:
        device = get_device()

    model_config = get_model_config(model_family, model_id)
    processor = model_config["processor_class"].from_pretrained(model_config["model_id"])
    model = model_config["model_class"].from_pretrained(model_config["model_id"])
    wrapper = model_config["wrapper_class"](model=model, processor=processor)

    model.to(device)
    model.eval()

    inputs = wrapper.process_inputs(text=[query])
    with torch.no_grad():
        embeds = wrapper.get_text_embeddings(inputs)

    vector = embeds.cpu().numpy().astype("float32")
    faiss.normalize_L2(vector)
    return vector


def search_frames(
    video_path: str,
    video_name: str,
    query: str,
    k: int = 10,
    model_family: str = "clip",
    model_id: Optional[str] = None,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Embed `query` and search `<video_name>.faiss` (built by
    write_faiss_index.build_index_for_frames) for the `k` most similar
    shot-boundary frames. Returns each match's frame path, frame index,
    playback time (seconds) and cosine-similarity score, ranked best first.
    """
    index = faiss.read_index(faiss_index_path(video_path, video_name))
    id_map = load_id_map(video_path, video_name)

    k = min(k, index.ntotal)
    if k == 0:
        return []

    query_vector = embed_query(query, model_family, model_id, device)
    scores, ids = index.search(query_vector, k)

    fps = get_video_fps(video_path)
    frames_dir = frames_dir_for_video(video_path)

    results = []
    for score, faiss_id in zip(scores[0].tolist(), ids[0].tolist()):
        entry = id_map.get(faiss_id)
        if entry is None:
            continue
        label = entry["label"]
        frame_idx = frame_idx_from_label(label)
        time = (frame_idx / fps) if (frame_idx is not None and fps) else None
        results.append({
            "label": label,
            "path": os.path.join(frames_dir, label),
            "frame_idx": frame_idx,
            "time": time,
            "score": score,
        })
    return results
