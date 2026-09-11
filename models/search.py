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
import torch.nn.functional as F
import faiss
faiss.omp_set_num_threads(1)

from models.configs import load_vlm_wrapper
from models.frame_extractor import frames_dir_for_video, get_video_fps
from models.relevance_feedback import ImageEmbeddingRelevanceFeedback, RocchioUpdate
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

    # Cached across requests — see load_vlm_wrapper() — so repeated searches
    # don't pay from_pretrained()'s cost (a few seconds) every time.
    wrapper = load_vlm_wrapper(model_family, model_id, device)

    inputs = wrapper.process_inputs(text=[query])
    with torch.no_grad():
        embeds = wrapper.get_text_embeddings(inputs)

    vector = embeds.cpu().numpy().astype("float32")
    faiss.normalize_L2(vector)
    return vector


def frame_path_for_label(video_path: str, label: str) -> str:
    return os.path.join(frames_dir_for_video(video_path), label)


def _search_index(
    video_path: str,
    video_name: str,
    query_vector: np.ndarray,
    k: int,
) -> List[Dict[str, Any]]:
    """Search `<video_name>.faiss` with an already-embedded, L2-normalized
    `query_vector` for the `k` most similar shot-boundary frames. Returns
    each match's frame path, frame index, playback time (seconds) and
    cosine-similarity score, ranked best first."""
    index = faiss.read_index(faiss_index_path(video_path, video_name))
    id_map = load_id_map(video_path, video_name)

    k = min(k, index.ntotal)
    if k == 0:
        return []

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
    query_vector = embed_query(query, model_family, model_id, device)
    return _search_index(video_path, video_name, query_vector, k)


def feedback_search_frames(
    video_path: str,
    video_name: str,
    query: str,
    positive_labels: List[str],
    negative_labels: List[str],
    k: int = 10,
    model_family: str = "clip",
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    alpha: float = 0.8,
    beta: float = 0.3,
    gamma: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Re-run the search after relevance feedback on a previous result set:
    fold the average image embedding of `negative_labels` (frames the user
    selected as not what they wanted) away from the query, and of
    `positive_labels` (the rest of that result set, left unselected) toward
    it, via Rocchio's algorithm — then re-rank the whole FAISS index
    against that updated query. Same return shape as search_frames.
    """
    if device is None:
        device = get_device()

    wrapper = load_vlm_wrapper(model_family, model_id, device)

    text_inputs = wrapper.process_inputs(text=[query])
    with torch.no_grad():
        query_embeds = wrapper.get_text_embeddings(text_inputs)
    query_embeds = F.normalize(query_embeds, p=2, dim=-1)

    feedback = ImageEmbeddingRelevanceFeedback(vlm_wrapper_retrieval=wrapper)
    feedback_embeds = feedback(
        query=query,
        positive_image_paths=[frame_path_for_label(video_path, label) for label in positive_labels],
        negative_image_paths=[frame_path_for_label(video_path, label) for label in negative_labels],
    )

    rocchio_update = RocchioUpdate(alpha=alpha, beta=beta, gamma=gamma)
    updated_query_embeds = rocchio_update(
        query_embeddings=query_embeds,
        positive_embeddings=feedback_embeds["positive"],
        negative_embeddings=feedback_embeds["negative"],
    )

    query_vector = updated_query_embeds.cpu().numpy().astype("float32")
    faiss.normalize_L2(query_vector)

    return _search_index(video_path, video_name, query_vector, k)
